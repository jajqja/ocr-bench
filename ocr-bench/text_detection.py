import collections
import copy
import json
from typing import Optional

import click

from utils.bbox import get_pdf_lines
from utils.metrics import precision_recall, calculate_iou_metrics
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
from surya.debug.draw import draw_polys_on_image
from surya.common.util import rescale_bbox
from surya.settings import settings
from surya.detection import DetectionPredictor

import os
import time
from tabulate import tabulate
import datasets


@click.command(help="Benchmark detection model on PDF dataset.")
@click.option(
    "--pdf_path", type=str, help="Path to PDF to detect bboxes in.", default=None
)
@click.option(
    "--dataset_name", type=str, help="Hugging Face dataset name.", default=None
)
@click.option(
    "--results_dir",
    type=str,
    help="Path to directory for results.",
    default=os.path.join(settings.RESULT_DIR, "detection_benchmark"),
)
@click.option(
    "--max_rows", type=int, help="Maximum number of pages to process.", default=100
)
@click.option("--debug", is_flag=True, help="Enable debug mode.", default=False)
@click.option("--model_path", type=str, required=True, help="Path to detection model")
def main(
    pdf_path: Optional[str],
    dataset_name: Optional[str],
    results_dir: str,
    max_rows: int,
    debug: bool,
    model_path: str,
    hf_token: Optional[str] = None,
):
    """Main benchmark function for detection."""
    det_predictor = DetectionPredictor(checkpoint=model_path)

    # Load data
    if pdf_path is not None:
        print(f"Loading PDF: {pdf_path}")
        pathname = os.path.basename(pdf_path).split(".")[0]
        doc = open_pdf(pdf_path)
        page_count = min(len(doc), max_rows)
        page_indices = list(range(page_count))

        images = get_page_images(doc, page_indices)
        doc.close()

        image_sizes = [img.size for img in images]
        correct_boxes = get_pdf_lines(pdf_path, image_sizes)
    elif dataset_name is not None and dataset_name == "vikp/doclaynet_bench":
        print(f"Loading dataset: {dataset_name}")
        pathname = dataset_name
        dataset = datasets.load_dataset(dataset_name, split=f"train[:{max_rows}]")
        images = list(dataset["image"])
        images = convert_if_not_rgb(images)
        correct_boxes = []
        for i, boxes in enumerate(dataset["bboxes"]):
            img_size = images[i].size
            # 1000,1000 is bbox size for doclaynet
            correct_boxes.append(
                [rescale_bbox(b, (1000, 1000), img_size) for b in boxes]
            )
    else:
        raise ValueError("Either pdf_path or dataset_name must be provided")

    print(f"Loaded {len(images)} images")

    if settings.DETECTOR_STATIC_CACHE:
        # Run through one batch to compile the model
        det_predictor(images[:1])

    # Run inference
    print("Running inference...")
    start = time.time()
    predictions = det_predictor(images)
    inference_time = time.time() - start

    folder_name = pathname
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    # Calculate metrics
    print("Calculating metrics...")
    page_metrics = collections.OrderedDict()
    iou_scores = []

    for idx, (sb, cb) in enumerate(zip(predictions, correct_boxes)):
        surya_boxes = [s.bbox for s in sb.bboxes]
        surya_polys = [s.polygon for s in sb.bboxes]

        # Calculate precision and recall
        raw_metrics = precision_recall(surya_boxes, cb)
        surya_metrics = {k: float(v) for k, v in raw_metrics.items()}

        # Calculate IOU score
        iou = calculate_iou_metrics(surya_boxes, cb)
        iou_scores.append(iou)
        surya_metrics["iou"] = iou

        page_metrics[idx] = {
            "surya": surya_metrics,
        }

        if debug:
            bbox_image = draw_polys_on_image(surya_polys, copy.deepcopy(images[idx]))
            bbox_image.save(os.path.join(result_path, f"{idx}_bbox.png"))

    # Calculate mean metrics
    mean_metrics = {}
    metric_types = sorted(page_metrics[0]["surya"].keys())
    models = ["surya"]

    for k in models:
        mean_metrics[k] = {}
        for m in metric_types:
            metric_values = [page_metrics[page][k][m] for page in page_metrics]
            mean_metrics[k][m] = sum(metric_values) / len(metric_values)

    # Save results
    out_data = {
        "dataset": pathname,
        "model": model_path,
        "num_samples": len(images),
        "times": {
            "surya": inference_time,
            "per_sample": inference_time / len(images),
        },
        "metrics": mean_metrics,
        "page_metrics": page_metrics,
    }

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    # Print results
    table_headers = ["Model", "Time (s)", "Time per sample (s)"] + metric_types
    table_data = [
        ["surya", f"{inference_time:.2f}", f"{inference_time / len(images):.4f}"]
        + [f"{mean_metrics['surya'][m]:.4f}" for m in metric_types],
    ]

    print("\n" + "=" * 70)
    print("DETECTION METRICS")
    print("=" * 70)
    print(tabulate(table_data, headers=table_headers, tablefmt="github"))

    print("\nMetric Descriptions:")
    print("  - Precision/Recall: Coverage threshold at 0.5")
    print("  - IOU: Intersection over Union (penalized for multiple overlapping boxes)")
    print(
        f"  - Inference Time: {inference_time:.2f}s total, {inference_time/len(images):.4f}s per sample"
    )

    print(f"\n✓ Results saved to {result_path}")


if __name__ == "__main__":
    main()
