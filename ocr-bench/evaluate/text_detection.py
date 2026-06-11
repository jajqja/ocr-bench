import collections
import copy
import json
import os
import time
import gc
from typing import Optional

import click
from tabulate import tabulate
from tqdm import tqdm

from utils.bbox import draw_bboxes_on_image
from utils.metrics import calculate_iou, precision_recall_f1_coverage
from utils.datasets import (
    load_doclaynet_detection,
    load_nvidia_detection,
    load_pdfa_detection,
    load_pdf_detection,
)
from models.detection import load as load_detection_model


@click.command(help="Benchmark a detection model on a dataset.")
@click.option("--pdf_path", type=str, default=None, help="Path to PDF file.")
@click.option("--dataset_name", type=str, default=None, help="HuggingFace dataset name.")
@click.option(
    "--results_dir", type=str, default="./results/detection_benchmark",
    help="Directory to write results.",
)
@click.option("--max_rows", type=int, default=100, help="Max pages/samples to process.")
@click.option("--debug", is_flag=True, default=False, help="Save bbox visualisation images.")
@click.option(
    "--model", type=str, default="surya",
    help="Detection model name (must be registered in models/detection/__init__.py).",
)
@click.option("--model_path", type=str, required=True, help="Path to model checkpoint.")
@click.option(
    "--language", type=str, default="en",
    help="Language for NVIDIA dataset (en, ja, ko, ru, zh_hans, zh_hant).",
)
@click.option(
    "--h5_files", type=str, default="train_000",
    help="Comma-separated H5 filenames for the NVIDIA dataset.",
)
@click.option("--batch_size", type=int, default=8, help="Inference batch size.")
@click.option(
    "--chunk_size", type=int, default=10000,
    help="Samples per chunk when streaming large datasets.",
)
@click.option(
    "--max_size_limit", type=int, default=None,
    help="Resize images so the longest side does not exceed this value.",
)
def main(
    pdf_path: Optional[str],
    dataset_name: Optional[str],
    results_dir: str,
    max_rows: int,
    debug: bool,
    model: str,
    model_path: str,
    language: str,
    h5_files: str,
    batch_size: int,
    chunk_size: int,
    max_size_limit: Optional[int],
):
    print(f"Loading detection model '{model}' from {model_path}...")
    det_model = load_detection_model(model, checkpoint=model_path)

    if max_size_limit:
        print(f"Max image dimension: {max_size_limit}px")

    # --- Build data generator ---
    if pdf_path is not None:
        pathname = os.path.basename(pdf_path).split(".")[0]
        print(f"Loading PDF: {pdf_path}")
        images, correct_boxes = load_pdf_detection(pdf_path, max_rows)
        data_generator = [(images, correct_boxes)]
    elif dataset_name == "vikp/doclaynet_bench":
        pathname = dataset_name
        print(f"Loading dataset: {dataset_name}")
        images, correct_boxes = load_doclaynet_detection(dataset_name, max_rows)
        data_generator = [(images, correct_boxes)]
    elif dataset_name == "pixparse/pdfa-eng-wds":
        pathname = dataset_name.replace("/", "_")
        print(f"Loading dataset: {dataset_name}")
        images, correct_boxes = load_pdfa_detection(dataset_name, max_rows)
        data_generator = [(images, correct_boxes)]
    elif dataset_name == "nvidia/OCR-Synthetic-Multilingual-v1":
        pathname = f"nvidia_ocr_{language}"
        print(f"Streaming dataset: {dataset_name} in chunks of {chunk_size}")
        h5_file_list = [f.strip() for f in h5_files.split(",")]
        data_generator = load_nvidia_detection(
            h5_file_list, max_rows, language,
            max_size_limit=max_size_limit, chunk_size=chunk_size,
        )
    else:
        raise ValueError("Either --pdf_path or --dataset_name must be provided")

    # --- Inference loop ---
    total_inference_time = 0.0
    page_metrics = collections.OrderedDict()
    global_idx = 0
    total_samples = 0

    if debug:
        debug_path = os.path.join(results_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)

    result_path = os.path.join(results_dir, pathname)
    os.makedirs(result_path, exist_ok=True)

    for img_chunk, bbox_chunk in data_generator:
        chunk_length = len(img_chunk)
        total_samples += chunk_length
        print(f"\nProcessing chunk of {chunk_length} images...")

        start = time.time()
        predictions = det_model.predict(img_chunk, batch_size=batch_size)
        total_inference_time += time.time() - start

        for local_idx, (pred_boxes, gt_boxes) in enumerate(
            tqdm(zip(predictions, bbox_chunk), total=chunk_length, leave=False)
        ):
            raw_metrics = precision_recall_f1_coverage(pred_boxes, gt_boxes)
            metrics = {k: float(v) for k, v in raw_metrics.items()}
            metrics["page_iou"] = calculate_iou(pred_boxes, gt_boxes)
            page_metrics[global_idx] = metrics

            if debug:
                combined = copy.deepcopy(img_chunk[local_idx])
                combined = draw_bboxes_on_image(combined, gt_boxes, color="green", width=2)
                combined = draw_bboxes_on_image(combined, pred_boxes, color="red", width=2)
                combined.save(os.path.join(debug_path, f"{global_idx}_bbox_debug.png"))

            global_idx += 1

        del img_chunk, bbox_chunk, predictions
        gc.collect()

    print(f"\nFinished processing {total_samples} total images.")

    # --- Aggregate metrics ---
    mean_metrics = {}
    metric_types = []
    if page_metrics:
        metric_types = sorted(page_metrics[0].keys())
        for m in metric_types:
            vals = [page_metrics[page][m] for page in page_metrics]
            mean_metrics[m] = sum(vals) / len(vals)
    else:
        print("Warning: No metrics calculated. Dataset may be empty.")

    time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0.0

    out_data = {
        "dataset": pathname,
        "model": model_path,
        "model_name": model,
        "num_samples": total_samples,
        "times": {"total": total_inference_time, "per_sample": time_per_sample},
        "mean_metrics": mean_metrics,
        "sample_details": page_metrics,
    }

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Model", "Time (s)", "Time/sample (s)"] + metric_types
    table_data = [
        [model, f"{total_inference_time:.2f}", f"{time_per_sample:.4f}"]
        + [f"{mean_metrics[m]:.4f}" for m in metric_types]
    ]

    print("\n" + "=" * 70)
    print("DETECTION METRICS")
    print("=" * 70)
    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print("\nMetric descriptions:")
    print("  Precision/Recall: coverage threshold 0.5")
    print("  F1: harmonic mean of precision and recall")
    print("  Page IoU: polygon-union IoU for the entire page")
    print(f"\n✓ Results saved to {result_path}")


if __name__ == "__main__":
    main()
