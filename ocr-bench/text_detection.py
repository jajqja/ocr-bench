import collections
import copy
import json
from typing import Optional, Tuple, List

import click

from utils.bbox import get_pdf_lines
from utils.metrics import precision_recall_f1, calculate_iou_metrics
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
from surya.debug.draw import draw_polys_on_image
from surya.common.util import rescale_bbox
from surya.settings import settings
from surya.detection import DetectionPredictor

import os
import time
from tabulate import tabulate
from tqdm import tqdm
import datasets
from pdf2image import convert_from_bytes
import io
from PIL import Image
import h5py
import requests
import gc


def download_h5_file(url: str, output_path: str) -> str:
    """Download H5 file from URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file

    Returns:
        Path to downloaded file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"  Downloaded {downloaded}/{total_size} bytes ({pct:.1f}%)")

    return output_path


def load_h5_detection_data(
    h5_path: str, max_rows: int = 100, max_size_limit: Optional[int] = None
) -> Tuple[List, List]:
    """Load detection data from a single H5 file with memory optimization.

    Args:
        h5_path: Path to H5 file
        max_rows: Maximum number of samples to load
        max_size_limit: Optional maximum size for the largest dimension (maintains aspect ratio)

    Returns:
        Tuple of (images, bboxes)
    """
    images = []
    bboxes = []
    sample_count = 0

    with h5py.File(h5_path, "r") as f:
        num_samples = len(f["images"])

        for idx in tqdm(
            range(num_samples), total=min(num_samples, max_rows), desc="Loading H5 data"
        ):
            if sample_count >= max_rows:
                break

            try:
                # Load image from bytes
                img_bytes = f["images"][idx]
                image = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
                original_size = image.size  # (width, height)

                # Resize image if requested (to save memory), maintaining aspect ratio
                scale_factor = 1.0
                if max_size_limit:
                    max_dim = max(original_size)
                    if max_dim > max_size_limit:
                        scale_factor = max_size_limit / max_dim
                        new_width = int(original_size[0] * scale_factor)
                        new_height = int(original_size[1] * scale_factor)
                        image = image.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )

                images.append(image)

                # Parse annotation JSON
                annotation_str = f["annotations"][idx]
                if isinstance(annotation_str, bytes):
                    annotation_str = annotation_str.decode("utf-8")
                annotation = json.loads(annotation_str)

                # Extract line_bboxes
                line_bboxes = []
                for line in annotation.get("line_bboxes", []):
                    bbox = line.get("bbox", [])
                    if bbox:
                        # bbox format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
                        x, y, w, h = bbox
                        # Scale bbox if image was resized
                        if scale_factor != 1.0:
                            line_bboxes.append(
                                [
                                    x * scale_factor,
                                    y * scale_factor,
                                    (x + w) * scale_factor,
                                    (y + h) * scale_factor,
                                ]
                            )
                        else:
                            line_bboxes.append([x, y, x + w, y + h])

                bboxes.append(line_bboxes)
                sample_count += 1

                # Garbage collection every 50 samples to free memory
                if sample_count % 50 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    return images, bboxes


def load_nvidia_ocr_multilingual_dataset(
    h5_files: List[str],
    max_rows: int = 100,
    language: str = "en",
    max_size_limit: Optional[int] = None,
) -> Tuple[List, List]:
    """Load NVIDIA OCR Synthetic Multilingual dataset from H5 files with memory optimization.

    Args:
        h5_files: List of H5 filenames to download (e.g., ["train_000", "train_001"])
                  Files will be downloaded from HuggingFace Hub
        max_rows: Maximum total number of samples to load
        language: Language to load (en, ja, ko, ru, zh_hans, zh_hant)
        max_size_limit: Optional maximum size for the largest dimension (maintains aspect ratio)

    Returns:
        Tuple of (images, bboxes) where:
        - images: List of PIL images
        - bboxes: List of line_bboxes per image
    """
    base_url = f"https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/resolve/main/{language}/train"
    cache_dir = os.path.expanduser("~/.cache/nvidia_ocr_multilingual")

    images = []
    bboxes = []
    sample_count = 0

    for h5_file in h5_files:
        if sample_count >= max_rows:
            break

        # Ensure filename ends with .h5
        if not h5_file.endswith(".h5"):
            h5_file = f"{h5_file}.h5"

        # Download file
        url = f"{base_url}/{h5_file}?download=true"
        local_path = os.path.join(cache_dir, language, h5_file)

        try:
            local_path = download_h5_file(url, local_path)
            print(f"Loading data from {h5_file}...")

            # Load from H5 file
            remaining = max_rows - sample_count
            img_batch, bbox_batch = load_h5_detection_data(
                local_path, remaining, max_size_limit=max_size_limit
            )

            images.extend(img_batch)
            bboxes.extend(bbox_batch)
            sample_count += len(img_batch)

            # Clear batch after extending to free memory
            del img_batch, bbox_batch
            gc.collect()

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue

    print(f"Loaded {len(images)} total images")
    return images, bboxes


def load_pdfa_detection_dataset(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List, List]:
    """Load PDFA dataset for detection benchmark.

    Args:
        dataset_name: Name of the PDFA dataset (pixparse/pdfa-eng-wds)
        max_rows: Maximum number of documents to load

    Returns:
        Tuple of (images, bboxes) where:
        - images: List of PIL images from PDF pages
        - bboxes: List of lists of bboxes per page
    """
    dataset = datasets.load_dataset(dataset_name, split="train", streaming=False)

    images = []
    bboxes = []

    for idx, sample in enumerate(dataset):
        if idx >= max_rows:
            break

        try:
            # Extract PDF and render to images
            pdf_bytes = sample["pdf"]
            pdf_pages = convert_from_bytes(pdf_bytes, dpi=300)
            pdf_pages = convert_if_not_rgb(pdf_pages)

            # Extract metadata from JSON
            metadata = (
                json.loads(sample["ocr"])
                if isinstance(sample["ocr"], str)
                else sample["ocr"]
            )

            # Process each page
            for page_idx, page_data in enumerate(metadata.get("pages", [])):
                if page_idx >= len(pdf_pages):
                    break

                img = pdf_pages[page_idx]
                images.append(img)

                # Extract bounding boxes from words (using normalized coords)
                page_bboxes = []
                for word_item in page_data.get("lines", []):
                    word_bboxes = word_item.get("bbox", [])
                    for bbox in word_bboxes:
                        # bbox format: [left, top, width, height] (normalized 0-1)
                        # Convert to pixel coordinates
                        x1 = int(bbox[0] * img.width)
                        y1 = int(bbox[1] * img.height)
                        x2 = int((bbox[0] + bbox[2]) * img.width)
                        y2 = int((bbox[1] + bbox[3]) * img.height)
                        page_bboxes.append([x1, y1, x2, y2])

                bboxes.append(page_bboxes)

        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue

    return images, bboxes


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
@click.option(
    "--language",
    type=str,
    help="Language for NVIDIA dataset (en, ja, ko, ru, zh_hans, zh_hant).",
    default="en",
)
@click.option(
    "--h5_files",
    type=str,
    help="Comma-separated H5 file names to load (e.g., 'train_000,train_001,train_002'). For NVIDIA dataset only.",
    default="train_000",
)
@click.option(
    "--batch_size",
    type=int,
    help="Batch size for inference (default: 8). Adjust based on GPU memory.",
    default=8,
)
@click.option(
    "--max_size_limit",
    type=int,
    help="Maximum size for largest image dimension (maintains aspect ratio, e.g., 1024). Leave empty to keep original size.",
    default=None,
)
def main(
    pdf_path: Optional[str],
    dataset_name: Optional[str],
    results_dir: str,
    max_rows: int,
    debug: bool,
    model_path: str,
    language: str,
    h5_files: str,
    batch_size: int,
    max_size_limit: Optional[int],
    hf_token: Optional[str] = None,
):
    """Main benchmark function for detection."""
    det_predictor = DetectionPredictor(checkpoint=model_path)

    if max_size_limit:
        print(
            f"Images will be resized to maintain aspect ratio with max size limit = {max_size_limit} px"
        )

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
    elif dataset_name is not None and dataset_name == "pixparse/pdfa-eng-wds":
        print(f"Loading dataset: {dataset_name}")
        pathname = dataset_name.replace("/", "_")
        images, correct_boxes = load_pdfa_detection_dataset(dataset_name, max_rows)
    elif (
        dataset_name is not None
        and dataset_name == "nvidia/OCR-Synthetic-Multilingual-v1"
    ):
        print(f"Loading dataset: {dataset_name}")
        pathname = f"nvidia_ocr_{language}"
        h5_file_list = [f.strip() for f in h5_files.split(",")]
        images, correct_boxes = load_nvidia_ocr_multilingual_dataset(
            h5_file_list, max_rows, language, max_size_limit=max_size_limit
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
    predictions = det_predictor(images, batch_size=batch_size)
    inference_time = time.time() - start

    folder_name = pathname
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    # Calculate metrics
    print("Calculating metrics...")
    page_metrics = collections.OrderedDict()
    iou_scores = []

    for idx, (sb, cb) in enumerate(
        tqdm(zip(predictions, correct_boxes), total=len(predictions))
    ):
        surya_boxes = [s.bbox for s in sb.bboxes]
        surya_polys = [s.polygon for s in sb.bboxes]

        # Calculate precision and recall
        raw_metrics = precision_recall_f1(surya_boxes, cb)
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
    print("  - F1 Score: Harmonic mean of Precision and Recall")
    print("  - IOU: Intersection over Union (penalized for multiple overlapping boxes)")
    print(
        f"  - Inference Time: {inference_time:.2f}s total, {inference_time/len(images):.4f}s per sample"
    )

    print(f"\n✓ Results saved to {result_path}")


if __name__ == "__main__":
    main()
