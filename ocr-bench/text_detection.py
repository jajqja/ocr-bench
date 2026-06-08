import collections
import copy
import json
from typing import Optional, Tuple, List, Generator

import click

from utils.bbox import get_pdf_lines
from utils.metrics import (
    precision_recall_f1_coverage,
    calculate_iou,
)
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
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
from utils.bbox import draw_bboxes_on_image
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
    h5_path: str,
    max_rows: int = 100,
    max_size_limit: Optional[int] = None,
    chunk_size: int = 10000,
) -> Generator[Tuple[List, List], None, None]:
    """Load detection data from a single H5 file using a Generator to save memory."""
    images = []
    all_line_bboxes = []
    sample_count = 0

    with h5py.File(h5_path, "r") as f:
        num_samples = len(f["images"])
        total_to_load = min(num_samples, max_rows)

        for idx in tqdm(
            range(total_to_load), desc=f"Loading H5 data ({os.path.basename(h5_path)})"
        ):
            try:
                # Load image from bytes
                img_bytes = f["images"][idx]
                image = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
                original_size = image.size

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

                line_bboxes = []
                for line in annotation.get("line_bboxes", []):
                    line_bbox = line.get("bbox", [])
                    if line_bbox:
                        x, y, w, h = line_bbox
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

                all_line_bboxes.append(line_bboxes)
                sample_count += 1

                # Yield chunk when it reaches chunk_size
                if len(images) >= chunk_size:
                    yield images, all_line_bboxes
                    images = []
                    all_line_bboxes = []
                    gc.collect()

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

        # Yield remaining samples
        if images:
            yield images, all_line_bboxes


def load_nvidia_ocr_multilingual_dataset(
    h5_files: List[str],
    max_rows: int = 100,
    language: str = "en",
    max_size_limit: Optional[int] = None,
    chunk_size: int = 10000,
) -> Generator[Tuple[List, List], None, None]:
    base_url = f"https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/resolve/main/{language}/train"
    cache_dir = os.path.expanduser("~/.cache/nvidia_ocr_multilingual")

    sample_count = 0

    for h5_file in h5_files:
        if sample_count >= max_rows:
            break

        if not h5_file.endswith(".h5"):
            h5_file = f"{h5_file}.h5"

        url = f"{base_url}/{h5_file}?download=true"
        local_path = os.path.join(cache_dir, language, h5_file)

        try:
            local_path = download_h5_file(url, local_path)
            print(f"Streaming data from {h5_file}...")

            remaining = max_rows - sample_count

            # Iterate through the chunks generated by the H5 loader
            for img_batch, line_bboxes in load_h5_detection_data(
                local_path,
                remaining,
                max_size_limit=max_size_limit,
                chunk_size=chunk_size,
            ):
                yield img_batch, line_bboxes
                sample_count += len(img_batch)

                if sample_count >= max_rows:
                    break

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue


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

    chunk_size = 10000
    data_generator = None
    total_samples = 0

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

        data_generator = [
            (images, correct_boxes)
        ]  # Wrap in a single chunk for compatibility
        total_samples = len(images)
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
        data_generator = [(images, correct_boxes)]
        total_samples = len(images)
    elif dataset_name is not None and dataset_name == "pixparse/pdfa-eng-wds":
        print(f"Loading dataset: {dataset_name}")
        pathname = dataset_name.replace("/", "_")
        images, correct_boxes = load_pdfa_detection_dataset(dataset_name, max_rows)
        data_generator = [(images, correct_boxes)]
        total_samples = len(images)
    elif (
        dataset_name is not None
        and dataset_name == "nvidia/OCR-Synthetic-Multilingual-v1"
    ):
        print(f"Streaming dataset: {dataset_name} in chunks of {chunk_size}")
        pathname = f"nvidia_ocr_{language}"
        h5_file_list = [f.strip() for f in h5_files.split(",")]
        # This is now a generator
        data_generator = load_nvidia_ocr_multilingual_dataset(
            h5_file_list,
            max_rows,
            language,
            max_size_limit=max_size_limit,
            chunk_size=chunk_size,
        )
    else:
        raise ValueError("Either pdf_path or dataset_name must be provided")


    # Bắt đầu từ đoạn chạy inference trong hàm main()
    print("Running inference and calculating metrics in chunks...")

    total_inference_time = 0
    page_metrics = collections.OrderedDict()
    global_idx = 0
    total_samples = 0  # Đảm bảo biến này đã được khởi tạo trước đó

    if debug:
        debug_path = os.path.join(results_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)

    folder_name = pathname
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    # 1. Process Chunk by Chunk
    for img_chunk, bbox_chunk in data_generator:
        chunk_length = len(img_chunk)
        total_samples += chunk_length
        print(f"\nProcessing chunk of {chunk_length} images...")

        # Inference
        start = time.time()
        predictions = det_predictor(img_chunk, batch_size=batch_size)
        total_inference_time += time.time() - start

        # Metrics cho chunk hiện tại
        for local_idx, (sb, clb) in enumerate(zip(predictions, bbox_chunk)):
            surya_boxes = [s.bbox for s in sb.bboxes]

            # Tính precision, recall, f1
            raw_metrics = precision_recall_f1_coverage(surya_boxes, clb)
            metrics = {k: float(v) for k, v in raw_metrics.items()}
            metrics["page_iou"] = calculate_iou(surya_boxes, clb)

            page_metrics[global_idx] = metrics

            # Lưu ảnh debug nếu cần
            if debug:
                combined_image = copy.deepcopy(img_chunk[local_idx])
                combined_image = draw_bboxes_on_image(
                    combined_image, clb, color="green", width=2
                )
                combined_image = draw_bboxes_on_image(
                    combined_image, surya_boxes, color="red", width=2
                )
                combined_image.save(
                    os.path.join(debug_path, f"{global_idx}_bbox_debug.png")
                )

            global_idx += 1

        # CLEAR RAM NGAY LẬP TỨC CHO CHUNK NÀY
        del img_chunk, bbox_chunk, predictions
        gc.collect()

    print(f"\nFinished processing {total_samples} total images.")

    # 2. Calculate mean metrics (Đã gom chung từ tất cả các chunks)
    print("Calculating mean metrics...")
    mean_metrics = {}
    metric_types = []

    if page_metrics:
        metric_types = sorted(page_metrics[0].keys())
        for m in metric_types:
            metric_values = [page_metrics[page][m] for page in page_metrics]
            mean_metrics[m] = sum(metric_values) / len(metric_values)
    else:
        print("Warning: No metrics calculated. Dataset might be empty.")

    # 3. Save results
    time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0

    out_data = {
        "dataset": pathname,
        "model": model_path,
        "num_samples": total_samples,
        "times": {
            "total": total_inference_time,
            "per_sample": time_per_sample,
        },
        "mean_metrics": mean_metrics,
        "sample_details": page_metrics,
    }

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    # 4. Print results
    table_headers = ["Model", "Time (s)", "Time per sample (s)"] + metric_types
    table_data = [
        ["surya", f"{total_inference_time:.2f}", f"{time_per_sample:.4f}"]
        + [f"{mean_metrics[m]:.4f}" for m in metric_types],
    ]

    print("\n" + "=" * 70)
    print("DETECTION METRICS")
    print("=" * 70)
    print(tabulate(table_data, headers=table_headers, tablefmt="github"))

    print("\nMetric Descriptions:")
    print("  - Precision/Recall: Coverage threshold at 0.5")
    print("  - F1 Score: Harmonic mean of Precision and Recall")
    print(
        "  - Page IOU: Intersection over Union for the entire page (higher is better)"
    )
    print(
        f"  - Inference Time: {total_inference_time:.2f}s total, {time_per_sample:.4f}s per sample"
    )

    print(f"\n✓ Results saved to {result_path}")


if __name__ == "__main__":
    main()
