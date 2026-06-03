import click
import json
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
from surya.common.util import rescale_bbox
from surya.settings import settings
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from tabulate import tabulate
import datasets
from pdf2image import convert_from_bytes
import io
import h5py
import requests

from utils.metrics import calculate_recognition_metrics


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


def load_h5_recognition_data(
    h5_path: str, max_rows: int = 100
) -> Tuple[List, List, List]:
    """Load recognition data from a single H5 file.

    Args:
        h5_path: Path to H5 file
        max_rows: Maximum number of samples to load

    Returns:
        Tuple of (images, texts, bboxes)
    """
    images = []
    texts = []
    bboxes = []
    sample_count = 0

    with h5py.File(h5_path, "r") as f:
        num_samples = len(f["images"])

        for idx in range(num_samples):
            if sample_count >= max_rows:
                break

            try:
                # Load image from bytes
                img_bytes = f["images"][idx]
                image = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
                images.append(image)

                # Parse annotation JSON
                annotation_str = f["annotations"][idx]
                if isinstance(annotation_str, bytes):
                    annotation_str = annotation_str.decode("utf-8")
                annotation = json.loads(annotation_str)

                # Extract line_bboxes and text
                line_bboxes = []
                for line in annotation.get("line_bboxes", []):
                    text = line.get("text", "")
                    bbox = line.get("bbox", [])
                    if text.strip() and bbox:
                        texts.append(text)
                        # bbox format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
                        x, y, w, h = bbox
                        line_bboxes.append([x, y, x + w, y + h])

                bboxes.append(line_bboxes)
                sample_count += 1

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    return images, texts, bboxes


def load_nvidia_ocr_multilingual_dataset(
    h5_files: List[str], max_rows: int = 100, language: str = "en"
) -> Tuple[List, List, List]:
    """Load NVIDIA OCR Synthetic Multilingual dataset from H5 files.

    Args:
        h5_files: List of H5 filenames to download (e.g., ["train_000", "train_001"])
                  Files will be downloaded from HuggingFace Hub
        max_rows: Maximum total number of samples to load
        language: Language to load (en, ja, ko, ru, zh_hans, zh_hant)

    Returns:
        Tuple of (images, texts, bboxes) where:
        - images: List of PIL images
        - texts: Flattened list of text lines
        - bboxes: List of line_bboxes per image
    """
    base_url = f"https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/resolve/main/{language}/train"
    cache_dir = os.path.expanduser("~/.cache/nvidia_ocr_multilingual")

    images = []
    texts = []
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
            img_batch, text_batch, bbox_batch = load_h5_recognition_data(
                local_path, remaining
            )

            images.extend(img_batch)
            texts.extend(text_batch)
            bboxes.extend(bbox_batch)
            sample_count += len(img_batch)

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue

    print(f"Loaded {len(images)} total images and {len(texts)} text lines")
    return images, texts, bboxes


def extract_text_from_pdf(
    pdf_path: str, max_pages: int = 100
) -> Tuple[List, List, List]:
    """Extract text lines and bboxes from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract

    Returns:
        Tuple of (page_images, text_line_ground_truths, page_text_line_bboxes)
    """
    doc = open_pdf(pdf_path)
    page_count = min(len(doc), max_pages)
    page_indices = list(range(page_count))

    images = get_page_images(doc, page_indices)
    images = convert_if_not_rgb(images)

    ground_truth_texts = []
    page_bboxes = []
    for idx, image in zip(page_indices, images):
        page = doc[idx]
        blocks = page.get_text("dict", sort=True)["blocks"]
        page_box = page.bound()
        page_size = (page_box[2] - page_box[0], page_box[3] - page_box[1])

        line_bboxes = []
        for block in blocks:
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not text.strip():
                    continue

                ground_truth_texts.append(text)
                line_bboxes.append(
                    [
                        int(round(value))
                        for value in rescale_bbox(line["bbox"], page_size, image.size)
                    ]
                )

        page_bboxes.append(line_bboxes)

    doc.close()
    return images, ground_truth_texts, page_bboxes


def get_full_image_bboxes(images: List) -> List[List[List[int]]]:
    """Create one full-image bbox per image for recognition datasets."""
    return [[[0, 0, image.size[0], image.size[1]]] for image in images]


def load_recognition_dataset(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List, List, List]:
    """Load recognition dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset
        max_rows: Maximum number of samples

    Returns:
        Tuple of (images, texts)
    """
    dataset = datasets.load_dataset(dataset_name, split=f"train[:{max_rows}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    texts = [box_words for image_words in dataset["words"] for box_words in image_words]
    correct_boxes = []
    for i, boxes in enumerate(dataset["bboxes"]):
        img_size = images[i].size
        # 1000,1000 is bbox size for doclaynet
        correct_boxes.append([rescale_bbox(b, (1000, 1000), img_size) for b in boxes])
    return images, texts, correct_boxes


def load_pdfa_recognition_dataset(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List, List, List]:
    """Load PDFA dataset for recognition benchmark.

    Args:
        dataset_name: Name of the PDFA dataset (pixparse/pdfa-eng-wds)
        max_rows: Maximum number of documents to load

    Returns:
        Tuple of (images, texts, bboxes) where:
        - images: List of PIL images from PDF pages
        - texts: List of text lines (flattened)
        - bboxes: List of lists of bboxes per page
    """
    dataset = datasets.load_dataset(dataset_name, split="train", streaming=False)

    images = []
    texts = []
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

                # Extract text and bounding boxes from words
                page_texts = []
                page_bboxes = []

                for word_item in page_data.get("words", []):
                    word_list = word_item.get("text", [])
                    word_bboxes = word_item.get("bbox", [])

                    for word_text, bbox in zip(word_list, word_bboxes):
                        if word_text.strip():
                            page_texts.append(word_text)

                            # bbox format: [left, top, width, height] (normalized 0-1)
                            # Convert to pixel coordinates
                            x1 = int(bbox[0] * img.width)
                            y1 = int(bbox[1] * img.height)
                            x2 = int((bbox[0] + bbox[2]) * img.width)
                            y2 = int((bbox[1] + bbox[3]) * img.height)
                            page_bboxes.append([x1, y1, x2, y2])

                texts.extend(page_texts)
                bboxes.append(page_bboxes)

        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue

    return images, texts, bboxes


def load_recognition_folder(
    data_dir: str, image_folder: str, label_file: str, max_rows: int = 100
) -> Tuple[List, List, List]:
    """Load recognition dataset from a local folder.

    Expected structure:
        data_dir/
            images/
                image_1.jpg
                image_2.png
            labels.txt

    labels.txt format:
        image_1.jpg<TAB>label text
        image_2.png<TAB>another label

    Args:
        data_dir: Path to folder containing images/ and labels.txt
        image_folder: Name of the folder containing images
        label_file: Name of the labels file
        max_rows: Maximum number of samples

    Returns:
        Tuple of (images, texts)
    """
    data_path = Path(data_dir)
    images_dir = data_path / image_folder
    labels_path = data_path / label_file

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    images = []
    texts = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if len(images) >= max_rows:
                break

            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid labels.txt format at line {line_number}: "
                    "expected '<image_name>\t<label>'"
                )

            image_name, label = parts
            image_path = images_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found for labels.txt line {line_number}: {image_path}"
                )

            with Image.open(image_path) as image:
                images.append(image.copy())
            texts.append(label)

    images = convert_if_not_rgb(images)
    bboxes = get_full_image_bboxes(images)
    return images, texts, bboxes


def batch_recognize(
    predictor, images: List, bboxes: List[List[List[int]]], batch_size: int = 8
) -> List[str]:  # Trả về List[str] phẳng hoàn toàn
    """Recognize text from images in batches.

    Args:
        predictor: Recognition predictor model
        images: List of PIL images
        bboxes: List of textline bboxes per image
        batch_size: Batch size for processing

    Returns:
        Flattened list of recognized text lines
    """
    predictions = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_bboxes = bboxes[i : i + batch_size]
        batch_results = predictor(
            batch,
            bboxes=batch_bboxes,
            recognition_batch_size=batch_size,
        )
        for result in batch_results:
            # Dùng .extend để trải phẳng (flatten) tất cả text_lines của các ảnh vào 1 danh sách duy nhất
            predictions.extend(text_line.text for text_line in result.text_lines)

    return predictions


@click.command(help="Benchmark recognition model on PDF dataset.")
@click.option(
    "--pdf_path", type=str, help="Path to PDF file for evaluation.", default=None
)
@click.option(
    "--dataset_name", type=str, help="Hugging Face dataset name.", default=None
)
@click.option(
    "--data_dir",
    type=str,
    help="Path to local recognition dataset folder with images/ and labels.txt.",
    default=None,
)
@click.option(
    "--image_folder",
    type=str,
    help="Name of the folder containing images.",
    default="images",
)
@click.option(
    "--label_file",
    type=str,
    help="Name of the labels file.",
    default="labels.txt",
)
@click.option(
    "--results_dir",
    type=str,
    help="Path to directory for results.",
    default=os.path.join(settings.RESULT_DIR, "recognition_benchmark"),
)
@click.option(
    "--max_rows", type=int, help="Maximum number of samples to evaluate.", default=100
)
@click.option("--batch_size", type=int, help="Batch size for inference.", default=8)
@click.option("--model_path", type=str, required=True, help="Path to recognition model")
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
def main(
    pdf_path: Optional[str],
    dataset_name: Optional[str],
    data_dir: Optional[str],
    image_folder: Optional[str],
    label_file: Optional[str],
    results_dir: str,
    max_rows: int,
    batch_size: int,
    model_path: str,
    language: str,
    h5_files: str,
):
    """Benchmark text recognition model."""

    # Load model
    print(f"Loading recognition model from {model_path}...")
    foundation_model = FoundationPredictor(checkpoint=model_path)
    rec_predictor = RecognitionPredictor(foundation_model)

    # Load data
    if pdf_path is not None:
        print(f"Loading data from PDF: {pdf_path}")
        pathname = Path(pdf_path).stem
        images, ground_truth_texts, bboxes = extract_text_from_pdf(pdf_path, max_rows)
    elif dataset_name is not None and dataset_name == "vikp/doclaynet_bench":
        print(f"Loading dataset: {dataset_name}")
        pathname = dataset_name.replace("/", "_")

        images, ground_truth_texts, bboxes = load_recognition_dataset(
            dataset_name, max_rows
        )
    elif dataset_name is not None and dataset_name == "pixparse/pdfa-eng-wds":
        print(f"Loading dataset: {dataset_name}")
        pathname = dataset_name.replace("/", "_")

        images, ground_truth_texts, bboxes = load_pdfa_recognition_dataset(
            dataset_name, max_rows
        )
    elif (
        dataset_name is not None
        and dataset_name == "nvidia/OCR-Synthetic-Multilingual-v1"
    ):
        print(f"Loading dataset: {dataset_name}")
        pathname = f"nvidia_ocr_{language}"

        h5_file_list = [f.strip() for f in h5_files.split(",")]
        images, ground_truth_texts, bboxes = load_nvidia_ocr_multilingual_dataset(
            h5_file_list, max_rows, language
        )
    elif data_dir is not None:
        print(f"Loading local dataset: {data_dir}")
        pathname = Path(data_dir).name
        images, ground_truth_texts, bboxes = load_recognition_folder(
            data_dir, image_folder or "images", label_file or "labels.txt", max_rows
        )
    else:
        raise ValueError("Either pdf_path, dataset_name, or data_dir must be provided")

    flat_bboxes = [bbox for image_bboxes in bboxes for bbox in image_bboxes]
    sample_count = len(ground_truth_texts)

    print(f"Loaded {len(images)} images and {sample_count} text lines")

    # Run inference
    print("Running inference...")
    start_time = time.time()
    predictions = batch_recognize(rec_predictor, images, bboxes, batch_size)
    inference_time = time.time() - start_time

    print(
        f"Inference completed in {inference_time:.2f}s ({inference_time/len(images):.4f}s per image)"
    )

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_recognition_metrics(ground_truth_texts, predictions)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{pathname}_results.json")

    output_data = {
        "model": model_path,
        "dataset": pathname,
        "num_samples": sample_count,
        "num_images": len(images),
        "inference_time_total": inference_time,
        "inference_time_per_sample": (
            inference_time / sample_count if sample_count else 0
        ),
        "metrics": {
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "accuracy": metrics["accuracy"],
        },
        "predictions": [
            {
                "ground_truth": gt,
                "prediction": pred,
                "bbox": flat_bboxes[i],
                "cer": metrics["cer_scores"][i],
                "wer": metrics["wer_scores"][i],
            }
            for i, (gt, pred) in enumerate(zip(ground_truth_texts, predictions))
        ],
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print results table
    table_data = [
        ["Character Error Rate (CER)", f"{metrics['cer']:.4f}"],
        ["Word Error Rate (WER)", f"{metrics['wer']:.4f}"],
        ["Accuracy (exact match)", f"{metrics['accuracy']:.4f}"],
        ["Inference Time (total)", f"{inference_time:.2f}s"],
        [
            "Inference Time (per text line)",
            f"{inference_time / sample_count if sample_count else 0:.4f}s",
        ],
    ]

    print("\n" + "=" * 50)
    print("RECOGNITION METRICS")
    print("=" * 50)
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))
    print(f"\nResults saved to: {result_path}")

    # Show sample predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS (first 5)")
    print("=" * 50)
    for i in range(min(5, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {ground_truth_texts[i][:100]}...")
        print(f"  Prediction:   {predictions[i][:100]}...")
        print(
            f"  CER: {metrics['cer_scores'][i]:.4f}, WER: {metrics['wer_scores'][i]:.4f}"
        )


if __name__ == "__main__":
    main()
