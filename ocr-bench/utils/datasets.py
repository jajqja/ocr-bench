"""Dataset loading utilities shared across all benchmark scripts."""

import gc
import io
import json
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import h5py
import requests
from PIL import Image
from tqdm import tqdm

import datasets as hf_datasets
from pdf2image import convert_from_bytes
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb

from utils.bbox import get_pdf_lines, rescale_bbox

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def download_h5_file(url: str, output_path: str) -> str:
    """Download an H5 file from a URL, skipping if already cached."""
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


# ---------------------------------------------------------------------------
# Detection data loaders
# ---------------------------------------------------------------------------


def load_pdf_detection(
    pdf_path: str, max_pages: int = 100
) -> Tuple[List[Image.Image], List[List[List[float]]]]:
    """Load a PDF and return page images with GT bboxes for detection."""
    doc = open_pdf(pdf_path)
    page_count = min(len(doc), max_pages)
    page_indices = list(range(page_count))
    images = get_page_images(doc, page_indices)
    doc.close()
    image_sizes = [img.size for img in images]
    correct_boxes = get_pdf_lines(pdf_path, image_sizes)
    return images, correct_boxes


def load_doclaynet_detection(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List[Image.Image], List[List[List[float]]]]:
    """Load vikp/doclaynet_bench for detection."""
    dataset = hf_datasets.load_dataset(dataset_name, split=f"train[:{max_rows}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    correct_boxes = []
    for i, boxes in enumerate(dataset["bboxes"]):
        img_size = images[i].size
        correct_boxes.append([rescale_bbox(b, (1000, 1000), img_size) for b in boxes])
    return images, correct_boxes


def load_pdfa_detection(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List[Image.Image], List[List[List[float]]]]:
    """Load pixparse/pdfa-eng-wds for detection."""
    dataset = hf_datasets.load_dataset(dataset_name, split="train", streaming=False)
    images = []
    bboxes = []

    for idx, sample in enumerate(dataset):
        if idx >= max_rows:
            break
        try:
            pdf_bytes = sample["pdf"]
            pdf_pages = convert_from_bytes(pdf_bytes, dpi=300)
            pdf_pages = convert_if_not_rgb(pdf_pages)

            metadata = (
                json.loads(sample["ocr"])
                if isinstance(sample["ocr"], str)
                else sample["ocr"]
            )

            for page_idx, page_data in enumerate(metadata.get("pages", [])):
                if page_idx >= len(pdf_pages):
                    break
                img = pdf_pages[page_idx]
                images.append(img)

                page_bboxes = []
                for word_item in page_data.get("lines", []):
                    for bbox in word_item.get("bbox", []):
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


def load_h5_detection_data(
    h5_path: str,
    max_rows: int = 100,
    max_size_limit: Optional[int] = None,
    chunk_size: int = 10000,
) -> Generator[Tuple[List, List], None, None]:
    """Yield (images, bboxes) chunks from a single H5 file."""
    images = []
    all_line_bboxes = []

    with h5py.File(h5_path, "r") as f:
        total_to_load = min(len(f["images"]), max_rows)

        for idx in tqdm(
            range(total_to_load), desc=f"Loading H5 ({os.path.basename(h5_path)})"
        ):
            try:
                img_bytes = f["images"][idx]
                image = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
                original_size = image.size

                scale_factor = 1.0
                if max_size_limit:
                    max_dim = max(original_size)
                    if max_dim > max_size_limit:
                        scale_factor = max_size_limit / max_dim
                        new_w = int(original_size[0] * scale_factor)
                        new_h = int(original_size[1] * scale_factor)
                        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

                images.append(image)

                annotation_str = f["annotations"][idx]
                if isinstance(annotation_str, bytes):
                    annotation_str = annotation_str.decode("utf-8")
                annotation = json.loads(annotation_str)

                line_bboxes = []
                for line in annotation.get("line_bboxes", []):
                    bbox = line.get("bbox", [])
                    if bbox:
                        x, y, w, h = bbox
                        sf = scale_factor
                        line_bboxes.append([x * sf, y * sf, (x + w) * sf, (y + h) * sf])
                all_line_bboxes.append(line_bboxes)

                if len(images) >= chunk_size:
                    yield images, all_line_bboxes
                    images = []
                    all_line_bboxes = []
                    gc.collect()

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    if images:
        yield images, all_line_bboxes


def load_nvidia_detection(
    h5_files: List[str],
    max_rows: int = 100,
    language: str = "en",
    max_size_limit: Optional[int] = None,
    chunk_size: int = 10000,
) -> Generator[Tuple[List, List], None, None]:
    """Stream detection data from NVIDIA OCR Synthetic Multilingual dataset."""
    base_url = (
        f"https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1"
        f"/resolve/main/{language}/train"
    )
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

            for img_batch, bbox_batch in load_h5_detection_data(
                local_path,
                remaining,
                max_size_limit=max_size_limit,
                chunk_size=chunk_size,
            ):
                yield img_batch, bbox_batch
                sample_count += len(img_batch)
                if sample_count >= max_rows:
                    break

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue


# ---------------------------------------------------------------------------
# Recognition data loaders
# ---------------------------------------------------------------------------


def get_full_image_bboxes(images: List[Image.Image]) -> List[List[List[int]]]:
    """Create a single full-image bbox per image (for whole-image recognition)."""
    return [[[0, 0, image.size[0], image.size[1]]] for image in images]


def extract_text_from_pdf(
    pdf_path: str, max_pages: int = 100
) -> Tuple[List[Image.Image], List[str], List[List[List[int]]]]:
    """Extract page images, GT text lines, and bboxes from a PDF."""
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
                        int(round(v))
                        for v in rescale_bbox(line["bbox"], page_size, image.size)
                    ]
                )
        page_bboxes.append(line_bboxes)

    doc.close()
    return images, ground_truth_texts, page_bboxes


def load_doclaynet_recognition(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List[Image.Image], List[str], List[List[List[float]]]]:
    """Load vikp/doclaynet_bench for recognition."""
    dataset = hf_datasets.load_dataset(dataset_name, split=f"train[:{max_rows}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    texts = [word for image_words in dataset["words"] for word in image_words]
    correct_boxes = []
    for i, boxes in enumerate(dataset["bboxes"]):
        img_size = images[i].size
        correct_boxes.append([rescale_bbox(b, (1000, 1000), img_size) for b in boxes])
    return images, texts, correct_boxes


def load_pdfa_recognition(
    dataset_name: str, max_rows: int = 100
) -> Tuple[List[Image.Image], List[str], List[List[List[int]]]]:
    """Load pixparse/pdfa-eng-wds for recognition."""
    dataset = hf_datasets.load_dataset(dataset_name, split="train", streaming=False)
    images = []
    texts = []
    bboxes = []

    for idx, sample in enumerate(dataset):
        if idx >= max_rows:
            break
        try:
            pdf_bytes = sample["pdf"]
            pdf_pages = convert_from_bytes(pdf_bytes, dpi=300)
            pdf_pages = convert_if_not_rgb(pdf_pages)

            metadata = (
                json.loads(sample["ocr"])
                if isinstance(sample["ocr"], str)
                else sample["ocr"]
            )

            for page_idx, page_data in enumerate(metadata.get("pages", [])):
                if page_idx >= len(pdf_pages):
                    break
                img = pdf_pages[page_idx]
                images.append(img)

                page_texts = []
                page_bboxes = []
                for word_item in page_data.get("words", []):
                    for word_text, bbox in zip(
                        word_item.get("text", []), word_item.get("bbox", [])
                    ):
                        if word_text.strip():
                            page_texts.append(word_text)
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


def load_h5_recognition_data(
    h5_path: str, max_rows: int = 100, max_size_limit: Optional[int] = None
) -> Tuple[List[Image.Image], List[str], List[List[List[float]]]]:
    """Load recognition data from a single H5 file."""
    images = []
    texts = []
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
                img_bytes = f["images"][idx]
                image = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
                original_size = image.size

                scale_factor = 1.0
                if max_size_limit:
                    max_dim = max(original_size)
                    if max_dim > max_size_limit:
                        scale_factor = max_size_limit / max_dim
                        new_w = int(original_size[0] * scale_factor)
                        new_h = int(original_size[1] * scale_factor)
                        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

                images.append(image)

                annotation_str = f["annotations"][idx]
                if isinstance(annotation_str, bytes):
                    annotation_str = annotation_str.decode("utf-8")
                annotation = json.loads(annotation_str)

                line_bboxes = []
                for line in annotation.get("line_bboxes", []):
                    text = line.get("text", "")
                    bbox = line.get("bbox", [])
                    if text.strip() and bbox:
                        texts.append(text)
                        x, y, w, h = bbox
                        sf = scale_factor
                        line_bboxes.append([x * sf, y * sf, (x + w) * sf, (y + h) * sf])

                bboxes.append(line_bboxes)
                sample_count += 1

                if sample_count % 50 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    return images, texts, bboxes


def load_nvidia_recognition(
    h5_files: List[str],
    max_rows: int = 100,
    language: str = "en",
    max_size_limit: Optional[int] = None,
) -> Tuple[List[Image.Image], List[str], List[List[List[float]]]]:
    """Load recognition data from NVIDIA OCR Synthetic Multilingual dataset."""
    base_url = (
        f"https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1"
        f"/resolve/main/{language}/train"
    )
    cache_dir = os.path.expanduser("~/.cache/nvidia_ocr_multilingual")
    images = []
    texts = []
    bboxes = []
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
            print(f"Loading data from {h5_file}...")
            remaining = max_rows - sample_count
            img_batch, text_batch, bbox_batch = load_h5_recognition_data(
                local_path, remaining, max_size_limit=max_size_limit
            )
            images.extend(img_batch)
            texts.extend(text_batch)
            bboxes.extend(bbox_batch)
            sample_count += len(img_batch)

            del img_batch, text_batch, bbox_batch
            gc.collect()

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue

    print(f"Loaded {len(images)} images and {len(texts)} text lines")
    return images, texts, bboxes


def load_recognition_folder(
    data_dir: str, image_folder: str, label_file: str, max_rows: int = 100
) -> Tuple[List[Image.Image], List[str], List[List[List[int]]]]:
    """Load recognition dataset from a local folder.

    Expected layout:
        data_dir/
            <image_folder>/
                image_1.jpg
            <label_file>     # format: "image_1.jpg<TAB>label text"
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
                    "expected '<image_name>\\t<label>'"
                )
            image_name, label = parts
            image_path = images_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found at line {line_number}: {image_path}"
                )
            with Image.open(image_path) as image:
                images.append(image.copy())
            texts.append(label)

    images = convert_if_not_rgb(images)
    bboxes = get_full_image_bboxes(images)
    return images, texts, bboxes
