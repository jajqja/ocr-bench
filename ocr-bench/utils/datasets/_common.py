"""Helper dùng chung cho các dataset (tải file, đọc H5)."""

import gc
import io
import json
import os
from typing import Generator, List, Optional, Tuple

import h5py
import requests
from PIL import Image
from tqdm import tqdm


def download_h5_file(url: str, output_path: str) -> str:
    """Tải file H5 từ URL, bỏ qua nếu đã có sẵn (cache)."""
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


def _resize_to_limit(
    image: Image.Image, max_size_limit: Optional[int]
) -> Tuple[Image.Image, float]:
    """Resize ảnh để cạnh dài nhất không vượt max_size_limit. Trả (ảnh, scale)."""
    if not max_size_limit:
        return image, 1.0
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= max_size_limit:
        return image, 1.0
    scale = max_size_limit / max_dim
    image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image, scale


def load_h5_detection_data(
    h5_path: str,
    max_rows: int = 100,
    max_size_limit: Optional[int] = None,
    chunk_size: int = 10000,
) -> Generator[Tuple[List, List], None, None]:
    """Yield các chunk (images, bboxes) từ một file H5."""
    images: List[Image.Image] = []
    all_line_bboxes: List = []

    with h5py.File(h5_path, "r") as f:
        total_to_load = min(len(f["images"]), max_rows)

        for idx in tqdm(
            range(total_to_load), desc=f"Loading H5 ({os.path.basename(h5_path)})"
        ):
            try:
                img_bytes = f["images"][idx]
                image = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
                image, sf = _resize_to_limit(image, max_size_limit)
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


def load_h5_recognition_data(
    h5_path: str, max_rows: int = 100, max_size_limit: Optional[int] = None
) -> Tuple[List[Image.Image], List[str], List[List[List[float]]]]:
    """Đọc dữ liệu recognition từ một file H5."""
    images: List[Image.Image] = []
    texts: List[str] = []
    bboxes: List = []
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
                image, sf = _resize_to_limit(image, max_size_limit)
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
                        line_bboxes.append([x * sf, y * sf, (x + w) * sf, (y + h) * sf])

                bboxes.append(line_bboxes)
                sample_count += 1

                if sample_count % 50 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    return images, texts, bboxes
