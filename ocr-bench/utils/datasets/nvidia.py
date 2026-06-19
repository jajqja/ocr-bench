"""Dataset nvidia/OCR-Synthetic-Multilingual-v1 (stream H5 từ HuggingFace).

Opts:
    h5_files        (mặc định "train_000")  Danh sách file H5, phân tách bằng dấu phẩy.
    language        (mặc định "en")          en, ja, ko, ru, zh_hans, zh_hant.
    max_size_limit  (mặc định không giới hạn) Cạnh dài nhất tối đa của ảnh.
    chunk_size      (mặc định 10000)         Số mẫu mỗi chunk khi stream (chỉ detection).
"""

import gc
import os
from typing import Dict, Generator, List, Optional, Tuple

from PIL import Image

from utils.datasets._common import (
    download_h5_file,
    load_h5_detection_data,
    load_h5_recognition_data,
)
from utils.datasets.base import BaseDataset

_BASE_URL = (
    "https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1"
    "/resolve/main/{language}/train"
)
_CACHE_DIR = os.path.expanduser("~/.cache/nvidia_ocr_multilingual")


def _parse_opts(opts: Dict[str, str]):
    h5_files = [f.strip() for f in opts.get("h5_files", "train_000").split(",")]
    language = opts.get("language", "en")
    msl: Optional[int] = (
        int(opts["max_size_limit"]) if opts.get("max_size_limit") else None
    )
    chunk_size = int(opts.get("chunk_size", 10000))
    return h5_files, language, msl, chunk_size


def _local_path(language: str, h5_file: str) -> Tuple[str, str]:
    if not h5_file.endswith(".h5"):
        h5_file = f"{h5_file}.h5"
    url = f"{_BASE_URL.format(language=language)}/{h5_file}?download=true"
    return url, os.path.join(_CACHE_DIR, language, h5_file)


class NvidiaDataset(BaseDataset):
    name = "nvidia"

    def pathname(self, opts: Dict[str, str]) -> str:
        return f"nvidia_ocr_{opts.get('language', 'en')}"

    def detection(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Generator[Tuple[List[Image.Image], List], None, None]:
        h5_files, language, msl, chunk_size = _parse_opts(opts)
        sample_count = 0

        for h5_file in h5_files:
            if sample_count >= max_rows:
                break
            url, local_path = _local_path(language, h5_file)
            try:
                local_path = download_h5_file(url, local_path)
                print(f"Streaming data from {h5_file}...")
                remaining = max_rows - sample_count
                for img_batch, bbox_batch in load_h5_detection_data(
                    local_path, remaining, max_size_limit=msl, chunk_size=chunk_size
                ):
                    yield img_batch, bbox_batch
                    sample_count += len(img_batch)
                    if sample_count >= max_rows:
                        break
            except Exception as e:
                print(f"Error processing {h5_file}: {e}")
                continue

    def recognition(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Tuple[List[Image.Image], List[str], List]:
        h5_files, language, msl, _ = _parse_opts(opts)
        images: List[Image.Image] = []
        texts: List[str] = []
        bboxes: List = []
        sample_count = 0

        for h5_file in h5_files:
            if sample_count >= max_rows:
                break
            url, local_path = _local_path(language, h5_file)
            try:
                local_path = download_h5_file(url, local_path)
                print(f"Loading data from {h5_file}...")
                remaining = max_rows - sample_count
                img_batch, text_batch, bbox_batch = load_h5_recognition_data(
                    local_path, remaining, max_size_limit=msl
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
