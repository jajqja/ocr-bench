"""Dataset pixparse/pdfa-eng-wds (HuggingFace) — detection và recognition.

Opts:
    name  (mặc định "pixparse/pdfa-eng-wds")  Tên dataset HuggingFace.
"""

import json
from typing import Dict, Generator, List, Tuple

import datasets as hf_datasets
from PIL import Image
from pdf2image import convert_from_bytes
from surya.input.processing import convert_if_not_rgb

from utils.datasets.base import BaseDataset

_DEFAULT = "pixparse/pdfa-eng-wds"


def _iter_pages(sample):
    """Yield (img, page_data) cho từng trang của một sample pdfa."""
    pdf_pages = convert_if_not_rgb(convert_from_bytes(sample["pdf"], dpi=300))
    metadata = (
        json.loads(sample["ocr"]) if isinstance(sample["ocr"], str) else sample["ocr"]
    )
    for page_idx, page_data in enumerate(metadata.get("pages", [])):
        if page_idx >= len(pdf_pages):
            break
        yield pdf_pages[page_idx], page_data


class PdfaDataset(BaseDataset):
    name = "pdfa"

    def pathname(self, opts: Dict[str, str]) -> str:
        return opts.get("name", _DEFAULT).replace("/", "_")

    def detection(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Generator[Tuple[List[Image.Image], List], None, None]:
        dataset_name = opts.get("name", _DEFAULT)
        dataset = hf_datasets.load_dataset(dataset_name, split="train", streaming=False)
        images: List[Image.Image] = []
        bboxes: List = []

        for idx, sample in enumerate(dataset):
            if idx >= max_rows:
                break
            try:
                for img, page_data in _iter_pages(sample):
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

        yield images, bboxes

    def recognition(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Tuple[List[Image.Image], List[str], List]:
        dataset_name = opts.get("name", _DEFAULT)
        dataset = hf_datasets.load_dataset(dataset_name, split="train", streaming=False)
        images: List[Image.Image] = []
        texts: List[str] = []
        bboxes: List = []

        for idx, sample in enumerate(dataset):
            if idx >= max_rows:
                break
            try:
                for img, page_data in _iter_pages(sample):
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
