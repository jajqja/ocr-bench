"""Dataset từ một file PDF cục bộ (hỗ trợ cả detection và recognition).

Opts:
    path  (bắt buộc)  Đường dẫn tới file PDF.
"""

from pathlib import Path
from typing import Dict, Generator, List, Tuple

from PIL import Image
from surya.input.processing import (
    convert_if_not_rgb,
    get_page_images,
    open_pdf,
)

from utils.bbox import get_pdf_lines, rescale_bbox
from utils.datasets.base import BaseDataset


def _require_path(opts: Dict[str, str]) -> str:
    path = opts.get("path")
    if not path:
        raise ValueError("Dataset 'pdf' cần --opt path=<đường dẫn PDF>")
    return path


class PdfDataset(BaseDataset):
    name = "pdf"

    def pathname(self, opts: Dict[str, str]) -> str:
        return Path(_require_path(opts)).stem

    def detection(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Generator[Tuple[List[Image.Image], List], None, None]:
        pdf_path = _require_path(opts)
        doc = open_pdf(pdf_path)
        page_count = min(len(doc), max_rows)
        images = get_page_images(doc, list(range(page_count)))
        doc.close()
        image_sizes = [img.size for img in images]
        correct_boxes = get_pdf_lines(pdf_path, image_sizes)
        yield images, correct_boxes

    def recognition(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Tuple[List[Image.Image], List[str], List]:
        pdf_path = _require_path(opts)
        doc = open_pdf(pdf_path)
        page_count = min(len(doc), max_rows)
        page_indices = list(range(page_count))
        images = convert_if_not_rgb(get_page_images(doc, page_indices))

        ground_truth_texts: List[str] = []
        page_bboxes: List = []
        for idx, image in zip(page_indices, images):
            page = doc[idx]
            blocks = page.get_text("dict", sort=True)["blocks"]
            page_box = page.bound()
            page_size = (page_box[2] - page_box[0], page_box[3] - page_box[1])

            line_bboxes = []
            for block in blocks:
                for line in block.get("lines", []):
                    text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    )
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
