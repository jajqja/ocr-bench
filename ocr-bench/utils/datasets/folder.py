"""Dataset recognition từ thư mục cục bộ (chỉ hỗ trợ recognition).

Bố cục mong đợi:
    data_dir/
        <image_folder>/ image_1.jpg ...
        <label_file>     # mỗi dòng: "image_1.jpg<TAB>nhãn"

Opts:
    data_dir      (bắt buộc)          Thư mục dataset.
    image_folder  (mặc định "images")  Thư mục con chứa ảnh.
    label_file    (mặc định "labels.txt") Tên file nhãn.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from surya.input.processing import convert_if_not_rgb

from utils.datasets.base import BaseDataset, get_full_image_bboxes


class FolderDataset(BaseDataset):
    name = "folder"

    def _data_dir(self, opts: Dict[str, str]) -> str:
        data_dir = opts.get("data_dir")
        if not data_dir:
            raise ValueError("Dataset 'folder' cần --opt data_dir=<thư mục>")
        return data_dir

    def pathname(self, opts: Dict[str, str]) -> str:
        return Path(self._data_dir(opts)).name

    def recognition(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Tuple[List[Image.Image], List[str], List]:
        data_path = Path(self._data_dir(opts))
        images_dir = data_path / opts.get("image_folder", "images")
        labels_path = data_path / opts.get("label_file", "labels.txt")

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        images: List[Image.Image] = []
        texts: List[str] = []
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
