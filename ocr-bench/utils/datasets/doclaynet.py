"""Dataset vikp/doclaynet_bench (HuggingFace) — detection và recognition.

Opts:
    name  (mặc định "vikp/doclaynet_bench")  Tên dataset HuggingFace.
"""

from typing import Dict, Generator, List, Tuple

import datasets as hf_datasets
from PIL import Image
from surya.input.processing import convert_if_not_rgb

from utils.bbox import rescale_bbox
from utils.datasets.base import BaseDataset

_DEFAULT = "vikp/doclaynet_bench"


class DoclaynetDataset(BaseDataset):
    name = "doclaynet"

    def pathname(self, opts: Dict[str, str]) -> str:
        return opts.get("name", _DEFAULT).replace("/", "_")

    def _load(self, max_rows: int, opts: Dict[str, str]):
        dataset_name = opts.get("name", _DEFAULT)
        dataset = hf_datasets.load_dataset(dataset_name, split=f"train[:{max_rows}]")
        images = convert_if_not_rgb(list(dataset["image"]))
        correct_boxes = []
        for i, boxes in enumerate(dataset["bboxes"]):
            img_size = images[i].size
            correct_boxes.append(
                [rescale_bbox(b, (1000, 1000), img_size) for b in boxes]
            )
        return dataset, images, correct_boxes

    def detection(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Generator[Tuple[List[Image.Image], List], None, None]:
        _, images, correct_boxes = self._load(max_rows, opts)
        yield images, correct_boxes

    def recognition(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Tuple[List[Image.Image], List[str], List]:
        dataset, images, correct_boxes = self._load(max_rows, opts)
        texts = [word for image_words in dataset["words"] for word in image_words]
        return images, texts, correct_boxes
