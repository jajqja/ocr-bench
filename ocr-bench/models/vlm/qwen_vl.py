"""Qwen-VL end-to-end OCR model stub.

To implement:
1. pip install transformers torch accelerate
2. Download model: python ocr-bench/utils/model_download.py --repo_id Qwen/Qwen-VL --local_dir ./model_path/qwen_vl
3. Implement predict() below
4. Register in models/vlm/__init__.py: REGISTRY["qwen_vl"] = QwenVLModel
"""

from typing import List

from PIL import Image

from models.base import BaseEndToEndModel


class QwenVLModel(BaseEndToEndModel):
    def __init__(self, checkpoint: str):
        raise NotImplementedError(
            "QwenVLModel is not yet implemented. See models/vlm/qwen_vl.py."
        )

    def predict(
        self, images: List[Image.Image], batch_size: int = 4
    ) -> List[List[dict]]:
        """Returns per-image list of {"text": str, "bbox": [x1, y1, x2, y2] | None}."""
        raise NotImplementedError
