"""Claude API end-to-end OCR model stub.

To implement:
1. pip install anthropic
2. Set ANTHROPIC_API_KEY environment variable
3. Implement predict() below using the Anthropic vision API
4. Register in models/api/__init__.py: REGISTRY["claude"] = ClaudeOCRModel
"""
import os
from typing import List

from PIL import Image

from models.base import BaseEndToEndModel


class ClaudeOCRModel(BaseEndToEndModel):
    def __init__(self, model_id: str = "claude-sonnet-4-6"):
        raise NotImplementedError(
            "ClaudeOCRModel is not yet implemented. See models/api/claude.py."
        )

    def predict(
        self, images: List[Image.Image], batch_size: int = 4
    ) -> List[List[dict]]:
        """Returns per-image list of {"text": str, "bbox": None}."""
        raise NotImplementedError
