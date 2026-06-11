from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class BaseDetectionModel(ABC):
    @abstractmethod
    def predict(
        self, images: List[Image.Image], batch_size: int = 8
    ) -> List[List[List[float]]]:
        """Run detection on a batch of images.

        Returns:
            Per-image list of bboxes, each bbox as [x1, y1, x2, y2].
        """
        pass


class BaseRecognitionModel(ABC):
    @abstractmethod
    def predict(
        self,
        images: List[Image.Image],
        bboxes: List[List[List[int]]],
        batch_size: int = 8,
    ) -> List[str]:
        """Run recognition on text regions.

        Args:
            images: List of PIL images.
            bboxes: Per-image list of [x1, y1, x2, y2] bboxes.

        Returns:
            Flat list of recognized text strings across all images and bboxes.
        """
        pass


class BaseEndToEndModel(ABC):
    @abstractmethod
    def predict(
        self, images: List[Image.Image], batch_size: int = 8
    ) -> List[List[dict]]:
        """Run end-to-end OCR on a batch of images.

        Returns:
            Per-image list of {"text": str, "bbox": [x1, y1, x2, y2]}.
            bbox may be None for models that do not return location.
        """
        pass
