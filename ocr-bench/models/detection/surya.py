from typing import List

from PIL import Image
from surya.detection import DetectionPredictor

from models.base import BaseDetectionModel


class SuryaDetectionModel(BaseDetectionModel):
    def __init__(self, checkpoint: str):
        self.predictor = DetectionPredictor(checkpoint=checkpoint)

    def predict(
        self, images: List[Image.Image], batch_size: int = 8
    ) -> List[List[List[float]]]:
        results = self.predictor(images, batch_size=batch_size)
        return [[s.bbox for s in result.bboxes] for result in results]
