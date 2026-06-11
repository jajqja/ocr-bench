from typing import List

from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor

from models.base import BaseRecognitionModel


class SuryaRecognitionModel(BaseRecognitionModel):
    def __init__(self, checkpoint: str):
        foundation_model = FoundationPredictor(checkpoint=checkpoint)
        self.predictor = RecognitionPredictor(foundation_model)

    def predict(
        self,
        images: List[Image.Image],
        bboxes: List[List[List[int]]],
        batch_size: int = 8,
    ) -> List[str]:
        predictions = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_bboxes = bboxes[i : i + batch_size]
            results = self.predictor(
                batch_imgs,
                bboxes=batch_bboxes,
                recognition_batch_size=batch_size,
            )
            for result in results:
                predictions.extend(line.text for line in result.text_lines)
        return predictions
