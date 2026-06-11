from models.base import BaseDetectionModel
from models.detection.surya import SuryaDetectionModel

REGISTRY: dict = {
    "surya": SuryaDetectionModel,
}


def load(name: str, **kwargs) -> BaseDetectionModel:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown detection model '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)
