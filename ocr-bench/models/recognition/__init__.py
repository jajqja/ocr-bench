from models.base import BaseRecognitionModel
from models.recognition.surya import SuryaRecognitionModel

REGISTRY: dict = {
    "surya": SuryaRecognitionModel,
}


def load(name: str, **kwargs) -> BaseRecognitionModel:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown recognition model '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)
