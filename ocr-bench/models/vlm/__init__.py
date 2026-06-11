from models.base import BaseEndToEndModel

REGISTRY: dict = {
    # "qwen_vl": QwenVLModel,  # uncomment after implementing models/vlm/qwen_vl.py
}


def load(name: str, **kwargs) -> BaseEndToEndModel:
    if name not in REGISTRY:
        raise ValueError(f"Unknown VLM '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
