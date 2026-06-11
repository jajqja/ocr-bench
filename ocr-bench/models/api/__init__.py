from models.base import BaseEndToEndModel

REGISTRY: dict = {
    # "claude": ClaudeOCRModel,  # uncomment after implementing models/api/claude.py
}


def load(name: str, **kwargs) -> BaseEndToEndModel:
    if name not in REGISTRY:
        raise ValueError(f"Unknown API model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
