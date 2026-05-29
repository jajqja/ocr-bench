from huggingface_hub import snapshot_download
from pathlib import Path
import click
from typing import Optional


def hg_download(repo_id: str, local_dir: Path, hf_token: Optional[str] = None) -> None:
    """Download model from HuggingFace Hub."""
    # Kiểm tra xem thư mục có tồn tại và có file nào bên trong chưa
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Model already present at {local_dir}")
        return

    print(f"Downloading OCR model {repo_id} to {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=hf_token or None,
    )


# Thêm decorator @click.command() vào đây
@click.command()
@click.option(
    "--repo_id",
    type=str,
    required=True,
    help="HuggingFace repo ID (e.g., 'username/model-name')",
)
@click.option(
    "--local_dir",
    type=str,
    default="./model_path/model",
    help="Local directory to save model",
)
@click.option(
    "--hf_token", type=str, default=None, help="HuggingFace token for private models"
)
def download_detection(repo_id: str, local_dir: str, hf_token: Optional[str] = None):
    """Download detection model from HuggingFace."""
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    hg_download(repo_id, local_path, hf_token)
    print(f"✓ Model downloaded to {local_dir}")


if __name__ == "__main__":
    download_detection()
