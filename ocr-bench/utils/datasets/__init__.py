"""Registry các dataset của benchmark.

Thêm dataset mới: tạo một module với lớp con của BaseDataset rồi đăng ký vào
REGISTRY bên dưới. Các script evaluate chỉ cần ``--dataset <name>`` và truyền
tham số riêng qua ``--opt key=value`` (lặp lại được).
"""

from typing import Dict, Iterable

from utils.datasets.base import BaseDataset
from utils.datasets.doclaynet import DoclaynetDataset
from utils.datasets.folder import FolderDataset
from utils.datasets.nvidia import NvidiaDataset
from utils.datasets.pdf import PdfDataset
from utils.datasets.pdfa import PdfaDataset

REGISTRY: Dict[str, type] = {
    PdfDataset.name: PdfDataset,
    DoclaynetDataset.name: DoclaynetDataset,
    PdfaDataset.name: PdfaDataset,
    NvidiaDataset.name: NvidiaDataset,
    FolderDataset.name: FolderDataset,
}


def load_dataset(name: str) -> BaseDataset:
    """Khởi tạo dataset theo tên đã đăng ký."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def parse_opts(pairs: Iterable[str]) -> Dict[str, str]:
    """Chuyển danh sách 'key=value' (từ --opt) thành dict.

    Ví dụ: ("language=vi", "max_size_limit=1600") -> {"language": "vi",
    "max_size_limit": "1600"}.
    """
    opts: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"--opt không hợp lệ: '{item}'. Định dạng phải là key=value."
            )
        key, value = item.split("=", 1)
        opts[key.strip()] = value.strip()
    return opts


__all__ = ["REGISTRY", "load_dataset", "parse_opts", "BaseDataset"]
