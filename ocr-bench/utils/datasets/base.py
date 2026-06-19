"""Giao diện chung cho mọi dataset của benchmark.

Mỗi dataset là một lớp con của ``BaseDataset``, tự đóng gói cách load dữ liệu
và các tham số riêng (truyền qua dict ``opts`` lấy từ CLI ``--opt key=value``).
Nhờ vậy thêm dataset mới chỉ cần thêm một file + đăng ký vào REGISTRY, không
phải sửa các script evaluate.

Quy ước trả về:
- ``detection`` -> generator các chunk ``(images, line_bboxes)`` (cho phép
  stream dataset lớn theo lô).
- ``recognition`` -> một bộ ``(images, texts, line_bboxes)`` (nạp toàn bộ).

``line_bboxes`` là list theo từng ảnh, mỗi phần tử là list bbox ``[x1,y1,x2,y2]``.
"""

from abc import ABC
from typing import Dict, Generator, List, Tuple

from PIL import Image

DetectionChunk = Tuple[List[Image.Image], List[List[List[float]]]]
RecognitionData = Tuple[List[Image.Image], List[str], List[List[List[float]]]]


class BaseDataset(ABC):
    """Lớp cơ sở cho dataset. Đặt ``name`` ở lớp con để đăng ký."""

    name: str = ""

    def detection(
        self, max_rows: int, opts: Dict[str, str]
    ) -> Generator[DetectionChunk, None, None]:
        """Yield các chunk (images, line_bboxes) cho bài toán detection."""
        raise NotImplementedError(f"Dataset '{self.name}' không hỗ trợ detection.")

    def recognition(self, max_rows: int, opts: Dict[str, str]) -> RecognitionData:
        """Trả về (images, texts, line_bboxes) cho bài toán recognition."""
        raise NotImplementedError(f"Dataset '{self.name}' không hỗ trợ recognition.")

    def pathname(self, opts: Dict[str, str]) -> str:
        """Tên dùng để đặt file kết quả. Mặc định là tên dataset."""
        return self.name


def get_full_image_bboxes(images: List[Image.Image]) -> List[List[List[int]]]:
    """Tạo một bbox phủ toàn ảnh cho mỗi ảnh (recognition cả ảnh)."""
    return [[[0, 0, image.size[0], image.size[1]]] for image in images]
