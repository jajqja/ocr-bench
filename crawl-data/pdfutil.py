"""Tiện ích đọc PDF bằng PyMuPDF: thống kê text, phát hiện scan, trích line bbox.

Dùng chung cho filter_pdf.py và extract.py. Chỉ phụ thuộc `pymupdf`.
"""

import re
import unicodedata
from typing import Dict, List, Tuple

import pymupdf

from common import normalize_text

_WS_RE = re.compile(r"\s+")

# Cờ get_text giống logic trong ocr-bench/utils/bbox.py (bỏ ligatures/images).
_TEXT_FLAGS = (
    pymupdf.TEXTFLAGS_DICT
    & ~pymupdf.TEXT_PRESERVE_LIGATURES
    & ~pymupdf.TEXT_PRESERVE_IMAGES
)

# Chữ riêng của tiếng Việt không phải chỉ là dấu thanh (horn/breve/stroke).
_VI_SPECIAL = set("ăâđêôơưĂÂĐÊÔƠƯ")
# Từ chức năng tiếng Việt phổ biến (tín hiệu ngôn ngữ mạnh, không dấu dễ trùng).
_VI_WORDS = {
    "của",
    "và",
    "các",
    "được",
    "trong",
    "người",
    "những",
    "một",
    "có",
    "không",
    "cho",
    "này",
    "đến",
    "với",
    "là",
    "để",
    "khi",
    "đã",
    "tại",
}


def page_text_lines(page: pymupdf.Page) -> List[Tuple[List[float], str]]:
    """Trả về list (bbox_point, text) cho từng dòng text trên trang.

    bbox ở hệ tọa độ điểm (point) của PDF. text đã chuẩn hóa NFC.
    """
    blocks = page.get_text("dict", sort=True, flags=_TEXT_FLAGS)["blocks"]
    lines = []
    for block in blocks:
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = normalize_text(text)
            if text.strip():
                lines.append((list(line["bbox"]), text))
    return lines


def clean_text(text: str) -> str:
    """Làm sạch text một dòng:

    - Bỏ ký tự điều khiển/định dạng/Private-Use-Area (category Unicode 'C*'),
      ví dụ glyph icon \\uf02a của font symbol — vô nghĩa trong ground-truth.
    - Gom mọi khoảng trắng (kể cả tab/xuống dòng) thành một dấu cách.
    - Bỏ khoảng trắng đầu/cuối dòng.
    """
    text = "".join(
        ch for ch in text if ch in "\t\n\r" or unicodedata.category(ch)[0] != "C"
    )
    return _WS_RE.sub(" ", text).strip()


def is_noise_text(text: str) -> bool:
    """True nếu dòng chỉ gồm ký tự phân cách/dấu (không có chữ hay số),
    ví dụ '_______', '----', '......', '•'. Dòng có số như
    'ĐT.: 84-934401212.' vẫn được giữ vì chứa chữ số."""
    return not any(ch.isalnum() for ch in text)


def pdf_stats(path: str, max_pages: int = 50) -> Dict:
    """Thu thập thống kê để quyết định PDF có phải digital + tiếng Việt không."""
    doc = pymupdf.open(path)
    n_pages = min(len(doc), max_pages)

    total_chars = 0
    pages_with_text = 0
    pages_image_only = 0
    sample_text_parts = []

    for i in range(n_pages):
        page = doc[i]
        lines = page_text_lines(page)
        page_chars = sum(len(t) for _, t in lines)
        total_chars += page_chars
        if page_chars >= 50:
            pages_with_text += 1
            sample_text_parts.append(" ".join(t for _, t in lines))
        # Trang "scan": hầu như không có text nhưng có ảnh phủ trang.
        if page_chars < 20 and page.get_images():
            pages_image_only += 1

    doc.close()
    sample_text = normalize_text(" ".join(sample_text_parts))
    return {
        "n_pages": n_pages,
        "total_chars": total_chars,
        "mean_chars_per_page": total_chars / n_pages if n_pages else 0,
        "pages_with_text": pages_with_text,
        "pages_image_only": pages_image_only,
        "sample_text": sample_text[:15000],
    }


def vietnamese_score(text: str) -> float:
    """Điểm 0..1 ước lượng mức độ 'tiếng Việt' của một đoạn text.

    Dựa trên mật độ dấu thanh (đếm qua phân tách NFD nên không phụ thuộc cách
    mã hóa, không phân biệt hoa/thường), cộng chữ riêng tiếng Việt và từ chức
    năng phổ biến. Cách này bền với tài liệu song ngữ hơn langdetect (vốn dễ
    đoán nhầm khi phần đầu là abstract tiếng Anh).
    """
    text = text.strip()
    if not text:
        return 0.0

    # Tách dấu thanh ra khỏi chữ cái cơ bản để đếm trực tiếp.
    nfd = unicodedata.normalize("NFD", text)
    base_letters = [c for c in nfd if c.isalpha() and not unicodedata.combining(c)]
    if not base_letters:
        return 0.0
    combining = sum(1 for c in nfd if unicodedata.combining(c))

    # Tín hiệu 1: mật độ dấu thanh (tiếng Việt thường ~0.2-0.4).
    tone_ratio = combining / len(base_letters)
    # Tín hiệu 2: chữ riêng tiếng Việt (ă â đ ê ô ơ ư).
    special_ratio = sum(1 for c in text if c in _VI_SPECIAL) / len(base_letters)
    # Tín hiệu 3: tỷ lệ từ chức năng tiếng Việt.
    words = text.lower().split()
    word_ratio = (
        sum(1 for w in words if w.strip(".,;:()[]\"'") in _VI_WORDS) / len(words)
        if words
        else 0.0
    )

    return min(1.0, tone_ratio * 3 + special_ratio * 4 + word_ratio * 5)
