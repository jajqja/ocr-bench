"""Tiện ích dùng chung cho pipeline crawl PDF tiếng Việt.

Tất cả script trong thư mục này độc lập với package `ocr-bench`; chúng chỉ
phụ thuộc `requests` và `pymupdf`. Đường dẫn output mặc định nằm ngay trong
`crawl-data/` để dễ kiểm tra.
"""

import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator

# Thư mục gốc của pipeline = thư mục chứa file này.
ROOT = Path(__file__).resolve().parent

PDF_DIR = ROOT / "pdfs"
JSON_DIR = ROOT / "json"
PREVIEW_DIR = ROOT / "preview"

CANDIDATES_FILE = ROOT / "candidates.jsonl"
MANIFEST_FILE = ROOT / "manifest.jsonl"
KEYWORDS_FILE = ROOT / "keywords.txt"

# Email cho "polite pool" của OpenAlex (tốc độ ổn định hơn khi gắn mailto).
CONTACT_EMAIL = os.environ.get("CRAWL_CONTACT_EMAIL", "dangminhhoang@newai.vn")
USER_AGENT = f"ocr-bench-crawler/0.1 (mailto:{CONTACT_EMAIL})"


def ensure_dirs() -> None:
    """Tạo các thư mục output nếu chưa có."""
    for d in (PDF_DIR, JSON_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC để dấu tiếng Việt không bị tách rời."""
    return unicodedata.normalize("NFC", text)


def stable_id(*parts: str) -> str:
    """Sinh id ngắn, ổn định từ các thành phần (vd doi hoặc url)."""
    key = "||".join(p for p in parts if p)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def read_keywords(path: Path = KEYWORDS_FILE) -> list:
    """Đọc keywords.txt, bỏ qua dòng trống và dòng bắt đầu bằng '#'."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file từ khóa: {path}")
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    return queries


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    """Ghi danh sách dict ra file JSONL. Trả về số dòng đã ghi."""
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Dict]:
    """Đọc file JSONL, yield từng dict."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
