"""Bước 3 — Filter: lọc các PDF đã tải, chỉ giữ PDF *digital* + *tiếng Việt*.

- Digital (born-digital): bốc được text trực tiếp, không phải ảnh scan.
  Tiêu chí: số ký tự trung bình/trang đủ lớn và tỷ lệ trang chỉ-có-ảnh thấp.
- Tiếng Việt: điểm vietnamese_score trên text mẫu vượt ngưỡng.

Kết quả ghi filtered.jsonl (gồm cả bản ghi accept và reject kèm lý do).

Ví dụ:
    python crawl-data/filter_pdf.py --min-chars 200 --min-vi 0.5
"""

import argparse
from typing import Dict, List

from common import (
    PDF_DIR,
    ROOT,
    ensure_dirs,
    read_jsonl,
    write_jsonl,
    CANDIDATES_FILE,
)
from pdfutil import pdf_stats, vietnamese_score

FILTERED_FILE = ROOT / "filtered.jsonl"


def classify(
    stats: Dict, min_chars: float, min_text_pages_ratio: float, min_vi: float
) -> Dict:
    """Quyết định accept/reject dựa trên thống kê PDF."""
    reasons = []
    n_pages = stats["n_pages"]
    text_pages_ratio = stats["pages_with_text"] / n_pages if n_pages else 0
    image_ratio = stats["pages_image_only"] / n_pages if n_pages else 0

    if stats["mean_chars_per_page"] < min_chars:
        reasons.append("ít_text(scan?)")
    if text_pages_ratio < min_text_pages_ratio:
        reasons.append("ít_trang_có_text")
    if image_ratio > 0.5:
        reasons.append("đa_số_ảnh(scan)")

    vi = vietnamese_score(stats["sample_text"])
    if vi < min_vi:
        reasons.append(f"không_đủ_tiếng_việt({vi:.2f})")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "vietnamese_score": round(vi, 3),
        "text_pages_ratio": round(text_pages_ratio, 3),
        "image_only_ratio": round(image_ratio, 3),
        "mean_chars_per_page": round(stats["mean_chars_per_page"], 1),
        "n_pages": n_pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lọc PDF digital + tiếng Việt")
    parser.add_argument("--min-chars", type=float, default=200.0)
    parser.add_argument("--min-text-pages", type=float, default=0.5)
    parser.add_argument("--min-vi", type=float, default=0.5)
    parser.add_argument("--max-pages", type=int, default=50, help="Số trang quét/PDF")
    args = parser.parse_args()

    ensure_dirs()

    # Tra cứu metadata candidate theo id để bổ sung thông tin vào filtered.jsonl.
    meta = {c["id"]: c for c in read_jsonl(CANDIDATES_FILE)}

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    rows: List[Dict] = []
    accepted = 0
    for i, path in enumerate(pdf_paths, 1):
        pdf_id = path.stem
        try:
            stats = pdf_stats(str(path), max_pages=args.max_pages)
        except Exception as e:
            rows.append({"id": pdf_id, "accepted": False, "reasons": [f"lỗi_đọc:{e}"]})
            print(f"[{i}/{len(pdf_paths)}] LỖI {pdf_id}: {e}")
            continue

        verdict = classify(stats, args.min_chars, args.min_text_pages, args.min_vi)
        c = meta.get(pdf_id, {})
        row = {
            "id": pdf_id,
            "pdf_path": str(path.relative_to(ROOT)),
            "title": c.get("title", ""),
            "doi": c.get("doi", ""),
            "license": c.get("license"),
            "source": c.get("source"),
            **verdict,
        }
        rows.append(row)
        if verdict["accepted"]:
            accepted += 1
        tag = "OK " if verdict["accepted"] else "BỎ"
        why = "" if verdict["accepted"] else " <- " + ",".join(verdict["reasons"])
        print(
            f"[{i}/{len(pdf_paths)}] {tag} vi={verdict['vietnamese_score']:.2f} "
            f"chars/trang={verdict['mean_chars_per_page']:.0f} "
            f"{c.get('title','')[:45]}{why}"
        )

    write_jsonl(FILTERED_FILE, rows)
    print(f"\nChấp nhận {accepted}/{len(pdf_paths)} PDF -> {FILTERED_FILE}")


if __name__ == "__main__":
    main()
