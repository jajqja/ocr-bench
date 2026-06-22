"""Bước 4 — Extract: trích line bbox + text từ các PDF đã được chấp nhận.

Với mỗi PDF accepted trong filtered.jsonl, lấy từng dòng text kèm bbox (tái sử
dụng logic get_text('dict', sort=True) như ocr-bench/utils/bbox.py). bbox được
quy đổi về hệ tọa độ pixel ở một DPI cố định để tái lập được khi render ảnh.

Output: json/<id>.json — mỗi file một PDF.

Ví dụ:
    python crawl-data/extract.py --dpi 200
"""

import argparse
import json
from typing import Dict, List

import pymupdf

from common import JSON_DIR, ROOT, ensure_dirs, read_jsonl
from filter_pdf import FILTERED_FILE
from pdfutil import clean_text, dedup_overlapping, is_noise_text, page_text_lines


def extract_pdf(
    pdf_path: str,
    dpi: int,
    max_pages: int,
    keep_noise: bool = False,
    keep_dups: bool = False,
    min_size: float = 3.0,
) -> List[Dict]:
    """Trích line bbox (pixel @dpi) + text cho từng trang.

    Mặc định làm sạch text (gom khoảng trắng thừa), bỏ các dòng chỉ gồm ký tự
    phân cách như '_______', khử dòng vẽ trùng (overprint), và bỏ chữ siêu nhỏ
    < min_size point (măng-sét trang trí). Đặt keep_noise/keep_dups=True hoặc
    min_size=0 để giữ nguyên.
    """
    scale = dpi / 72.0  # PDF point = 1/72 inch
    doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        rect = page.rect
        width = int(round(rect.width * scale))
        height = int(round(rect.height * scale))

        lines = []
        for bbox_pt, text in page_text_lines(page, min_size=min_size):
            text = clean_text(text)
            if not text or (not keep_noise and is_noise_text(text)):
                continue
            x1, y1, x2, y2 = bbox_pt
            lines.append(
                {
                    "bbox": [
                        int(round(x1 * scale)),
                        int(round(y1 * scale)),
                        int(round(x2 * scale)),
                        int(round(y2 * scale)),
                    ],
                    "text": text,
                }
            )
        if not keep_dups:
            lines = dedup_overlapping(lines)
        pages.append(
            {"page_index": i, "width": width, "height": height, "lines": lines}
        )
    doc.close()
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích line bbox + text ra JSON")
    parser.add_argument("--dpi", type=int, default=200, help="DPI render tham chiếu")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Giữ nguyên các dòng phân cách như '_______' (mặc định bỏ)",
    )
    parser.add_argument(
        "--keep-dups",
        action="store_true",
        help="Giữ nguyên dòng vẽ trùng/overprint (mặc định khử)",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=3.0,
        help="Bỏ dòng có cỡ chữ < giá trị này (point); 0 để tắt (mặc định 3.0)",
    )
    args = parser.parse_args()

    ensure_dirs()

    accepted = [r for r in read_jsonl(FILTERED_FILE) if r.get("accepted")]
    total_lines = 0
    for i, r in enumerate(accepted, 1):
        pdf_path = ROOT / r["pdf_path"]
        try:
            pages = extract_pdf(
                str(pdf_path),
                args.dpi,
                args.max_pages,
                args.keep_noise,
                args.keep_dups,
                args.min_size,
            )
        except Exception as e:
            print(f"[{i}/{len(accepted)}] LỖI {r['id']}: {e}")
            continue

        n_lines = sum(len(p["lines"]) for p in pages)
        total_lines += n_lines
        out = {
            "id": r["id"],
            "pdf_path": r["pdf_path"],
            "source": r.get("source"),
            "doi": r.get("doi"),
            "license": r.get("license"),
            "title": r.get("title"),
            "language": "vi",
            "render_dpi": args.dpi,
            "pages": pages,
        }
        out_path = JSON_DIR / f"{r['id']}.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{i}/{len(accepted)}] {len(pages)} trang, {n_lines} dòng -> "
            f"{out_path.relative_to(ROOT)}"
        )

    print(f"\nĐã trích {len(accepted)} PDF, tổng {total_lines} dòng -> {JSON_DIR}")


if __name__ == "__main__":
    main()
