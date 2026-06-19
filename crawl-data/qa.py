"""Bước 5 — QA: vẽ bbox lên trang để kiểm tra mắt + xuất manifest & thống kê.

- preview/: render một trang mẫu của mỗi PDF, vẽ line bbox lên để kiểm tra
  bbox có khớp text không.
- manifest.jsonl: một dòng cho mỗi PDF đã trích (metadata + thống kê).

Ví dụ:
    python crawl-data/qa.py --preview-pages 1 --max-preview 6
"""

import argparse
import io
import json
from typing import Dict, List

import pymupdf
from PIL import Image, ImageDraw

from common import JSON_DIR, MANIFEST_FILE, PREVIEW_DIR, ROOT, ensure_dirs, write_jsonl


def render_page(pdf_path: str, page_index: int, dpi: int) -> Image.Image:
    """Render một trang PDF thành ảnh PIL ở DPI cho trước."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def draw_bboxes(img: Image.Image, lines: List[Dict]) -> Image.Image:
    """Vẽ các line bbox (đỏ) lên ảnh."""
    draw = ImageDraw.Draw(img)
    for ln in lines:
        x1, y1, x2, y2 = map(int, ln["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return img


def doc_stats(doc: Dict) -> Dict:
    pages = doc["pages"]
    n_lines = sum(len(p["lines"]) for p in pages)
    n_chars = sum(len(ln["text"]) for p in pages for ln in p["lines"])
    return {
        "id": doc["id"],
        "title": doc.get("title", ""),
        "doi": doc.get("doi", ""),
        "license": doc.get("license"),
        "source": doc.get("source"),
        "render_dpi": doc.get("render_dpi"),
        "n_pages": len(pages),
        "n_lines": n_lines,
        "n_chars": n_chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QA: vẽ bbox + xuất manifest")
    parser.add_argument(
        "--preview-pages",
        type=int,
        nargs="+",
        default=[1],
        help="Các page_index sẽ render preview (mặc định trang thứ 2)",
    )
    parser.add_argument(
        "--max-preview", type=int, default=6, help="Số PDF render preview"
    )
    args = parser.parse_args()

    ensure_dirs()
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print("Chưa có file JSON nào. Hãy chạy extract.py trước.")
        return

    manifest: List[Dict] = []
    previews = 0
    for jf in json_files:
        doc = json.loads(jf.read_text(encoding="utf-8"))
        manifest.append(doc_stats(doc))

        if previews >= args.max_preview:
            continue
        pdf_path = ROOT / doc["pdf_path"]
        dpi = doc.get("render_dpi", 200)
        for pidx in args.preview_pages:
            page = next((p for p in doc["pages"] if p["page_index"] == pidx), None)
            if page is None:
                continue
            try:
                img = render_page(str(pdf_path), pidx, dpi)
                img = draw_bboxes(img, page["lines"])
                out = PREVIEW_DIR / f"{doc['id']}_p{pidx}.png"
                img.save(out)
                print(f"preview -> {out.relative_to(ROOT)} ({len(page['lines'])} bbox)")
            except Exception as e:
                print(f"LỖI preview {doc['id']} trang {pidx}: {e}")
        previews += 1

    write_jsonl(MANIFEST_FILE, manifest)

    # Thống kê tổng.
    from collections import Counter

    tot_pages = sum(m["n_pages"] for m in manifest)
    tot_lines = sum(m["n_lines"] for m in manifest)
    lic = Counter(str(m["license"]) for m in manifest)
    print("\n===== THỐNG KÊ =====")
    print(f"PDF:    {len(manifest)}")
    print(f"Trang:  {tot_pages}")
    print(f"Dòng:   {tot_lines}")
    print(f"License: {dict(lic)}")
    print(f"Manifest -> {MANIFEST_FILE}")
    print(f"Preview  -> {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
