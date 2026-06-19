"""Bước 2 — Download: tải PDF từ candidates.jsonl về thư mục pdfs/.

Có rate-limit, retry/backoff, kiểm tra Content-Type và giới hạn kích thước.
File đã tải sẽ được bỏ qua (idempotent). Kết quả tải ghi vào download_log.jsonl.

Ví dụ:
    python crawl-data/download.py --max 200 --sleep 1.0
"""

import argparse
import time
from typing import Dict, Optional, Tuple

import requests

from common import (
    CANDIDATES_FILE,
    PDF_DIR,
    ROOT,
    USER_AGENT,
    ensure_dirs,
    read_jsonl,
    write_jsonl,
)

DOWNLOAD_LOG = ROOT / "download_log.jsonl"
PDF_MAGIC = b"%PDF-"


def download_one(
    cand: Dict,
    session: requests.Session,
    max_bytes: int,
    retries: int,
    timeout: int,
) -> Tuple[str, Optional[str]]:
    """Tải 1 PDF. Trả về (status, đường-dẫn-tương-đối hoặc None)."""
    out_path = PDF_DIR / f"{cand['id']}.pdf"
    if out_path.exists() and out_path.stat().st_size > 0:
        return "cached", str(out_path.relative_to(ROOT))

    url = cand["pdf_url"]
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                clen = int(resp.headers.get("Content-Length") or 0)
                if clen and clen > max_bytes:
                    return "too_large", None

                chunks = bytearray()
                for chunk in resp.iter_content(chunk_size=65536):
                    chunks.extend(chunk)
                    if len(chunks) > max_bytes:
                        return "too_large", None

                # Phải đúng là file PDF (một số link OA trả về HTML landing page).
                if not bytes(chunks[:5]) == PDF_MAGIC and "pdf" not in ctype:
                    return "not_pdf", None

                out_path.write_bytes(bytes(chunks))
                return "ok", str(out_path.relative_to(ROOT))
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2**attempt)  # backoff: 2s, 4s, ...
    return f"error:{last_err[:120]}", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải PDF candidate về local")
    parser.add_argument("--max", type=int, default=200, help="Số PDF tối đa cần tải")
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Nghỉ giữa các tải (s)"
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--max-mb", type=float, default=40.0, help="Bỏ qua PDF lớn hơn (MB)"
    )
    args = parser.parse_args()

    ensure_dirs()
    max_bytes = int(args.max_mb * 1024 * 1024)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})

    candidates = list(read_jsonl(CANDIDATES_FILE))
    logs = []
    ok = 0
    for i, cand in enumerate(candidates, 1):
        if ok >= args.max:
            break
        status, rel_path = download_one(
            cand, session, max_bytes, args.retries, args.timeout
        )
        logs.append(
            {
                "id": cand["id"],
                "pdf_url": cand["pdf_url"],
                "status": status,
                "path": rel_path,
            }
        )
        is_new = status == "ok"
        if status in ("ok", "cached"):
            ok += 1
        print(f"[{i}/{len(candidates)}] {status:>10}  {cand['title'][:55]}")
        if is_new:
            time.sleep(args.sleep)

    write_jsonl(DOWNLOAD_LOG, logs)
    from collections import Counter

    summary = Counter(log["status"].split(":")[0] for log in logs)
    print(f"\nKết quả: {dict(summary)}")
    print(f"Log -> {DOWNLOAD_LOG}")


if __name__ == "__main__":
    main()
