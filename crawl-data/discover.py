"""Bước 1 — Discover: truy vấn OpenAlex để lấy danh sách PDF ứng viên (open-access).

OpenAlex là API học thuật miễn phí, không cần key. Với mỗi từ khóa tiếng Việt
trong keywords.txt, script gọi endpoint /works (lọc open-access, ưu tiên ngôn
ngữ tiếng Việt), trích link PDF và metadata, dedup rồi ghi candidates.jsonl.

Ví dụ:
    python crawl-data/discover.py --per-query 25 --max-total 400
"""

import argparse
import time
from typing import Dict, List, Optional

import requests

from common import (
    CANDIDATES_FILE,
    CONTACT_EMAIL,
    USER_AGENT,
    ensure_dirs,
    normalize_text,
    read_keywords,
    stable_id,
    write_jsonl,
)

OPENALEX_WORKS = "https://api.openalex.org/works"


def _pick_pdf_url(work: Dict) -> Optional[str]:
    """Chọn link PDF tốt nhất từ một bản ghi OpenAlex."""
    for loc_key in ("best_oa_location", "primary_location"):
        loc = work.get(loc_key) or {}
        url = loc.get("pdf_url")
        if url:
            return url
    return (work.get("open_access") or {}).get("oa_url")


def _pick_license(work: Dict) -> Optional[str]:
    for loc_key in ("best_oa_location", "primary_location"):
        loc = work.get(loc_key) or {}
        if loc.get("license"):
            return loc["license"]
    return None


def query_openalex(
    query: str,
    per_query: int,
    language: Optional[str],
    session: requests.Session,
) -> List[Dict]:
    """Trả về danh sách candidate cho một từ khóa."""
    filters = ["open_access.is_oa:true", "has_fulltext:true"]
    if language:
        filters.append(f"language:{language}")

    params = {
        "search": query,
        "filter": ",".join(filters),
        "per-page": min(per_query, 200),
        "select": (
            "id,doi,title,publication_year,language,authorships,"
            "best_oa_location,primary_location,open_access"
        ),
        "mailto": CONTACT_EMAIL,
    }

    resp = session.get(OPENALEX_WORKS, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    candidates = []
    for work in results:
        pdf_url = _pick_pdf_url(work)
        if not pdf_url:
            continue
        title = normalize_text(work.get("title") or "")
        authors = [
            normalize_text((a.get("author") or {}).get("display_name") or "")
            for a in (work.get("authorships") or [])
        ]
        doi = work.get("doi") or ""
        candidates.append(
            {
                "id": stable_id(doi, pdf_url),
                "source": "openalex",
                "openalex_id": work.get("id"),
                "doi": doi,
                "title": title,
                "authors": [a for a in authors if a],
                "year": work.get("publication_year"),
                "language": work.get("language"),
                "license": _pick_license(work),
                "pdf_url": pdf_url,
                "query": query,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover PDF tiếng Việt từ OpenAlex")
    parser.add_argument("--per-query", type=int, default=25, help="Số kết quả/từ khóa")
    parser.add_argument(
        "--max-total", type=int, default=400, help="Tổng số candidate tối đa"
    )
    parser.add_argument(
        "--language",
        default="vi",
        help="Lọc theo mã ngôn ngữ OpenAlex (vd 'vi'); để rỗng để bỏ lọc",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3, help="Nghỉ giữa các request (giây)"
    )
    args = parser.parse_args()

    ensure_dirs()
    queries = read_keywords()
    language = args.language.strip() or None

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    seen = set()
    all_candidates: List[Dict] = []
    for i, query in enumerate(queries, 1):
        if len(all_candidates) >= args.max_total:
            break
        try:
            cands = query_openalex(query, args.per_query, language, session)
        except requests.RequestException as e:
            print(f"[{i}/{len(queries)}] LỖI '{query}': {e}")
            continue

        new = 0
        for c in cands:
            key = c["doi"] or c["pdf_url"]
            if key in seen:
                continue
            seen.add(key)
            all_candidates.append(c)
            new += 1
            if len(all_candidates) >= args.max_total:
                break
        print(
            f"[{i}/{len(queries)}] '{query}': {len(cands)} kết quả, +{new} mới "
            f"(tổng {len(all_candidates)})"
        )
        time.sleep(args.sleep)

    n = write_jsonl(CANDIDATES_FILE, all_candidates)
    print(f"\nĐã ghi {n} candidate -> {CANDIDATES_FILE}")


if __name__ == "__main__":
    main()
