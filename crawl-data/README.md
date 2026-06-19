# crawl-data — Thu thập PDF digital tiếng Việt + line bbox

Pipeline crawl PDF học thuật **born-digital** (bốc được text trực tiếp, không
phải ảnh scan) tiếng Việt từ **nguồn học thuật mở** (OpenAlex), rồi trích
**line bbox + text** làm ground-truth cho benchmark OCR.

> Thay cho Google Scholar (không có API, chặn scraper gắt). OpenAlex miễn phí,
> không cần key, có filter open-access + ngôn ngữ.

## Phụ thuộc

Chỉ cần `requests` và `pymupdf` (thêm `Pillow` cho bước QA). `langdetect` là
tùy chọn — pipeline tự dùng heuristic dựa trên mật độ dấu thanh nếu thiếu.

```bash
pip install requests pymupdf Pillow
```

## Quy trình (chạy từ trong thư mục `crawl-data/`)

| Bước | Script | Vào | Ra |
|---|---|---|---|
| 1. Discover | `discover.py` | `keywords.txt` | `candidates.jsonl` |
| 2. Download | `download.py` | `candidates.jsonl` | `pdfs/`, `download_log.jsonl` |
| 3. Filter | `filter_pdf.py` | `pdfs/` | `filtered.jsonl` |
| 4. Extract | `extract.py` | `filtered.jsonl` | `json/<id>.json` |
| 5. QA | `qa.py` | `json/` | `preview/`, `manifest.jsonl` |

```bash
python discover.py   --per-query 25 --max-total 400
python download.py   --max 200 --sleep 1.0
python filter_pdf.py --min-chars 200 --min-vi 0.5
python extract.py    --dpi 200
python qa.py         --preview-pages 1 --max-preview 6
```

## Định dạng output `json/<id>.json`

```json
{
  "id": "…", "pdf_path": "pdfs/….pdf", "source": "openalex",
  "doi": "…", "license": null, "title": "…", "language": "vi",
  "render_dpi": 200,
  "pages": [
    {
      "page_index": 0, "width": 1575, "height": 2205,
      "lines": [{"bbox": [x1, y1, x2, y2], "text": "…"}]
    }
  ]
}
```

`bbox` ở hệ tọa độ **pixel** tại `render_dpi` (gốc PDF point × dpi/72), nên
render lại trang bằng `page.get_pixmap(dpi=render_dpi)` là khớp ngay. Text đã
chuẩn hóa **Unicode NFC** để dấu tiếng Việt không bị tách.

## Lưu ý chất lượng & bản quyền

- **Bộ lọc tiếng Việt**: chấm điểm theo mật độ dấu thanh (NFD) + chữ riêng
  (ă â đ ê ô ơ ư) + từ chức năng. Bền với tài liệu song ngữ — loại được bài
  *nội dung tiếng Anh* dù tiêu đề/metadata tiếng Việt.
- **Bộ lọc digital**: loại PDF scan dựa trên số ký tự/trang và tỷ lệ trang
  chỉ-có-ảnh.
- **Bản quyền**: chỉ tải open-access; `license` được ghi vào `manifest.jsonl`
  (thường là `null` với tạp chí VN — cân nhắc trước khi tái phân phối).
- Một số host (vd `js.ktpt.edu.vn`) có thể chặn kết nối; pipeline bỏ qua và
  ghi log, không dừng.

## Mở rộng

- Thêm nguồn: Semantic Scholar (`openAccessPdf`), CORE (cần key) — viết hàm
  query tương tự trong `discover.py`.
- Chuyển sang H5 cho benchmark: `json/*.json` đã đủ thông tin để map
  `bbox [x1,y1,x2,y2] → [x,y,w,h]` khớp `load_h5_*` trong `ocr-bench/utils`.
