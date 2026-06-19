# OCR Benchmark Suite

A benchmarking suite for evaluating OCR models on **text detection**, **text recognition**, and **end-to-end OCR** — with local detection/recognition models (e.g. Surya), VLMs, and cloud API models behind a unified registry.

> Bộ công cụ benchmark đánh giá mô hình OCR cho phát hiện dòng chữ, nhận dạng chữ và OCR đầu-cuối.

## 📖 Documentation / Tài liệu

| | Guide | Metrics |
|---|-------|---------|
| 🇬🇧 **English** | [docs/README.en.md](docs/README.en.md) | [docs/METRICS.en.md](docs/METRICS.en.md) |
| 🇻🇳 **Tiếng Việt** | [docs/README.vi.md](docs/README.vi.md) | [docs/METRICS.vi.md](docs/METRICS.vi.md) |

## Quick install

```bash
git clone https://github.com/jajqja/ocr-bench.git
cd ocr-bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the usage guide in your language above.
