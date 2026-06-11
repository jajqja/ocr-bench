# OCR Benchmark Suite

A benchmarking suite for evaluating OCR models on text detection, text recognition, and end-to-end OCR tasks. Supports local detection/recognition models (e.g. Surya), VLMs, and cloud API models through a unified model registry.

## Features

- **Text Detection** — Precision, Recall, F1, Page IoU
- **Text Recognition** — CER, WER, Exact Match Accuracy
- **End-to-End OCR** — Page-level evaluation for VLMs and API models
- **Multi-model support** — plug in any model by implementing a base class and registering it
- **Multiple data sources** — PDF files, HuggingFace datasets, NVIDIA H5, local image folders
- **Excel reports** — formatted output from JSON results

---

## Project Structure

```
ocr-bench/
├── ocr-bench/
│   ├── evaluate/
│   │   ├── text_detection.py       # Detection benchmark (pipeline models)
│   │   ├── text_recognition.py     # Recognition benchmark (pipeline models)
│   │   └── end_to_end.py           # End-to-end benchmark (VLMs, API models)
│   ├── models/
│   │   ├── base.py                 # Abstract base classes
│   │   ├── detection/
│   │   │   ├── __init__.py         # Registry + load()
│   │   │   └── surya.py            # Surya detection wrapper
│   │   ├── recognition/
│   │   │   ├── __init__.py         # Registry + load()
│   │   │   └── surya.py            # Surya recognition wrapper
│   │   ├── vlm/
│   │   │   ├── __init__.py         # Registry + load()
│   │   │   └── qwen_vl.py          # Qwen-VL stub
│   │   └── api/
│   │       ├── __init__.py         # Registry + load()
│   │       └── claude.py           # Claude API stub
│   ├── make_report/
│   │   ├── text_detection.py       # Export detection results to Excel
│   │   └── text_recognition.py     # Export recognition results to Excel
│   └── utils/
│       ├── datasets.py             # Data loading for all benchmarks
│       ├── metrics.py              # IOU, CER, WER, Accuracy
│       ├── bbox.py                 # Bounding box utilities
│       └── model_download.py       # HuggingFace model downloader
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/jajqja/ocr-bench.git
cd ocr-bench

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick Start

### 1. Download a model from HuggingFace

```bash
python ocr-bench/utils/model_download.py \
  --repo_id username/detection-model \
  --local_dir model_path/text_detection \
  --hf_token hf_***  # optional, for private models

python ocr-bench/utils/model_download.py \
  --repo_id username/recognition-model \
  --local_dir model_path/text_recognition \
  --hf_token hf_***
```

### 2. Run Detection Benchmark

```bash
# On a PDF file
python ocr-bench/evaluate/text_detection.py \
  --pdf_path /path/to/document.pdf \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 100

# On a HuggingFace dataset
python ocr-bench/evaluate/text_detection.py \
  --dataset_name pixparse/pdfa-eng-wds \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 100

# With debug visualizations (saves bbox images)
python ocr-bench/evaluate/text_detection.py \
  --dataset_name vikp/doclaynet_bench \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 50 \
  --debug
```

### 3. Run Recognition Benchmark

```bash
# On a PDF file
python ocr-bench/evaluate/text_recognition.py \
  --pdf_path /path/to/document.pdf \
  --model surya \
  --model_path ./model_path/text_recognition \
  --max_rows 100

# On a local image folder
python ocr-bench/evaluate/text_recognition.py \
  --data_dir /path/to/dataset \
  --image_folder images \
  --label_file labels.txt \
  --model surya \
  --model_path ./model_path/text_recognition \
  --max_rows 1000
```

**Local folder format:**
```
dataset/
├── images/
│   ├── img_001.jpg
│   └── img_002.png
└── labels.txt      # one line per image: "img_001.jpg<TAB>ground truth text"
```

### 4. Run End-to-End Benchmark (VLMs / API models)

```bash
# VLM (local model)
python ocr-bench/evaluate/end_to_end.py \
  --model_type vlm \
  --model qwen_vl \
  --model_path ./model_path/qwen_vl \
  --dataset_name vikp/doclaynet_bench \
  --max_rows 100

# Cloud API model
python ocr-bench/evaluate/end_to_end.py \
  --model_type api \
  --model claude \
  --dataset_name pixparse/pdfa-eng-wds \
  --max_rows 100
```

End-to-end evaluation uses **page-level CER/WER** — all text lines on a page are joined and compared as one string, avoiding line-count mismatch issues with generative models.

---

## Supported Datasets

| Dataset | Tasks | Notes |
|---------|-------|-------|
| `vikp/doclaynet_bench` | Detection, Recognition | Document layout with bounding boxes |
| `pixparse/pdfa-eng-wds` | Detection, Recognition | English PDFs with word/line-level OCR |
| `nvidia/OCR-Synthetic-Multilingual-v1` | Detection, Recognition | Large-scale multilingual synthetic data |
| Local PDF | Detection, Recognition | GT extracted from embedded text layer |
| Local folder | Recognition | Images + `labels.txt` |

### NVIDIA multilingual dataset options

```bash
python ocr-bench/evaluate/text_detection.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model surya \
  --model_path ./model_path/text_detection \
  --language en \
  --h5_files train_000,train_001 \
  --max_rows 1000 \
  --batch_size 32
```

Supported languages: `en`, `ja`, `ko`, `ru`, `zh_hans`, `zh_hant`

---

## Metrics

### Detection

| Metric | Description |
|--------|-------------|
| **Precision** | % of predicted boxes that cover a GT box (threshold 0.5) |
| **Recall** | % of GT boxes covered by predictions (threshold 0.5) |
| **F1** | Harmonic mean of Precision and Recall |
| **Page IoU** | Polygon-union IoU across the full page |

### Recognition / End-to-End

| Metric | Description | Better |
|--------|-------------|--------|
| **CER** | Character Error Rate (Levenshtein / reference length) | Lower |
| **WER** | Word Error Rate (word-level Levenshtein) | Lower |
| **Accuracy** | Exact string match rate | Higher |

---

## Adding a New Model

### Detection or Recognition model

1. Create the implementation file:

```python
# ocr-bench/models/detection/paddleocr.py
from models.base import BaseDetectionModel

class PaddleOCRDetectionModel(BaseDetectionModel):
    def __init__(self, checkpoint: str):
        # load your model here
        ...

    def predict(self, images, batch_size=8):
        # return List[List[[x1, y1, x2, y2]]] — one bbox list per image
        ...
```

2. Register it:

```python
# ocr-bench/models/detection/__init__.py
from models.detection.paddleocr import PaddleOCRDetectionModel

REGISTRY = {
    "surya": SuryaDetectionModel,
    "paddleocr": PaddleOCRDetectionModel,   # add this
}
```

3. Run with `--model paddleocr`.

### VLM or API model

Same pattern — implement `BaseEndToEndModel.predict()` which returns `List[List[{"text": str, "bbox": ...}]]`, then register in `models/vlm/__init__.py` or `models/api/__init__.py`.

---

## Output

Results are saved as JSON under `--results_dir`:

```
results/
├── detection_benchmark/
│   └── <dataset_name>/
│       └── results.json
├── recognition_benchmark/
│   └── <dataset_name>_results.json
└── end_to_end_benchmark/
    └── <dataset>_<model_type>_<model>_results.json
```

### Generate Excel reports

```bash
python ocr-bench/make_report/text_detection.py
python ocr-bench/make_report/text_recognition.py
```

---

## Troubleshooting

**Out of memory** — reduce `--batch_size` or `--max_size_limit`:
```bash
python ocr-bench/evaluate/text_detection.py ... --batch_size 2 --max_size_limit 1024
```

**PDF extraction issues** — verify the PDF is text-based (not scanned):
```bash
python -c "import pymupdf; doc = pymupdf.open('file.pdf'); print(f'Pages: {len(doc)}')"
```

---

## Colab Example

```python
!git clone https://github.com/jajqja/ocr-bench.git
%cd ocr-bench
!pip install -r requirements.txt

# Download models
!python ocr-bench/utils/model_download.py \
  --repo_id newai-vn/newai-text-det \
  --local_dir model_path/text_detection \
  --hf_token hf_***

!python ocr-bench/utils/model_download.py \
  --repo_id newai-vn/newai-text-rec \
  --local_dir model_path/text_recognition \
  --hf_token hf_***

# Run benchmarks
!python ocr-bench/evaluate/text_detection.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model surya \
  --model_path ./model_path/text_detection \
  --language en --h5_files train_000 --max_rows 1000 --batch_size 32

!python ocr-bench/evaluate/text_recognition.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model surya \
  --model_path ./model_path/text_recognition \
  --language en --h5_files train_000 --max_rows 1000 --batch_size 128
```
