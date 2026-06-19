# OCR Benchmark Suite

**English** · [Tiếng Việt](README.vi.md) — see also the metrics spec: [English](METRICS.en.md) · [Tiếng Việt](METRICS.vi.md)

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
│       ├── datasets/               # Dataset registry (one module per dataset)
│       │   ├── __init__.py         #   REGISTRY + load_dataset() + parse_opts()
│       │   ├── base.py             #   BaseDataset interface
│       │   ├── pdf.py  doclaynet.py  pdfa.py  nvidia.py  folder.py
│       │   └── _common.py          #   shared H5 / download helpers
│       ├── metrics.py              # IOU, CER, WER, Accuracy
│       └── bbox.py                 # Bounding box utilities
├── docs/
│   ├── README.en.md  README.vi.md    # full docs (English / Vietnamese)
│   └── METRICS.en.md  METRICS.vi.md  # detailed metrics specification
├── requirements.txt
└── README.md                         # language switcher
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

## Usage

Every benchmark follows the same flow:

1. **Activate the environment** — `source .venv/bin/activate`.
2. **Pick a model** with `--model <name>` and point `--model_path` at its local
   checkpoint directory.
3. **Pick a dataset** with `--dataset <name>` and pass its parameters via
   repeatable `--opt key=value` flags — see [Supported Datasets](#supported-datasets).
4. **Run the script** — results are written to `--results_dir` as JSON
   (see [Output](#output)). Add `--debug` to detection to save bbox overlays.

Options shared by all three benchmark scripts:

| Option | Description |
|--------|-------------|
| `--dataset` | Dataset name: `pdf`, `doclaynet`, `pdfa`, `nvidia`, `folder` |
| `--opt key=value` | Dataset-specific parameter (repeatable) |
| `--model` | Model name from its registry (default `surya` for detection/recognition) |
| `--model_path` | Local directory containing the model checkpoint |
| `--max_rows` | Max pages/samples to evaluate |
| `--batch_size` | Inference batch size |
| `--results_dir` | Output directory for JSON results |

### 1. Text detection

```bash
# On a PDF file
python ocr-bench/evaluate/text_detection.py \
  --dataset pdf --opt path=/path/to/document.pdf \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 100

# On a HuggingFace dataset
python ocr-bench/evaluate/text_detection.py \
  --dataset pdfa \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 100

# With debug visualizations (saves bbox images)
python ocr-bench/evaluate/text_detection.py \
  --dataset doclaynet \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 50 \
  --debug
```

### 2. Text recognition

```bash
# On a PDF file
python ocr-bench/evaluate/text_recognition.py \
  --dataset pdf --opt path=/path/to/document.pdf \
  --model surya \
  --model_path ./model_path/text_recognition \
  --max_rows 100

# On a local image folder
python ocr-bench/evaluate/text_recognition.py \
  --dataset folder \
  --opt data_dir=/path/to/dataset \
  --opt image_folder=images \
  --opt label_file=labels.txt \
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

### 3. End-to-end OCR (VLM / API models)

```bash
# VLM (local model)
python ocr-bench/evaluate/end_to_end.py \
  --model_type vlm \
  --model qwen_vl \
  --model_path ./model_path/qwen_vl \
  --dataset doclaynet \
  --max_rows 100

# Cloud API model
python ocr-bench/evaluate/end_to_end.py \
  --model_type api \
  --model claude \
  --dataset pdfa \
  --max_rows 100
```

End-to-end evaluation uses **page-level CER/WER** — all text lines on a page are joined and compared as one string, avoiding line-count mismatch issues with generative models.

---

## Supported Datasets

Each dataset has a short `--dataset` name and its own `--opt key=value` options.

| `--dataset` | Tasks | `--opt` options | Notes |
|-------------|-------|-----------------|-------|
| `pdf` | Detection, Recognition | `path` (required) | GT from embedded text layer |
| `doclaynet` | Detection, Recognition | `name` (default `vikp/doclaynet_bench`) | Document layout with bboxes |
| `pdfa` | Detection, Recognition | `name` (default `pixparse/pdfa-eng-wds`) | English PDFs, word/line OCR |
| `nvidia` | Detection, Recognition | `h5_files`, `language`, `max_size_limit`, `chunk_size` | Multilingual synthetic |
| `folder` | Recognition | `data_dir` (required), `image_folder`, `label_file` | Images + `labels.txt` |

### NVIDIA multilingual dataset options

```bash
python ocr-bench/evaluate/text_detection.py \
  --dataset nvidia \
  --opt language=en \
  --opt h5_files=train_000,train_001 \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 1000 \
  --batch_size 32
```

Supported languages: `en`, `ja`, `ko`, `ru`, `zh_hans`, `zh_hant`

### Adding a new dataset

**1. Create `ocr-bench/utils/datasets/<name>.py`** with a `BaseDataset` subclass.
Implement `detection()` and/or `recognition()` (and optionally `pathname()`), and
read run-specific parameters from the `opts` dict (populated from `--opt key=value`):

```python
# ocr-bench/utils/datasets/mydata.py
from utils.datasets.base import BaseDataset


class MyDataset(BaseDataset):
    name = "mydata"  # the value passed to --dataset

    def pathname(self, opts):
        # used to name the results file; defaults to self.name if omitted
        return opts.get("split", self.name)

    def detection(self, max_rows, opts):
        src = opts["src"]  # --opt src=...   (raise/KeyError if required & missing)
        images, bboxes = load_my_data(src, max_rows)
        # detection yields chunks (lets you stream large datasets):
        #   images : list[PIL.Image]
        #   bboxes : list per image of [x1, y1, x2, y2]
        yield images, bboxes

    def recognition(self, max_rows, opts):
        src = opts["src"]
        images, texts, bboxes = load_my_data_with_text(src, max_rows)
        # recognition returns a single tuple (loaded into memory):
        #   texts  : flat list of GT strings (one per bbox, across all images)
        #   bboxes : list per image of [x1, y1, x2, y2]
        return images, texts, bboxes
```

Implement only the task(s) you need — the base class raises a clear
`NotImplementedError` for the other. `bbox` coordinates are pixel values in the
returned image's space.

**2. Register it** in `ocr-bench/utils/datasets/__init__.py`:

```python
from utils.datasets.mydata import MyDataset

REGISTRY = {
    # ... existing entries ...
    MyDataset.name: MyDataset,
}
```

**3. Run it** — no changes to the evaluate scripts are needed:

```bash
python ocr-bench/evaluate/text_recognition.py \
  --dataset mydata --opt src=/path/to/data \
  --model surya --model_path ./model_path/text_recognition
```

---

## Metrics

> Full formulas and worked explanations: [METRICS.en.md](METRICS.en.md).

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

**Out of memory** — reduce `--batch_size` (or, for the `nvidia` dataset, cap the
image size with `--opt max_size_limit=1024`):
```bash
python ocr-bench/evaluate/text_detection.py ... --batch_size 2 \
  --dataset nvidia --opt max_size_limit=1024
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

# Run benchmarks (point --model_path at your local checkpoint directory)
!python ocr-bench/evaluate/text_detection.py \
  --dataset nvidia --opt language=en --opt h5_files=train_000 \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 1000 --batch_size 32

!python ocr-bench/evaluate/text_recognition.py \
  --dataset nvidia --opt language=en --opt h5_files=train_000 \
  --model surya \
  --model_path ./model_path/text_recognition \
  --max_rows 1000 --batch_size 128
```
