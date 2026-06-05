# Fintuned Surya OCR Benchmark Suite

A comprehensive benchmarking suite for evaluating OCR models (Text Detection & Text Recognition) on PDF and Hugging Face datasets.

## Features

**Text Detection Evaluation**
- Precision & Recall metrics
- IOU (Intersection over Union) score
- Works with PDF files or HuggingFace datasets

**Text Recognition Evaluation**
- Word Error Rate (WER)
- Character Error Rate (CER)  
- Exact Match Accuracy
- Works with PDF files or HuggingFace datasets

**Model Management**
- Load models from local storage
- Download models from HuggingFace Hub
- Support for private models with HF token

**Comprehensive Reporting**
- JSON results with detailed metrics
- Sample predictions visualization
- Benchmark comparison reports

---

## Installation

```bash
# Clone repository
git clone https://github.com/jajqja/ocr-bench.git
cd ocr-bench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Load Models

#### Option: Download from HuggingFace

You can edit scripts from `model_download.py`

```bash
# Download detection model
python ocr-bench/model_download.py \
  --repo_id username/detection-model \
  --local_dir model_path/textdetection \
  --hf_token hf_*************

# Download recognition model
python ocr-bench/model_download.py \
  --repo_id username/recognition-model \
  --local_dir model_path/textrecognition \
  --hf_token hf_*************
```

### 2. Run Individual Benchmarks
#### 2.1 Digital pdf file
```bash
python ocr-bench/text_detection.py \
  --pdf_path /path/to/document.pdf \
  --model_path /path/to/detection/model \
  --max_rows 100 \
  --results_dir ./detection_results \
  --debug  # (optional) save visualization images
```

```bash
python ocr-bench/text_recognition.py \
  --pdf_path /path/to/document.pdf \
  --model_path /path/to/recognition/model \
  --max_rows 100 \
  --results_dir ./recognition_results
```

#### 2.2 Data line folder (for recognition only)
```bash
python ocr-bench/text_recognition.py \
  --data_dir /datadir \
  --image_folder images \
  --label_file labels.txt \
  --max_rows 1000 \
  --model_path /path/to/recognition/model \
  --results_dir ./recognition_results
```

#### 2.3 Using with HuggingFace Datasets

```bash
# Benchmark on HuggingFace dataset (PDFA)
python ocr-bench/text_detection.py \
  --dataset_name pixparse/pdfa-eng-wds \
  --model_path ./detection_model \
  --max_rows 100

python ocr-bench/text_recognition.py \
  --dataset_name pixparse/pdfa-eng-wds \
  --model_path ./recognition_model \
  --max_rows 100

```

```bash
# Benchmark on HuggingFace dataset (nvidia/OCR-Synthetic-Multilingual-v1)
python ocr-bench/text_detection.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model_path ./model_path/text_detection \
  --language en \
  --h5_files train_000 \
  --max_rows 1000 \
  --batch_size 32

python ocr-bench/text_recognition.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model_path ./model_path/text_recognition \
  --language en \
  --h5_files train_000 \
  --max_rows 10 \
  --batch_size 128

```

#### 2.4 Supported Datasets

| Dataset Name | Type | Features |
|-------------|------|----------|
| `vikp/doclaynet_bench` | HuggingFace | Document layout with bounding boxes |
| `pixparse/pdfa-eng-wds` | HuggingFace | English PDFs with word/line-level OCR annotations |
| `nvidia/OCR-Synthetic-Multilingual-v1` | HuggingFace | Large-scale synthetically generated OCR training dataset for multilingual text detection and recognition |

All datasets support detection and recognition benchmarks.

### 3. Compare Results

```bash
python ocr-bench/benchmark.py compare-results --results_dir ./results
```

---

## Metrics Explanation

### Detection Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Precision** | % of predicted boxes that correctly cover ground truth | 0-1 |
| **Recall** | % of ground truth boxes that are covered by predictions | 0-1 |
| **IOU** | Intersection over Union with penalty for overlapping boxes | 0-1 |

### Recognition Metrics

| Metric | Description | Range | Notes |
|--------|-------------|-------|-------|
| **CER** | Character Error Rate (Levenshtein distance) | 0+ | Lower is better |
| **WER** | Word Error Rate (word-level Levenshtein) | 0+ | Lower is better |
| **Accuracy** | Exact match accuracy (% perfectly correct) | 0-1 | Higher is better |

---

## Output Structure

```
results/
├── detection/
│   └── document_name/
│       ├── results.json          # Detailed metrics per page
│       └── 0_bbox.png, 1_bbox.png, ...  # (optional) debug images
├── recognition/
│   └── document_name_results.json # Recognition metrics
└── benchmark_summary.json         # Overall summary
```

### Sample Results File (Detection)
```json
{
  "dataset": "document",
  "model": "path/to/model",
  "num_samples": 50,
  "metrics": {
    "surya": {
      "precision": 0.95,
      "recall": 0.92,
      "iou": 0.88
    }
  },
  "page_metrics": { ... }
}
```

### Sample Results File (Recognition)
```json
{
  "model": "path/to/model",
  "dataset": "document",
  "num_samples": 50,
  "metrics": {
    "cer": 0.05,
    "wer": 0.08,
    "accuracy": 0.92
  },
  "predictions": [ ... ]
}
```

---

## Advanced Usage

### Debug Mode with Visualizations

```bash
python text_detection.py \
  --pdf_path document.pdf \
  --model_path ./model \
  --debug  # Saves detection bbox images
```

---

## Project Structure

```
ocr-bench/
├── ocr-bench/                     # Model download utilities & CLI
│   ├── model_download.py          # Model download utilities & CLI
│   ├── text_detection.py          # Detection benchmark script
│   ├── text_recognition.py        # Recognition benchmark script
│   ├── utils/
│   │   ├── metrics.py             # Metric calculations (IOU, CER, WER, etc.)
│   │   ├── bbox.py                # Bounding box utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Supported Models

### Detection Models
- Surya Detection (default)
- Any model compatible with Surya's DetectionPredictor

### Recognition Models
- Surya Recognition (default)
- Any model compatible with Surya's RecognitionPredictor

---

## Command-Line Help

```bash
# Model loading help
python ocr-bench/model_download.py --help

# Detection benchmark help
python ocr-bench/text_detection.py --help

# Recognition benchmark help
python ocr-bench/text_recognition.py --help
```

---

## Troubleshooting

### Out of memory errors
Reduce batch size or max_rows:
```bash
python ocr-bench/text_recognition.py \
  --pdf_path document.pdf \
  --model_path ./model \
  --max_rows 10 \
  --batch_size 2

python ocr-bench/text_detection.py \
  --pdf_path document.pdf \
  --model_path ./model \
  --max_rows 10 \
  --batch_size 2
```

### PDF extraction issues
Ensure PDF is readable:
```bash
# Test PDF loading
python -c "import fitz; doc = fitz.open('document.pdf'); print(f'Pages: {len(doc)}')"
```

---

## Colab Usage

```python
# In Google Colab
!git clone https://github.com/jajqja/ocr-bench.git
%cd ocr-bench

# Install requirements
!pip install -r requirements.txt

# Download detection model
!python ocr-bench/model_download.py \
  --repo_id newai-vn/newai-text-det \
  --local_dir model_path/text_detection \
  --hf_token hf_****************

# Download recognition model
!python ocr-bench/model_download.py \
  --repo_id newai-vn/newai-text-rec \
  --local_dir model_path/text_recognition \
  --hf_token hf_***************

# Run text detection
!python ocr-bench/text_detection.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model_path ./model_path/text_detection \
  --language en \
  --h5_files train_000 \
  --max_rows 1000 \
  --batch_size 32

# Run text recognition
!python ocr-bench/text_recognition.py \
  --dataset_name nvidia/OCR-Synthetic-Multilingual-v1 \
  --model_path ./model_path/text_recognition \
  --language en \
  --h5_files train_000 \
  --max_rows 1000 \
  --batch_size 128
```
