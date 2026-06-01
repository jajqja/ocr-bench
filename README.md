# Surya OCR Benchmark Suite

A comprehensive benchmarking suite for evaluating OCR models (Text Detection & Text Recognition) on PDF datasets.

## Features

✅ **Text Detection Evaluation**
- Precision & Recall metrics
- IOU (Intersection over Union) score
- Works with PDF files or HuggingFace datasets

✅ **Text Recognition Evaluation**
- Word Error Rate (WER)
- Character Error Rate (CER)  
- Exact Match Accuracy
- Works with PDF files or HuggingFace datasets

✅ **Model Management**
- Load models from local storage
- Download models from HuggingFace Hub
- Support for private models with HF token

✅ **Comprehensive Reporting**
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
python -m ocr-bench/model_download \
  --repo_id username/detection-model \
  --local_dir model_path/textdetection \
  --hf_token hf_*************

# Download recognition model
python -m ocr-bench/model_download \
  --repo_id username/recognition-model \
  --local_dir model_path/textrecognition \
  --hf_token hf_*************
```

### 2. Run Benchmark on Digital PDF file

```bash
# Full benchmark (detection + recognition)
python ocr-bench/benchmark.py benchmark-pdf \
  --pdf_path /path/to/document.pdf \
  --detection_model /path/to/detection/model \
  --recognition_model /path/to/recognition/model \
  --max_pages 100 \
  --results_dir ./results

```

### 3. Run Individual Benchmarks
#### 3.1 Digital pdf file
```bash
python -m ocr-bench/text_detection \
  --pdf_path /path/to/document.pdf \
  --model_path /path/to/detection/model \
  --max_rows 100 \
  --results_dir ./detection_results \
  --debug  # (optional) save visualization images
```

```bash
python -m ocr-bench/text_recognition \
  --pdf_path /path/to/document.pdf \
  --model_path /path/to/recognition/model \
  --max_rows 100 \
  --results_dir ./recognition_results
```

#### 3.2 Data line folder (for recognition only)
```bash
python -m ocr-bench/text_recognition \
  --data_dir /datadir \
  --image_folder images \
  --label_file labels.txt \
  --max_rows 1000 \
  --model_path /path/to/recognition/model \
  --results_dir ./recognition_results
```

#### 3.3 Using with HuggingFace Datasets

```bash
# Benchmark on HuggingFace dataset
python -m ocr-bench/text_detection \
  --dataset_name mnist-ocr-digits \
  --model_path ./detection_model \
  --max_rows 1000

python -m ocr-bench/text_recognition \
  --dataset_name ocr-text-recognition \
  --model_path ./recognition_model \
  --max_rows 1000
```

### 4. Compare Results

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
python -m text_detection \
  --pdf_path document.pdf \
  --model_path ./model \
  --debug  # Saves detection bbox images
```

---

## Project Structure

```
ocr-bench/
├── ocr-bench/          # Model download utilities & CLI
│   ├── model_download.py          # Model download utilities & CLI
│   ├── text_detection.py          # Detection benchmark script
│   ├── text_recognition.py        # Recognition benchmark script
│   ├── benchmark.py               # Main unified benchmark CLI
│   ├── utils/
│   │   ├── metrics.py             # Metric calculations (IOU, CER, WER, etc.)
│   │   ├── bbox.py                # Bounding box utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
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
# Main benchmark help
python ocr-bench/benchmark.py --help

# Individual command help
python ocr-bench/benchmark.py benchmark-pdf --help
python ocr-bench/benchmark.py benchmark-dataset --help
python ocr-bench/benchmark.py verify-models --help
python ocr-bench/benchmark.py compare-results --help

# Model loading help
python -m ocr-bench/model_download --help

# Detection benchmark help
python -m ocr-bench/text_detection --help

# Recognition benchmark help
python -m ocr-bench/text_recognition --help
```

---

## Troubleshooting

### Model not found error
```bash
# Verify models can be loaded
python ocr-bench/benchmark.py verify-models \
  --detection_model /path/to/model \
  --recognition_model /path/to/model
```

### Out of memory errors
Reduce batch size or max_rows:
```bash
python -m ocr-bench/text_recognition \
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

!pip install -r requirements.txt

# Run benchmark
!python -m ocr-bench/text_detection --pdf_path sample.pdf --model_path ./model
```
