# Bộ công cụ Benchmark OCR

[English](README.en.md) · **Tiếng Việt** — xem thêm tài liệu chỉ số: [English](METRICS.en.md) · [Tiếng Việt](METRICS.vi.md)

Bộ công cụ benchmark để đánh giá các mô hình OCR trên ba tác vụ: phát hiện dòng chữ (text detection), nhận dạng chữ (text recognition) và OCR đầu-cuối (end-to-end). Hỗ trợ mô hình detection/recognition cục bộ (vd Surya), VLM, và mô hình API đám mây thông qua một registry mô hình thống nhất.

## Tính năng

- **Text Detection** — Precision, Recall, F1, Page IoU
- **Text Recognition** — CER, WER, Accuracy (khớp tuyệt đối)
- **OCR đầu-cuối** — đánh giá ở mức trang cho VLM và mô hình API
- **Hỗ trợ đa mô hình** — cắm thêm mô hình bất kỳ bằng cách kế thừa base class và đăng ký
- **Nhiều nguồn dữ liệu** — file PDF, dataset HuggingFace, NVIDIA H5, thư mục ảnh cục bộ
- **Báo cáo Excel** — xuất từ kết quả JSON

---

## Cấu trúc dự án

```
ocr-bench/
├── ocr-bench/
│   ├── evaluate/
│   │   ├── text_detection.py       # Benchmark detection (mô hình pipeline)
│   │   ├── text_recognition.py     # Benchmark recognition (mô hình pipeline)
│   │   └── end_to_end.py           # Benchmark đầu-cuối (VLM, mô hình API)
│   ├── models/
│   │   ├── base.py                 # Các lớp cơ sở trừu tượng
│   │   ├── detection/
│   │   │   ├── __init__.py         # Registry + load()
│   │   │   └── surya.py            # Wrapper Surya detection
│   │   ├── recognition/
│   │   │   ├── __init__.py         # Registry + load()
│   │   │   └── surya.py            # Wrapper Surya recognition
│   │   ├── vlm/
│   │   │   ├── __init__.py         # Registry + load()
│   │   │   └── qwen_vl.py          # Stub Qwen-VL
│   │   └── api/
│   │       ├── __init__.py         # Registry + load()
│   │       └── claude.py           # Stub Claude API
│   ├── make_report/
│   │   ├── text_detection.py       # Xuất kết quả detection ra Excel
│   │   └── text_recognition.py     # Xuất kết quả recognition ra Excel
│   └── utils/
│       ├── datasets/               # Registry dataset (mỗi dataset một module)
│       │   ├── __init__.py         #   REGISTRY + load_dataset() + parse_opts()
│       │   ├── base.py             #   Giao diện BaseDataset
│       │   ├── pdf.py  doclaynet.py  pdfa.py  nvidia.py  folder.py
│       │   └── _common.py          #   helper tải/đọc H5 dùng chung
│       ├── metrics.py              # IOU, CER, WER, Accuracy
│       └── bbox.py                 # Tiện ích bounding box
├── docs/
│   ├── README.en.md  README.vi.md    # tài liệu đầy đủ (Anh / Việt)
│   └── METRICS.en.md  METRICS.vi.md  # đặc tả chỉ số chi tiết
├── requirements.txt
└── README.md                         # trang chuyển ngôn ngữ
```

---

## Cài đặt

```bash
git clone https://github.com/jajqja/ocr-bench.git
cd ocr-bench

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Cách sử dụng

Mọi benchmark đều theo cùng một quy trình:

1. **Kích hoạt môi trường** — `source .venv/bin/activate`.
2. **Chọn mô hình** bằng `--model <tên>` và trỏ `--model_path` tới thư mục checkpoint cục bộ.
3. **Chọn dataset** bằng `--dataset <tên>` và truyền tham số riêng qua cờ
   `--opt key=value` (lặp lại được) — xem [Dataset được hỗ trợ](#dataset-được-hỗ-trợ).
4. **Chạy script** — kết quả ghi ra `--results_dir` dạng JSON
   (xem [Kết quả](#kết-quả)). Thêm `--debug` cho detection để lưu ảnh vẽ bbox.

Các tùy chọn dùng chung cho cả ba script benchmark:

| Tùy chọn | Mô tả |
|----------|-------|
| `--dataset` | Tên dataset: `pdf`, `doclaynet`, `pdfa`, `nvidia`, `folder` |
| `--opt key=value` | Tham số riêng của dataset (lặp lại được) |
| `--model` | Tên mô hình trong registry (mặc định `surya` cho detection/recognition) |
| `--model_path` | Thư mục cục bộ chứa checkpoint mô hình |
| `--max_rows` | Số trang/mẫu tối đa cần đánh giá |
| `--batch_size` | Kích thước batch khi suy luận |
| `--results_dir` | Thư mục xuất kết quả JSON |

### 1. Phát hiện dòng chữ (detection)

```bash
# Trên một file PDF
python ocr-bench/evaluate/text_detection.py \
  --dataset pdf --opt path=/path/to/document.pdf \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 100

# Trên một dataset HuggingFace
python ocr-bench/evaluate/text_detection.py \
  --dataset pdfa \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 100

# Kèm ảnh debug (lưu ảnh vẽ bbox)
python ocr-bench/evaluate/text_detection.py \
  --dataset doclaynet \
  --model surya \
  --model_path ./model_path/text_detection \
  --max_rows 50 \
  --debug
```

### 2. Nhận dạng chữ (recognition)

```bash
# Trên một file PDF
python ocr-bench/evaluate/text_recognition.py \
  --dataset pdf --opt path=/path/to/document.pdf \
  --model surya \
  --model_path ./model_path/text_recognition \
  --max_rows 100

# Trên một thư mục ảnh cục bộ
python ocr-bench/evaluate/text_recognition.py \
  --dataset folder \
  --opt data_dir=/path/to/dataset \
  --opt image_folder=images \
  --opt label_file=labels.txt \
  --model surya \
  --model_path ./model_path/text_recognition \
  --max_rows 1000
```

**Định dạng thư mục cục bộ:**
```
dataset/
├── images/
│   ├── img_001.jpg
│   └── img_002.png
└── labels.txt      # mỗi dòng một ảnh: "img_001.jpg<TAB>nhãn ground truth"
```

### 3. OCR đầu-cuối (mô hình VLM / API)

```bash
# VLM (mô hình cục bộ)
python ocr-bench/evaluate/end_to_end.py \
  --model_type vlm \
  --model qwen_vl \
  --model_path ./model_path/qwen_vl \
  --dataset doclaynet \
  --max_rows 100

# Mô hình API đám mây
python ocr-bench/evaluate/end_to_end.py \
  --model_type api \
  --model claude \
  --dataset pdfa \
  --max_rows 100
```

Đánh giá đầu-cuối dùng **CER/WER ở mức trang** — toàn bộ dòng chữ trên một trang được ghép lại và so sánh như một chuỗi, tránh lỗi lệch số dòng thường gặp ở các mô hình sinh.

---

## Dataset được hỗ trợ

Mỗi dataset có một `--dataset` ngắn gọn và bộ `--opt key=value` riêng.

| `--dataset` | Tác vụ | Tùy chọn `--opt` | Ghi chú |
|-------------|--------|------------------|---------|
| `pdf` | Detection, Recognition | `path` (bắt buộc) | GT lấy từ lớp text nhúng trong PDF |
| `doclaynet` | Detection, Recognition | `name` (mặc định `vikp/doclaynet_bench`) | Bố cục tài liệu kèm bbox |
| `pdfa` | Detection, Recognition | `name` (mặc định `pixparse/pdfa-eng-wds`) | PDF tiếng Anh, OCR mức từ/dòng |
| `nvidia` | Detection, Recognition | `h5_files`, `language`, `max_size_limit`, `chunk_size` | Dữ liệu tổng hợp đa ngôn ngữ |
| `folder` | Recognition | `data_dir` (bắt buộc), `image_folder`, `label_file` | Ảnh + `labels.txt` |

### Tùy chọn dataset NVIDIA đa ngôn ngữ

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

Ngôn ngữ hỗ trợ: `en`, `ja`, `ko`, `ru`, `zh_hans`, `zh_hant`

### Thêm một dataset mới

**1. Tạo `ocr-bench/utils/datasets/<tên>.py`** với một lớp con của `BaseDataset`.
Cài đặt `detection()` và/hoặc `recognition()` (tùy chọn `pathname()`), và đọc tham
số runtime từ dict `opts` (lấy từ `--opt key=value`):

```python
# ocr-bench/utils/datasets/mydata.py
from utils.datasets.base import BaseDataset


class MyDataset(BaseDataset):
    name = "mydata"  # giá trị truyền cho --dataset

    def pathname(self, opts):
        # dùng để đặt tên file kết quả; mặc định là self.name nếu bỏ qua
        return opts.get("split", self.name)

    def detection(self, max_rows, opts):
        src = opts["src"]  # --opt src=...   (KeyError nếu bắt buộc mà thiếu)
        images, bboxes = load_my_data(src, max_rows)
        # detection yield theo từng chunk (cho phép stream dataset lớn):
        #   images : list[PIL.Image]
        #   bboxes : list theo từng ảnh, mỗi phần tử [x1, y1, x2, y2]
        yield images, bboxes

    def recognition(self, max_rows, opts):
        src = opts["src"]
        images, texts, bboxes = load_my_data_with_text(src, max_rows)
        # recognition trả về một tuple (nạp toàn bộ vào bộ nhớ):
        #   texts  : list phẳng các chuỗi GT (một chuỗi/bbox, trên mọi ảnh)
        #   bboxes : list theo từng ảnh, mỗi phần tử [x1, y1, x2, y2]
        return images, texts, bboxes
```

Chỉ cần cài đặt tác vụ bạn dùng — base class sẽ raise `NotImplementedError` rõ
ràng cho tác vụ còn lại. Toạ độ `bbox` là giá trị pixel trong hệ ảnh trả về.

**2. Đăng ký** trong `ocr-bench/utils/datasets/__init__.py`:

```python
from utils.datasets.mydata import MyDataset

REGISTRY = {
    # ... các mục có sẵn ...
    MyDataset.name: MyDataset,
}
```

**3. Chạy** — không cần sửa các script evaluate:

```bash
python ocr-bench/evaluate/text_recognition.py \
  --dataset mydata --opt src=/path/to/data \
  --model surya --model_path ./model_path/text_recognition
```

---

## Chỉ số (Metrics)

> Công thức đầy đủ và giải thích chi tiết: [METRICS.vi.md](METRICS.vi.md).

### Detection

| Chỉ số | Mô tả |
|--------|-------|
| **Precision** | % box dự đoán phủ đúng một box GT (ngưỡng 0.5) |
| **Recall** | % box GT được dự đoán phủ (ngưỡng 0.5) |
| **F1** | Trung bình điều hòa của Precision và Recall |
| **Page IoU** | IoU hợp-đa-giác trên toàn trang |

### Recognition / Đầu-cuối

| Chỉ số | Mô tả | Tốt hơn khi |
|--------|-------|-------------|
| **CER** | Tỷ lệ lỗi ký tự (Levenshtein / độ dài tham chiếu) | Thấp hơn |
| **WER** | Tỷ lệ lỗi từ (Levenshtein mức từ) | Thấp hơn |
| **Accuracy** | Tỷ lệ khớp chuỗi tuyệt đối | Cao hơn |

---

## Thêm một mô hình mới

### Mô hình Detection hoặc Recognition

1. Tạo file cài đặt:

```python
# ocr-bench/models/detection/paddleocr.py
from models.base import BaseDetectionModel

class PaddleOCRDetectionModel(BaseDetectionModel):
    def __init__(self, checkpoint: str):
        # nạp mô hình của bạn ở đây
        ...

    def predict(self, images, batch_size=8):
        # trả về List[List[[x1, y1, x2, y2]]] — một list bbox cho mỗi ảnh
        ...
```

2. Đăng ký:

```python
# ocr-bench/models/detection/__init__.py
from models.detection.paddleocr import PaddleOCRDetectionModel

REGISTRY = {
    "surya": SuryaDetectionModel,
    "paddleocr": PaddleOCRDetectionModel,   # thêm dòng này
}
```

3. Chạy với `--model paddleocr`.

### Mô hình VLM hoặc API

Cùng mẫu — cài đặt `BaseEndToEndModel.predict()` trả về `List[List[{"text": str, "bbox": ...}]]`, rồi đăng ký trong `models/vlm/__init__.py` hoặc `models/api/__init__.py`.

---

## Kết quả

Kết quả được lưu dạng JSON dưới `--results_dir`:

```
results/
├── detection_benchmark/
│   └── <tên_dataset>/
│       └── results.json
├── recognition_benchmark/
│   └── <tên_dataset>_results.json
└── end_to_end_benchmark/
    └── <dataset>_<model_type>_<model>_results.json
```

### Tạo báo cáo Excel

```bash
python ocr-bench/make_report/text_detection.py
python ocr-bench/make_report/text_recognition.py
```

---

## Xử lý sự cố

**Hết bộ nhớ (OOM)** — giảm `--batch_size` (hoặc, với dataset `nvidia`, giới hạn
kích thước ảnh bằng `--opt max_size_limit=1024`):
```bash
python ocr-bench/evaluate/text_detection.py ... --batch_size 2 \
  --dataset nvidia --opt max_size_limit=1024
```

**Lỗi trích xuất PDF** — kiểm tra PDF là loại text (không phải scan):
```bash
python -c "import pymupdf; doc = pymupdf.open('file.pdf'); print(f'Pages: {len(doc)}')"
```

---

## Ví dụ trên Colab

```python
!git clone https://github.com/jajqja/ocr-bench.git
%cd ocr-bench
!pip install -r requirements.txt

# Chạy benchmark (trỏ --model_path tới thư mục checkpoint cục bộ của bạn)
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
