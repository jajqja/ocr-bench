# Hệ Thống Độ Đo Hệ Thống OCR (OCR Evaluation Metrics)

Tài liệu này quy định và giải thích chi tiết cách thức tính toán các chỉ số đánh giá (metrics) cho cả hai tầng trong pipeline OCR: **Textline Detection (Phát hiện dòng chữ)** và **Text Recognition (Nhận diện chữ)**.

---

## 1. Tầng Nhận Diện Chữ (Text Recognition Metrics)

Tầng này đánh giá độ chính xác của văn bản được mô hình giải mã (Predicted Text) so với nhãn gốc thực tế (Ground Truth Text). Các độ đo được tính toán dựa trên thuật toán **Khoảng cách Levenshtein (Edit Distance)** ở cấp độ ký tự và cấp độ từ.

### 1.1. Character Error Rate (CER) - Tỷ lệ lỗi ký tự
CER đo lường tỷ lệ phần trăm ký tự bị đoán sai trên tổng số ký tự gốc.

* **Công thức:**
    $$CER = \frac{S + D + I}{N_c}$$
    * $S$ (Substitutions): Số ký tự bị thay thế sai (ví dụ: `u` thành `v`).
    * $D$ (Deletions): Số ký tự bị mô hình bỏ sót.
    * $I$ (Insertions): Số ký tự bị mô hình nhận diện thừa ra.
    * $N_c$: Tổng số ký tự trong nhãn gốc (Ground Truth).
* **Đặc điểm:** Chỉ số càng thấp càng tốt ($0.0$ là hoàn hảo). CER phản ánh rất rõ khả năng nhận diện các dấu thanh tiếng Việt dễ bị mất hoặc sai lệch.

### 1.2. Word Error Rate (WER) - Tỷ lệ lỗi từ
Tương tự như CER nhưng đơn vị tính toán được chuyển từ cấp độ ký tự sang **cấp độ từ (word)** (các chuỗi ký tự cách nhau bởi khoảng trắng).

* **Công thức:**
    $$WER = \frac{S_w + D_w + I_w}{N_w}$$
    * $S_w, D_w, I_w$: Số từ bị thay thế, bỏ sót, hoặc thêm thừa.
    * $N_w$: Tổng số từ trong nhãn gốc (Ground Truth).
* **Đặc điểm:** Chỉ số càng thấp càng tốt. Một từ chỉ cần sai một ký tự duy nhất cũng bị tính là lỗi $1$ từ.

### 1.3. Accuracy (Độ chính xác tuyệt đối ở cấp độ Textline)
Đo lường tỷ lệ phần trăm các dòng chữ mà mô hình nhận diện **đúng hoàn toàn 100%** không sai một dấu vết so với Ground Truth.

* **Công thức:**
    $$\text{Accuracy} = \frac{\text{Số dòng chữ khớp chuẩn hoàn toàn}}{\text{Tổng số dòng chữ đem đi đánh giá}}$$
* **Đặc điểm:** Khắt khe nhất trong 3 chỉ số. Một dòng chữ có 100 từ, chỉ cần sai một dấu chấm hoặc viết hoa/viết thường sai cũng bị tính là $0$ điểm cho dòng đó. Chỉ số này rất quan trọng khi đánh giá chất lượng đầu ra cho các tài liệu hành chính (Administrative Documents).

---

## 2. Tầng Phát Hiện Dòng Chữ (Textline Detection Metrics)

Tự thiết kế bộ đánh giá dựa trên **Diện tích bao phủ thực tế (Coverage/Area-based Approach)** kết hợp thư viện toán hình học `shapely` để xử lý bài toán gãy dòng và trùng lắp hộp.

### 2.1. Phương pháp tính độ phủ (`calculate_coverage`)
Thay vì tính toán IoU 1-1 thông thường, hệ thống đứng từ hai góc nhìn độc lập để chấm điểm từng Bounding Box:

* **Góc nhìn Precision:** Lấy từng hộp Dự đoán (`pred`) tính tỷ lệ diện tích nó bọc trúng vào các hộp nhãn gốc (`references`).
* **Góc nhìn Recall:** Lấy từng hộp Nhãn gốc (`reference`) tính tỷ lệ diện tích nó được che phủ bởi toàn bộ các hộp dự đoán (`preds`).

### 2.2. Cơ chế hình phạt (`penalize_double=True`)
Nhằm ngăn chặn mô hình "ăn gian" điểm số bằng cách đẻ ra nhiều hộp trùng đè lên nhau (Double Detection) hoặc gãy dòng, hệ thống áp dụng nguyên lý toán học:

$$\text{Overlap Area (Diện tích trùng)} = \text{Tổng diện tích giao thô} - \text{Diện tích phủ phẳng (Union)}$$
$$\text{Final Area (Diện tích sau phạt)} = \max(0.0, \text{Diện tích phủ phẳng} - \text{Overlap Area})$$

Vùng không gian bị đè càng nhiều, diện tích phạt càng lớn, giúp triệt tiêu điểm số của các hộp dự đoán dư thừa rác.

### 2.3. Precision & Recall ở cấp độ Hộp (Object-level)
Sau khi chấm điểm độ phủ diện tích cho từng hộp, hệ thống áp một ngưỡng nghiêm ngặt **$\text{Threshold} = 0.7$ (hoặc $0.8$)** để dán nhãn Đúng ($1$) hoặc Sai ($0$). Ngưỡng này đảm bảo các hộp dự đoán làm mất dấu thanh hoặc cắt cụt chữ sẽ bị loại bỏ.

* **Precision (Độ chính xác):**
    $$\text{Precision} = \frac{\text{Số lượng Bbox dự đoán đạt chuẩn độ phủ (> Threshold)}}{\text{Tổng số lượng Bbox mô hình đưa ra}}$$
* **Recall (Độ nhạy):**
    $$\text{Recall} = \frac{\text{Số lượng dòng chữ thật đạt chuẩn độ phủ (> Threshold)}}{\text{Tổng số lượng dòng chữ thực tế}}$$

---

## 3. Thang Hình Phạt Tổng Hợp (Penalized IoU)

Đối với các báo cáo cần quy về một chỉ số IoU tổng hợp đại diện duy nhất (`match_boxes`), hệ thống áp dụng cơ chế phân hóa hình phạt dựa trên mức độ nghiêm trọng của lỗi đối với pipeline OCR:

| Loại trạng thái hình học | Điểm IoU gán | Mức độ nặng | Ý nghĩa thực tế |
| :--- | :---: | :---: | :--- |
| **Khớp chuẩn hình học** | Từ $0.7 \rightarrow 1.0$ | Không phạt | Hộp bọc vừa vặn, ôm khít dòng chữ thật. |
| **Dự đoán dư thừa / Nhiễu** (`unassigned_pred`) | **$0.0$** | Nhẹ | Mô hình vẽ bậy ra nền trống. Tầng Recognition cắt ra chuỗi rỗng $\rightarrow$ dễ lọc bỏ. |
| **Bỏ sót hoàn toàn chữ** (`unassigned_actual`) | **$-1.0$** | Cực nặng | Mô hình làm mất hẳn dòng chữ. Tầng OCR phía sau hoàn toàn mù tịt $\rightarrow$ Phạt sập điểm hệ thống. |