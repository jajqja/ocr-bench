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

## 2. Textline Detection Metrics


### 2.1. Precision & Recall ở Object-level
Tự thiết kế bộ đánh giá dựa trên **Diện tích bao phủ thực tế (Coverage/Area-based Approach)** kết hợp thư viện toán hình học `shapely` để xử lý bài toán gãy dòng và trùng lắp hộp.

#### Phương pháp tính độ phủ

* **Precision:** Lấy từng hộp Dự đoán (`pred`) tính tỷ lệ diện tích nó bọc trúng vào các hộp nhãn gốc (`references`).
* **Recall:** Lấy từng hộp Nhãn gốc (`reference`) tính tỷ lệ diện tích nó được che phủ bởi toàn bộ các hộp dự đoán (`preds`).

#### Tính toán Precision & Recall
Sau khi chấm điểm độ phủ diện tích cho từng hộp, hệ thống áp một ngưỡng chấp nhận được **$\text{Threshold} = 0.5$** để dán nhãn Đúng ($1$) hoặc Sai ($0$). Ngưỡng này đảm bảo các hộp dự đoán làm mất dấu thanh hoặc cắt cụt chữ sẽ bị loại bỏ.

* **Precision (Độ chính xác):**
    $$\text{Precision} = \frac{\text{Số lượng Bbox dự đoán đạt chuẩn độ phủ (> Threshold)}}{\text{Tổng số lượng Bbox mô hình đưa ra}}$$
* **Recall (Độ nhạy):**
    $$\text{Recall} = \frac{\text{Số lượng dòng chữ thật đạt chuẩn độ phủ (> Threshold)}}{\text{Tổng số lượng dòng chữ thực tế}}$$
* **F1-Score (Điểm F1 tổng hợp):**
    $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
    *Chỉ số **F1-score** là trung bình điều hòa giữa Precision và Recall. **F1-Score** đóng vai trò là thước đo đại diện duy nhất để đánh giá toàn diện năng lực của mô hình Detection, đảm bảo mô hình không bị lệch về hướng bắt thừa **(Precision thấp)** hay bỏ sót **(Recall thấp)**.*

---

### 2.2. IoU cấp độ trang (page-level)

Độ đo này đánh giá sự khít khao, vuông vức và độ chính xác hình học tổng thể của toàn bộ hệ thống hộp dự đoán (`preds`) so với dòng nhãn gốc (`references`) trên phạm vi toàn bức ảnh.

#### Quy ước hình học
Thay vì tính toán IoU cho từng cặp hộp riêng lẻ (vốn dễ bị đánh lừa bởi lỗi gãy dòng hoặc trùng đè), sử dụng thư viện `shapely` để xử lý liên kết topo phẳng:
1. Gom toàn bộ các hộp dự đoán đơn lẻ thành một khối đa giác phẳng duy nhất ($\mathcal{P}_{\text{union}}$).
2. Gom toàn bộ các hộp thực tế đơn lẻ thành một khối đa giác phẳng duy nhất ($\mathcal{G}_{\text{union}}$).

#### 2.2.2. Công thức toán học

Chỉ số **Page-Level IoU** được tính toán dựa trên tỷ lệ giữa diện tích giao phẳng và diện tích hợp phẳng của hai khối thông tin khổng lồ này:

$$IoU_{\text{page}} = \frac{\text{Area}(\mathcal{P}_{\text{union}} \cap \mathcal{G}_{\text{union}})}{\text{Area}(\mathcal{P}_{\text{union}} \cup \mathcal{G}_{\text{union}})}$$

*Với:*

$$\text{Area}(\mathcal{P}_{\text{union}} \cup \mathcal{G}_{\text{union}}) = \text{Area}(\mathcal{P}_{\text{union}}) + \text{Area}(\mathcal{G}_{\text{union}}) - \text{Area}(\mathcal{P}_{\text{union}} \cap \mathcal{G}_{\text{union}})$$

*Trong đó:*
* $\mathcal{P}_{\text{union}} = \bigcup_{p \in \text{preds}} p$: Đa giác phủ phẳng của tập hợp các hộp dự đoán.
* $\mathcal{G}_{\text{union}} = \bigcup_{g \in \text{references}} g$: Đa giác phủ phẳng của tập hợp các hộp nhãn gốc.
* $\cap$: Phép toán giao hai tập hợp hình học (Intersection).
* $\cup$: Phép toán hợp hai tập hợp hình học (Union).

#### Các trường hợp biên (Corner Cases)
Hệ thống quy định chặt chẽ các giá trị trả về trong trường hợp trang tài liệu trống hoặc mô hình không phát hiện được thực thể:
* **$IoU_{\text{page}} = 1.0$**: Khi trang tài liệu hoàn toàn trống (không có chữ thật) và mô hình dự đoán cũng hoàn toàn sạch sẽ, không vẽ bậy ($\text{len}(\text{gts}) = 0 \text{ and } \text{len}(\text{preds}) = 0$).
* **$IoU_{\text{page}} = 0.0$**: Khi một trong hai tập hợp bị trống hoàn toàn trong khi tập hợp còn lại có dữ liệu (ví dụ: ảnh có chữ nhưng mô hình sót $100\%$, hoặc ảnh trống nhưng mô hình vẽ bậy khắp nơi).
* **Ngưỡng chấp nhận**: Trong các bài toán nhận diện vật thể (Object Detection), kết quả dự đoán thường được coi là đúng (True Positive) nếu **$IoU ≥ 0.5$**