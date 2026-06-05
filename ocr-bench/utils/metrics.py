from functools import partial
from itertools import repeat
from typing import List

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
from tqdm import tqdm
from shapely.geometry import box as ShapelyBox
from shapely.ops import unary_union


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou_metrics(predictions: List[List], ground_truth: List[List]) -> float:
    """Calculate mean IOU score for detection.

    Args:
        predictions: List of predicted bounding boxes [[x1,y1,x2,y2], ...]
        ground_truth: List of ground truth bounding boxes

    Returns:
        Mean IOU score
    """
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0

    return penalized_iou_score(predictions, ground_truth)


def penalized_iou_score(preds, references):
    matches = match_boxes(preds, references)
    iou = sum([match[2] for match in matches]) / len(matches)
    return iou


def match_boxes(preds, references, coverage_threshold=0.5):
    num_actual = len(references)
    num_predicted = len(preds)

    if num_actual == 0 or num_predicted == 0:
        return [(i, None, -1.0) for i in range(num_actual)] + [
            (None, j, 0.0) for j in range(num_predicted)
        ]

    # 1. Tính ma trận bao phủ (box1_only=True)
    # iou_matrix[i, j] nghĩa là: Pred_j bao phủ bao nhiêu % diện tích của GT_i
    iou_matrix = np.zeros((num_actual, num_predicted))
    for i, actual in enumerate(references):
        for j, pred in enumerate(preds):
            iou_matrix[i, j] = calculate_iou(actual, pred, box1_only=True)

    assigned_actual = set()
    assigned_pred = set()
    matches = []

    # 2. Bước quét thứ nhất: Tìm các cặp khớp tốt nhất vượt ngưỡng (Ví dụ > 0.5)
    # Không dùng vòng lặp Greedy toàn cục nữa, mà xét điều kiện bao phủ cho từng thực thể

    # Đối với mỗi GT_i, tìm xem có Pred_j nào bao phủ nó tốt không
    for i in range(num_actual):
        best_j = np.argmax(iou_matrix[i, :])
        highest_coverage = iou_matrix[i, best_j]

        if highest_coverage >= coverage_threshold:
            if highest_coverage > 0.95:
                highest_coverage = 1.0
            matches.append((i, best_j, highest_coverage))
            assigned_actual.add(i)
            assigned_pred.add(best_j)

    # Đối với mỗi Pred_j chưa được gán, kiểm tra xem nó có nằm gọn trong GT_i nào không
    # (Để giải quyết trường hợp 2 Pred nằm trong 1 GT mà bạn nói)
    for j in range(num_predicted):
        if j in assigned_pred:
            continue
        # Tìm GT_i mà Pred_j này bao phủ tốt nhất (hoặc ngược lại)
        best_i = np.argmax(iou_matrix[:, j])
        # Nếu Pred_j này đóng góp bao phủ tốt cho một GT_i nào đó
        if iou_matrix[best_i, j] >= coverage_threshold:
            matches.append((best_i, j, iou_matrix[best_i, j]))
            assigned_actual.add(best_i)
            assigned_pred.add(j)

    # 3. Gom các hộp không được giao và áp mức phạt theo ý bạn
    unassigned_actual = set(range(num_actual)) - assigned_actual
    unassigned_pred = set(range(num_predicted)) - assigned_pred

    # GT sót: Phạt cực nặng -1.0
    matches.extend([(i, None, -1.0) for i in unassigned_actual])
    # Pred dư/nhiễu: Phạt 0.0
    matches.extend([(None, j, 0.0) for j in unassigned_pred])

    return matches


def calculate_iou(box1, box2, box1_only=False):
    intersection = intersection_area(box1, box2)  # Area of overlap
    union = box_area(box1)  # Area of box1
    if not box1_only:
        union += box_area(box2) - intersection  # Total area covered by both boxes

    if union == 0:
        return 0
    return intersection / union


def intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def intersection_pixels(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return set()

    x_left, x_right = int(x_left), int(x_right)
    y_top, y_bottom = int(y_top), int(y_bottom)

    coords = np.meshgrid(np.arange(x_left, x_right), np.arange(y_top, y_bottom))
    pixels = set(zip(coords[0].flat, coords[1].flat))

    return pixels


def calculate_coverage(box, other_boxes, penalize_double=True):
    """Tính tỷ lệ bao phủ của box bởi các other_boxes sử dụng Shapely."""
    main_poly = ShapelyBox(box[0], box[1], box[2], box[3])
    if main_poly.area == 0 or len(other_boxes) == 0:
        return 0.0

    # Lấy danh sách các vùng giao nhau thực tế với hộp gốc
    intersections = []
    for ob in other_boxes:
        other_poly = ShapelyBox(ob[0], ob[1], ob[2], ob[3])
        inter = main_poly.intersection(other_poly)
        if not inter.is_empty:
            intersections.append(inter)

    if not intersections:
        return 0.0

    # Diện tích phủ phẳng thực tế (Vùng chồng lấn chỉ tính 1 lần duy nhất)
    net_coverage_area = unary_union(intersections).area

    if not penalize_double:
        # Nếu KHÔNG phạt trùng lắp: Trả về tỷ lệ phủ phẳng thuần túy
        return net_coverage_area / main_poly.area
    else:
        # Nếu CÓ phạt trùng lắp:
        # Lấy tổng diện tích giao thô (có tính trùng) trừ đi diện tích phủ phẳng
        gross_intersection_area = sum(inter.area for inter in intersections)
        overlap_area = gross_intersection_area - net_coverage_area

        # Phạt bằng cách lấy diện tích phủ phẳng trừ đi diện tích bị tính trùng
        final_area = max(0.0, net_coverage_area - overlap_area)
        return final_area / main_poly.area


def precision_recall_f1(preds, references, threshold=0.7, workers=8, penalize_double=True):
    """Tính Precision và Recall dựa trên diện tích bao phủ."""
    if len(references) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if len(preds) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Đồng nhất sử dụng duy nhất hàm calculate_coverage của Shapely để đảm bảo logic đúng
    with ThreadPoolExecutor(max_workers=workers) as executor:
        precision_func = partial(calculate_coverage, penalize_double=penalize_double)

        # Song song hóa việc tính toán cho từng bounding box
        precision_iou = executor.map(precision_func, preds, repeat(references))
        reference_iou = executor.map(precision_func, references, repeat(preds))

    # Áp ngưỡng threshold (Bạn nên để 0.7 hoặc 0.8 cho bài toán textline như đã thảo luận)
    precision_classes = [1 if i > threshold else 0 for i in precision_iou]
    precision = sum(precision_classes) / len(precision_classes)

    recall_classes = [1 if i > threshold else 0 for i in reference_iou]
    recall = sum(recall_classes) / len(recall_classes)

    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
    }


def mean_coverage(preds, references):
    coverages = []

    for box1 in references:
        coverage = calculate_coverage(box1, preds)
        coverages.append(coverage)

    for box2 in preds:
        coverage = calculate_coverage(box2, references)
        coverages.append(coverage)

    # Calculate the average coverage over all comparisons
    if len(coverages) == 0:
        return 0
    coverage = sum(coverages) / len(coverages)
    return {"coverage": coverage}


def rank_accuracy(preds, references):
    # Preds and references need to be aligned so each position refers to the same bbox
    pairs = []
    for i, pred in enumerate(preds):
        for j, pred2 in enumerate(preds):
            if i == j:
                continue
            pairs.append((i, j, pred > pred2))

    # Find how many of the prediction rankings are correct
    correct = 0
    for i, ref in enumerate(references):
        for j, ref2 in enumerate(references):
            if (i, j, ref > ref2) in pairs:
                correct += 1

    return correct / len(pairs)


# ==================== RECOGNITION METRICS ====================


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) using Levenshtein distance.

    CER = (S + D + I) / N
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of characters in reference

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        CER value between 0 and 1 (or higher if insertions exceed reference length)
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    # Calculate Levenshtein distance
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(reference)][len(hypothesis)] / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) using Levenshtein distance on words.

    WER = (S + D + I) / N
    where words are separated by whitespace.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        WER value between 0 and 1
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Calculate Levenshtein distance on words
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def recognition_accuracy(references: list, hypotheses: list) -> float:
    """Calculate exact match accuracy for OCR recognition.

    Accuracy = number of perfect matches / total number of samples

    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts

    Returns:
        Accuracy value between 0 and 1
    """
    if len(references) == 0:
        return 1.0

    correct = sum(1 for ref, hyp in zip(references, hypotheses) if ref == hyp)
    return correct / len(references)


def clean_text(text: str) -> str:
    """Clean and normalize text for OCR evaluation (CER/WER).

    - Converts to lowercase
    - Replaces non-alphanumeric characters with space
    - Collapses multiple whitespaces into a single space
    - Removes leading/trailing whitespaces
    """
    if not text:
        return ""

    # 1. Ép về chuỗi viết thường
    text = text.lower()

    # 2. Thay các kí tự KHÔNG phải chữ và số thành khoảng trắng
    # Regex này giữ lại chữ cái (bao gồm tiếng Việt có dấu) và chữ số
    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub("_", " ", text)

    # 3. Thay thế các ký tự xuống dòng, tab, nhiều khoảng trắng... thành khoảng trắng đơn
    text = re.sub(r"\s+", " ", text)

    # 4. Cắt khoảng trắng thừa ở 2 đầu
    text = text.strip()

    return text


def calculate_recognition_metrics(
    references: list, hypotheses: list, show_progress: bool = False
) -> dict:
    """Calculate all recognition metrics at once.

    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        show_progress: Whether to show a progress bar

    Returns:
        Dictionary with cer, wer, and accuracy
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )

    cleaned_refs = [clean_text(ref) for ref in references]
    cleaned_hyps = [clean_text(hyp) for hyp in hypotheses]

    # Calculate scores with optional progress bar
    desc = "Calculating metrics" if show_progress else None
    cer_scores = [
        character_error_rate(r, h)
        for r, h in tqdm(
            zip(cleaned_refs, cleaned_hyps),
            total=len(references),
            desc=desc,
            disable=not show_progress,
        )
    ]
    wer_scores = [
        word_error_rate(r, h)
        for r, h in tqdm(
            zip(cleaned_refs, cleaned_hyps),
            total=len(references),
            desc=desc,
            disable=not show_progress,
        )
    ]

    return {
        "cer": np.mean(cer_scores) if cer_scores else 0.0,
        "wer": np.mean(wer_scores) if wer_scores else 0.0,
        "accuracy": recognition_accuracy(cleaned_refs, cleaned_hyps),
        "cer_scores": cer_scores,
        "wer_scores": wer_scores,
    }
