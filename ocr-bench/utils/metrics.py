from functools import partial
from itertools import repeat

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
from tqdm import tqdm
from shapely.geometry import box as ShapelyBox
from shapely.ops import unary_union


def calculate_iou(preds, gts):
    """
    Tính IoU tổng hợp trên toàn bộ trang giấy.
    Công thức: Tổng diện tích giao nhau của (Preds và GTs) / Diện tích hợp nhau của (Preds và GTs)

    Format input:
    preds: List gồm các bbox dạng [x1, y1, x2, y2]
    gts: List gồm các bbox dạng [x1, y1, x2, y2]
    """
    if len(gts) == 0 and len(preds) == 0:
        return 1.0
    if len(gts) == 0 or len(preds) == 0:
        return 0.0

    pred_polys = [ShapelyBox(b[0], b[1], b[2], b[3]) for b in preds]
    gt_polys = [ShapelyBox(b[0], b[1], b[2], b[3]) for b in gts]

    pred_union = unary_union(pred_polys)
    gt_union = unary_union(gt_polys)

    intersection_poly = pred_union.intersection(gt_union)
    intersection_area = (
        intersection_poly.area if not intersection_poly.is_empty else 0.0
    )

    union_area = pred_union.area + gt_union.area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_coverage(box, other_boxes, penalize_double=True):
    """Tính tỷ lệ bao phủ của box bởi các other_boxes sử dụng Shapely."""
    main_poly = ShapelyBox(box[0], box[1], box[2], box[3])
    if main_poly.area == 0 or len(other_boxes) == 0:
        return 0.0

    intersections = []
    for ob in other_boxes:
        other_poly = ShapelyBox(ob[0], ob[1], ob[2], ob[3])
        inter = main_poly.intersection(other_poly)
        if not inter.is_empty:
            intersections.append(inter)

    if not intersections:
        return 0.0

    net_coverage_area = unary_union(intersections).area

    if not penalize_double:
        return net_coverage_area / main_poly.area
    else:
        gross_intersection_area = sum(inter.area for inter in intersections)
        overlap_area = gross_intersection_area - net_coverage_area

        final_area = max(0.0, net_coverage_area - overlap_area)
        return final_area / main_poly.area


def precision_recall_f1_coverage(
    preds, references, threshold=0.5, workers=8, penalize_double=True
):
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

    precision_classes = [1 if i > threshold else 0 for i in precision_iou]
    precision = sum(precision_classes) / len(precision_classes)

    recall_classes = [1 if i > threshold else 0 for i in reference_iou]
    recall = sum(recall_classes) / len(recall_classes)

    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        ),
    }


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
