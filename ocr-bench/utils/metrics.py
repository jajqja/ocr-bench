from functools import partial
from itertools import repeat
from typing import List

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
from tqdm import tqdm


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


def calculate_iou(box1, box2, box1_only=False):
    intersection = intersection_area(box1, box2)
    union = box_area(box1)
    if not box1_only:
        union += box_area(box2) - intersection

    if union == 0:
        return 0
    return intersection / union


def match_boxes(preds, references):
    num_actual = len(references)
    num_predicted = len(preds)

    iou_matrix = np.zeros((num_actual, num_predicted))
    for i, actual in enumerate(references):
        for j, pred in enumerate(preds):
            iou_matrix[i, j] = calculate_iou(actual, pred, box1_only=True)

    sorted_indices = np.argsort(iou_matrix, axis=None)[::-1]
    sorted_ious = iou_matrix.flatten()[sorted_indices]
    actual_indices, predicted_indices = np.unravel_index(
        sorted_indices, iou_matrix.shape
    )

    assigned_actual = set()
    assigned_pred = set()

    matches = []
    for idx, iou in zip(zip(actual_indices, predicted_indices), sorted_ious):
        i, j = idx
        if i not in assigned_actual and j not in assigned_pred:
            iou_val = iou_matrix[i, j]
            if iou_val > 0.95:  # Account for rounding on box edges
                iou_val = 1.0
            matches.append((i, j, iou_val))
            assigned_actual.add(i)
            assigned_pred.add(j)

    unassigned_actual = set(range(num_actual)) - assigned_actual
    unassigned_pred = set(range(num_predicted)) - assigned_pred
    matches.extend([(i, None, -1.0) for i in unassigned_actual])
    matches.extend([(None, j, 0.0) for j in unassigned_pred])

    return matches


def penalized_iou_score(preds, references):
    matches = match_boxes(preds, references)
    iou = sum([match[2] for match in matches]) / len(matches)
    return iou


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


def calculate_coverage(box, other_boxes, penalize_double=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    # find total coverage of the box
    covered_pixels = set()
    double_coverage = list()
    for other_box in other_boxes:
        ia = intersection_pixels(box, other_box)
        double_coverage.append(list(covered_pixels.intersection(ia)))
        covered_pixels = covered_pixels.union(ia)

    # Penalize double coverage - having multiple bboxes overlapping the same pixels
    double_coverage_penalty = len(double_coverage)
    if not penalize_double:
        double_coverage_penalty = 0
    covered_pixels_count = max(0, len(covered_pixels) - double_coverage_penalty)
    return covered_pixels_count / box_area


def intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def calculate_coverage_fast(box, other_boxes, penalize_double=False):
    box = np.array(box)
    other_boxes = np.array(other_boxes)

    # Calculate box area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    x_left = np.maximum(box[0], other_boxes[:, 0])
    y_top = np.maximum(box[1], other_boxes[:, 1])
    x_right = np.minimum(box[2], other_boxes[:, 2])
    y_bottom = np.minimum(box[3], other_boxes[:, 3])

    widths = np.maximum(0, x_right - x_left)
    heights = np.maximum(0, y_bottom - y_top)
    intersect_areas = widths * heights

    total_intersect = np.sum(intersect_areas)

    return min(1.0, total_intersect / box_area)


def precision_recall(preds, references, threshold=0.5, workers=8, penalize_double=True):
    if len(references) == 0:
        return {
            "precision": 1,
            "recall": 1,
        }

    if len(preds) == 0:
        return {
            "precision": 0,
            "recall": 0,
        }

    # If we're not penalizing double coverage, we can use a faster calculation
    coverage_func = calculate_coverage_fast
    if penalize_double:
        coverage_func = calculate_coverage

    with ThreadPoolExecutor(max_workers=workers) as executor:
        precision_func = partial(coverage_func, penalize_double=penalize_double)
        precision_iou = executor.map(precision_func, preds, repeat(references))
        reference_iou = executor.map(coverage_func, references, repeat(preds))

    precision_classes = [1 if i > threshold else 0 for i in precision_iou]
    precision = sum(precision_classes) / len(precision_classes)

    recall_classes = [1 if i > threshold else 0 for i in reference_iou]
    recall = sum(recall_classes) / len(recall_classes)

    return {
        "precision": precision,
        "recall": recall,
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
    - Removes leading/trailing whitespaces
    - Collapses multiple whitespaces into a single space
    """
    if not text:
        return ""

    # 1. Ép về chuỗi viết thường
    text = text.lower()

    # 2. Thay thế các ký tự xuống dòng, tab... thành khoảng trắng đơn
    text = re.sub(r"\s+", " ", text)

    # 3. Cắt khoảng trắng thừa ở 2 đầu
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
        "accuracy": recognition_accuracy(references, hypotheses),
        "cer_scores": cer_scores,
        "wer_scores": wer_scores,
    }
