"""Instance-level and face-level evaluation metrics."""

from __future__ import annotations

from typing import Dict, List, Tuple


def _face_iou(pred_faces: List[int], gt_faces: List[int]) -> float:
    """Intersection-over-Union at face level."""
    p, g = set(pred_faces), set(gt_faces)
    inter = len(p & g)
    union = len(p | g)
    return inter / max(1, union)


def _face_precision_recall(
    pred_faces: List[int],
    gt_faces: List[int],
) -> Tuple[float, float]:
    p, g = set(pred_faces), set(gt_faces)
    if not p:
        return 0.0, 0.0
    precision = len(p & g) / len(p)
    recall    = len(p & g) / max(1, len(g))
    return precision, recall


def instance_f1(
    predicted: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute instance-level Precision, Recall, F1 using IoU matching.

    A predicted instance matches a GT instance if face IoU >= iou_threshold.
    Each GT is matched at most once (greedy, sorted by confidence).

    Args:
        predicted    : list of {face_ids, confidence}
        ground_truth : list of {face_ids}
        iou_threshold: minimum IoU to count as a true positive

    Returns:
        (precision, recall, f1)
    """
    predicted    = sorted(predicted, key=lambda x: -x.get("confidence", 0.0))
    gt_matched   = [False] * len(ground_truth)
    tp = 0

    for pred in predicted:
        pred_faces = pred["face_ids"]
        best_iou   = 0.0
        best_gt_i  = -1
        for gi, gt in enumerate(ground_truth):
            if gt_matched[gi]:
                continue
            iou = _face_iou(pred_faces, gt["face_ids"])
            if iou > best_iou:
                best_iou  = iou
                best_gt_i = gi
        if best_iou >= iou_threshold and best_gt_i >= 0:
            tp += 1
            gt_matched[best_gt_i] = True

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)
    return precision, recall, f1


def face_level_f1(
    pred_face_ids: List[int],
    gt_face_ids: List[int],
) -> Tuple[float, float, float]:
    """Compute face-level Precision, Recall, F1."""
    precision, recall = _face_precision_recall(pred_face_ids, gt_face_ids)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return precision, recall, f1


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    """Brier score: mean squared error between confidence and correctness.

    Lower is better. Perfect calibration = 0.0.

    Args:
        y_true: List of 1 (correct) or 0 (incorrect)
        y_prob: List of predicted confidences

    Returns:
        Brier score in [0, 1]
    """
    import numpy as np

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def expected_calibration_error(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE): weighted average calibration error.

    Args:
        y_true: List of 1 (correct) or 0 (incorrect)
        y_prob: List of predicted confidences
        n_bins: Number of bins for calibration

    Returns:
        ECE in [0, 1], lower is better
    """
    import numpy as np

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob_arr >= bins[i]) & (y_prob_arr < bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = y_true_arr[mask].mean()
        bin_conf = y_prob_arr[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


def compute_calibration_metrics(
    results: List[Dict],
) -> Dict[str, float]:
    """Compute Brier score and ECE from instance results.

    Args:
        results: List of {confidence: float, is_true_positive: bool}

    Returns:
        Dictionary with brier_score and ece (both 0.0 if results is empty)
    """
    if not results:
        return {"brier_score": 0.0, "ece": 0.0}

    y_true = [int(r["is_true_positive"]) for r in results]
    y_prob = [r["confidence"] for r in results]

    return {
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
