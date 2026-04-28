"""Compute calibration metrics from existing inference results.

Reads a per_model_results.json produced by evaluate.py and computes
Brier score and ECE (Expected Calibration Error) without re-running inference.

Usage:
  python scripts/compute_calibration.py results/runs/run_001/eval/per_model_results.json
  python scripts/compute_calibration.py results/runs/run_002/eval/per_model_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_calibration_metrics


def _face_iou(pred_faces, gt_faces) -> float:
    p, g = set(pred_faces), set(gt_faces)
    inter = len(p & g)
    union = len(p | g)
    return inter / max(1, union)


def _is_true_positive(pred: dict, gt_instances: list, iou_threshold: float = 0.5) -> bool:
    """Return True if pred matches any GT instance at the given IoU threshold."""
    pred_faces = pred.get("face_ids", [])
    for gt in gt_instances:
        if _face_iou(pred_faces, gt.get("face_ids", [])) >= iou_threshold:
            return True
    return False


def load_calibration_pairs(path: Path, iou_threshold: float = 0.5) -> list[dict]:
    """Extract (confidence, is_true_positive) pairs from per_model_results.json."""
    with open(path) as f:
        results = json.load(f)

    pairs = []
    for model in results:
        pred_instances = model.get("predicted_instances", [])
        gt_instances = model.get("gt_instances", [])
        for pred in pred_instances:
            pairs.append({
                "confidence": float(pred.get("confidence", 0.0)),
                "is_true_positive": _is_true_positive(pred, gt_instances, iou_threshold),
            })
    return pairs


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_calibration.py <per_model_results.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    pairs = load_calibration_pairs(path)
    if not pairs:
        print("No predictions found in results file.", file=sys.stderr)
        sys.exit(1)

    metrics = compute_calibration_metrics(pairs)
    metrics["n_predictions"] = len(pairs)
    metrics["n_tp"] = sum(1 for p in pairs if p["is_true_positive"])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
