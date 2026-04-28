"""Non-maximum suppression for overlapping feature clusters."""

from __future__ import annotations

from typing import Dict, List


def _iou(set_a: set, set_b: set) -> float:
    """Face-level IoU between two face-index sets."""
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(1, union)


def non_max_suppression(
    instances: List[Dict],
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """Remove duplicate clusters with IoU > iou_threshold.

    Keeps the higher-confidence detection when two clusters overlap.

    Args:
        instances     : list of instance dicts, must include ``_cluster_set``
                        (set of integer face indices) and ``confidence`` keys.
        iou_threshold : clusters with IoU > this are suppressed.

    Returns:
        Filtered list of instance dicts.
    """
    if not instances:
        return []

    # Sort descending by confidence
    instances = sorted(instances, key=lambda x: -x["confidence"])
    kept = []

    for candidate in instances:
        c_set = candidate.get("_cluster_set", set(candidate.get("face_ids", [])))
        suppress = False
        for kept_inst in kept:
            k_set = kept_inst.get("_cluster_set", set(kept_inst.get("face_ids", [])))
            if _iou(c_set, k_set) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)

    return kept
