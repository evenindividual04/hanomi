"""Rule-based topological pattern matching baseline (Baseline 0).

Matches machining features by checking for canonical face-type sequences
connected by concave edges.  Fast but brittle to feature intersections.

Counterbored hole pattern:
  [cylinder, large radius]
      → concave edge →
  [flat annular face (plane)]
      → concave edge →
  [cylinder, small radius]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data


# Surface type constants (MFCAD++ V_1 column 4)
SURFACE_PLANE    = 0
SURFACE_CYLINDER = 1
SURFACE_CONE     = 2
SURFACE_SPHERE   = 3
SURFACE_TORUS    = 4
SURFACE_OTHER    = 5

# Edge convexity (edge_attr column 0)
CONVEX  = 1.0
CONCAVE = -1.0
SMOOTH  = 0.0


def _get_surface_type(data: Data, face_idx: int) -> int:
    return round(float(data.x[face_idx, 4].item()) * 11)


def _get_area(data: Data, face_idx: int) -> float:
    return float(data.x[face_idx, 0].item())


def _get_neighbors_by_convexity(
    data: Data,
    face_idx: int,
    convexity: float,
    tol: float = 0.1,
) -> List[Tuple[int, float]]:
    """Return (neighbor_idx, edge_area) for edges of given convexity type."""
    neighbors = []
    for col in range(data.edge_index.size(1)):
        src = int(data.edge_index[0, col])
        dst = int(data.edge_index[1, col])
        if src != face_idx:
            continue
        conv = float(data.edge_attr[col, 0]) if data.edge_attr is not None else 0.0
        if abs(conv - convexity) < tol:
            area = _get_area(data, dst)
            neighbors.append((dst, area))
    return neighbors


def match_through_hole(data: Data) -> List[Dict]:
    """Find through-hole instances (single cylinder face).

    A through-hole in MFCAD++ is a single cylindrical face with concave
    edges to flat (planar) faces on both ends.

    Returns:
        list of {face_ids, confidence}
    """
    instances = []
    visited: Set[int] = set()

    for fi in range(data.num_nodes):
        if fi in visited:
            continue
        if _get_surface_type(data, fi) != SURFACE_CYLINDER:
            continue
        # Check for concave edges to planar caps
        concave_nbrs = _get_neighbors_by_convexity(data, fi, CONCAVE)
        planar_caps  = [n for n, _ in concave_nbrs if _get_surface_type(data, n) == SURFACE_PLANE]
        if len(planar_caps) >= 2:
            visited.add(fi)
            instances.append({
                "face_ids":   [fi],
                "confidence": 0.85,
                "_cluster_set": {fi},
            })

    return instances


def match_counterbored_hole(data: Data) -> List[Dict]:
    """Find counterbored-hole instances.

    Pattern (simplified):
      outer_cylinder (large r)
        → concave →  annular_plane
        → concave →  inner_cylinder (small r)

    Returns:
        list of {face_ids, confidence}
    """
    instances = []

    for fi in range(data.num_nodes):
        if _get_surface_type(data, fi) != SURFACE_CYLINDER:
            continue
        outer_area = _get_area(data, fi)

        # Find planar annular face connected by concave edge
        for plane_idx, _ in _get_neighbors_by_convexity(data, fi, CONCAVE):
            if _get_surface_type(data, plane_idx) != SURFACE_PLANE:
                continue
            # Find smaller cylinder connected from the annular plane
            for cyl_idx, cyl_area in _get_neighbors_by_convexity(data, plane_idx, CONCAVE):
                if _get_surface_type(data, cyl_idx) != SURFACE_CYLINDER:
                    continue
                if cyl_idx == fi:
                    continue
                # Smaller cylinder heuristic: area < outer cylinder
                if cyl_area < outer_area:
                    face_ids = sorted({fi, plane_idx, cyl_idx})
                    instances.append({
                        "face_ids":   face_ids,
                        "confidence": 0.80,
                        "_cluster_set": set(face_ids),
                    })

    # Deduplicate by face set
    seen: Set[frozenset] = set()
    uniq = []
    for inst in instances:
        key = frozenset(inst["face_ids"])
        if key not in seen:
            seen.add(key)
            uniq.append(inst)
    return uniq


def run_rule_based(data: Data, feature_type: str) -> List[Dict]:
    """Dispatch to the correct rule-based matcher.

    Args:
        data         : PyG Data for one model
        feature_type : "through_hole" | "counterbored_hole" | ...

    Returns:
        list of instance dicts (face_ids, confidence)
    """
    if feature_type in ("through_hole",):
        return match_through_hole(data)
    elif feature_type in ("counterbored_hole", "blind_hole"):
        return match_counterbored_hole(data)
    else:
        raise NotImplementedError(f"No rule-based matcher for '{feature_type}'")
