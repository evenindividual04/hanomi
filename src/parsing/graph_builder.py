"""Converts parsed pythonocc dictionaries into pure PyTorch Geometric Data objects.

Takes the outputs from `feature_extractor.py` and creates a homogeneous
graph compatible with `src.models.FeatureRecognizer`.
"""

from typing import Dict, List, Tuple
import logging
import numpy as np
import torch
from torch_geometric.data import Data

from src.parsing.feature_extractor import extract_node_features, extract_edge_features


def merge_seam_faces(faces: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Detect and merge co-planar or co-cylindrical faces that share a seam edge.

    Some CAD kernels split 360° cylinders into two 180° halves (seam anomaly).
    This function detects and merges them.

    Detection: two cylinder faces with identical radius, identical axis direction,
    and a shared edge with dihedral angle < 1 degree.

    Args:
        faces: List of face dictionaries
        edges: List of edge dictionaries

    Returns:
        (merged_faces, merged_edges)
    """
    merged_faces = []
    skip_indices = set()
    merge_count = 0

    for i, face_i in enumerate(faces):
        if i in skip_indices:
            continue
        for j, face_j in enumerate(faces[i+1:], start=i+1):
            if j in skip_indices:
                continue
            if _is_seam_pair(face_i, face_j, edges):
                merged = _merge_face_pair(face_i, face_j)
                merged_faces.append(merged)
                skip_indices.update([i, j])
                merge_count += 1
                break
        else:
            merged_faces.append(face_i)

    if merge_count > 0:
        logging.info(f"Merged {merge_count} seam face pairs")

    merged_edges = _rebuild_adjacency(merged_faces, edges, skip_indices)
    return merged_faces, merged_edges


def _is_seam_pair(face_i: Dict, face_j: Dict, edges: List[Dict]) -> bool:
    """Check if two faces form a seam pair.

    Returns True if both are cylinders with identical radius and axis,
    and share an edge with small dihedral angle.
    """
    # Must both be cylinders
    if face_i.get("surface_type", "OTHER") != "CYLINDER":
        return False
    if face_j.get("surface_type", "OTHER") != "CYLINDER":
        return False

    # Must have nearly identical radius
    radius_i = face_i.get("cylinder_radius", 0.0)
    radius_j = face_j.get("cylinder_radius", 0.0)
    if abs(radius_i - radius_j) > 0.001:
        return False

    # Must have nearly identical axis direction
    axis_i = face_i.get("cylinder_axis", [0, 0, 0])
    axis_j = face_j.get("cylinder_axis", [0, 0, 0])
    if np.linalg.norm(np.array(axis_i) - np.array(axis_j)) > 0.01:
        return False

    # Must share an edge
    face_i_edges = set()
    face_j_edges = set()

    for edge_idx, edge in enumerate(edges):
        if edge["src"] in [i for i, f in enumerate([face_i, face_j])] and \
           edge["dst"] in [i for i, f in enumerate([face_i, face_j])]:
            face_i_edges.add(edge_idx)
            face_j_edges.add(edge_idx)

    shared_edges = face_i_edges & face_j_edges
    if not shared_edges:
        return False

    # Check dihedral angle for shared edges
    for edge_idx in shared_edges:
        dihedral = edges[edge_idx].get("dihedral_angle", 0.0)
        if dihedral < 0.017:  # ~1 degree in radians
            return True

    return False


def _merge_face_pair(face_i: Dict, face_j: Dict) -> Dict:
    """Merge two faces into one.

    Combines areas and averages centroids weighted by area.
    """
    merged = face_i.copy()

    # Combine areas
    area_i = face_i.get("area", 1.0)
    area_j = face_j.get("area", 1.0)
    merged["area"] = area_i + area_j

    # Average centroids weighted by area
    total_area = area_i + area_j
    merged["centroid_x"] = (
        face_i.get("centroid_x", 0.0) * area_i +
        face_j.get("centroid_x", 0.0) * area_j
    ) / total_area
    merged["centroid_y"] = (
        face_i.get("centroid_y", 0.0) * area_i +
        face_j.get("centroid_y", 0.0) * area_j
    ) / total_area
    merged["centroid_z"] = (
        face_i.get("centroid_z", 0.0) * area_i +
        face_j.get("centroid_z", 0.0) * area_j
    ) / total_area

    return merged


def _rebuild_adjacency(
    faces: List[Dict],
    old_edges: List[Dict],
    skip_indices: set,
) -> List[Dict]:
    """Rebuild adjacency after face merges.

    Updates edge indices to account for removed faces.
    """
    # Create mapping from old to new indices
    new_index_map = {}
    new_idx = 0
    for old_idx, face in enumerate(faces):
        if old_idx not in skip_indices:
            new_index_map[old_idx] = new_idx
            new_idx += 1

    # Rewrite adjacency with new indices
    new_edges = []
    for edge in old_edges:
        old_src = edge["src"]
        old_dst = edge["dst"]

        if old_src in skip_indices or old_dst in skip_indices:
            continue

        new_src = new_index_map.get(old_src, old_src)
        new_dst = new_index_map.get(old_dst, old_dst)

        new_edges.append({
            "src": new_src,
            "dst": new_dst,
            "dihedral_angle": edge.get("dihedral_angle", 0.0),
            "convexity": edge.get("convexity", 0.0),
            "edge_length": edge.get("edge_length", 0.0),
        })

    return new_edges

def build_data_object(faces: List[Dict], edges: List[Dict], model_id: str = "unknown") -> Data:
    """Constructs a homogeneous PyG Data graph from OCC faces and edges.

    Args:
        faces: List of node dictionaries. Must contain 'face_id' matching indices in edges.
        edges: List of edge dictionaries. Must contain 'src' and 'dst' indices.
        model_id: String identifier for the STEP file.

    Returns:
        Data object with x, edge_index, edge_attr, and identifier lists.
    """

    # 1. Filter degenerate faces
    valid_faces = []
    valid_mask = []
    for i, face in enumerate(faces):
        area = face.get("area", 0.0)
        if area > 1e-6:  # Filter out degenerate faces
            valid_faces.append(face)
            valid_mask.append(i)

    if len(valid_faces) < len(faces):
        n_removed = len(faces) - len(valid_faces)
        logging.warning(f"Removed {n_removed} degenerate faces (area <= 1e-6)")
        # Update edge indices
        old_to_new = {old: new for new, old in enumerate(valid_mask)}
        for edge in edges:
            edge["src"] = old_to_new.get(edge["src"], edge["src"])
            edge["dst"] = old_to_new.get(edge["dst"], edge["dst"])

    faces = valid_faces

    # 2. Seam merge preprocessing
    faces, edges = merge_seam_faces(faces, edges)

    # 3. Feature vectors
    x_matrix = extract_node_features(faces)
    edge_matrix = extract_edge_features(edges)

    x_tensor = torch.tensor(x_matrix, dtype=torch.float32)
    edge_attr_tensor = torch.tensor(edge_matrix, dtype=torch.float32)

    # Clamp surface type to valid range
    x_tensor[:, 4] = torch.clamp(x_tensor[:, 4], 0.0, 10.0/11.0)

    # 4. Extract topology
    src_list = []
    dst_list = []
    for edge in edges:
        src_list.append(edge["src"])
        dst_list.append(edge["dst"])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 5. Create structural placeholders
    face_ids = [f.get("face_id", i) for i, f in enumerate(faces)]
    occ_face_ids = [str(f.get("occ_face_id", f"#{i}")) for i, f in enumerate(faces)]

    # Empty label array for inference (to be populated during eval if supervision exists)
    y_tensor = torch.full((len(faces),), -1, dtype=torch.long)

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr_tensor,
        y=y_tensor,
        face_ids=face_ids,
        occ_face_ids=occ_face_ids,
        model_id=[model_id] * len(faces)
    )

    return data
