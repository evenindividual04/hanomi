"""Translates pythonocc BRep faces/edges into dimensional feature vectors.

Maps raw OCC geometric attributes to 9-dim node features and 3-dim edge features
for the GraphSAGE encoder.
"""

from typing import Dict, List
import copy
import numpy as np

SURFACE_TYPE_MAP = {
    "PLANE": 0,
    "CYLINDER": 1,
    "CONE": 2,
    "SPHERE": 3,
    "TORUS": 4,
    "OTHER": 5,
}

def extract_node_features(faces: List[Dict]) -> List[List[float]]:
    """Convert raw face dictionaries to 9-dimensional PyG node features.

    Schema:
    [
      surface_type (categorical -> float),
      area,
      normal_x, normal_y, normal_z,
      cylinder_radius, cylinder_axis_z,
      num_adjacent_faces, num_boundary_edges
    ]
    """
    node_features = []
    
    # Calculate bounding box diagonal for normalization if coords were passed
    # For now, we assume 'area' and 'cylinder_radius' are pre-normalized by the STEP parser.
    for f in faces:
        stype_str = f.get("surface_type", "OTHER")
        stype_idx = SURFACE_TYPE_MAP.get(stype_str, 5)
        
        vec = [
            float(stype_idx),
            float(f.get("area", 0.0)),
            float(f.get("normal_x", 0.0)),
            float(f.get("normal_y", 0.0)),
            float(f.get("normal_z", 0.0)),
            float(f.get("cylinder_radius", 0.0)),
            float(f.get("cylinder_axis_z", 0.0)),
            float(f.get("num_adjacent_faces", 0)),
            float(f.get("num_boundary_edges", 0))
        ]
        node_features.append(vec)
        
    return node_features


def extract_edge_features(edges: List[Dict]) -> List[List[float]]:
    """Convert raw adjacency dictionaries to 3-dimensional PyG edge features.

    Schema:
    [
      dihedral_angle (radians),
      convexity (1=convex, -1=concave, 0=smooth),
      edge_length (normalized)
    ]
    """
    edge_features = []
    for e in edges:
        vec = [
            float(e.get("dihedral_angle", 0.0)),
            float(e.get("convexity", 0.0)),
            float(e.get("edge_length", 0.0))
        ]
        edge_features.append(vec)
        
    return edge_features
