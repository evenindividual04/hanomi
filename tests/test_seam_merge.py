"""Tests for seam merge functionality."""

import pytest
import torch
import numpy as np
from src.parsing.graph_builder import merge_seam_faces, _is_seam_pair, _merge_face_pair


def test_seam_pair_detection():
    """Test that seam pairs are correctly detected."""
    # Create two cylinder faces that should be merged
    face_i = {
        "surface_type": "CYLINDER",
        "cylinder_radius": 5.0,
        "cylinder_axis": [0, 0, 1],
        "area": 10.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "centroid_z": 0.0,
    }
    face_j = {
        "surface_type": "CYLINDER",
        "cylinder_radius": 5.0,
        "cylinder_axis": [0, 0, 1],
        "area": 10.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "centroid_z": 0.0,
    }
    edges = [
        {"src": 0, "dst": 1, "dihedral_angle": 0.005, "convexity": 0.0, "edge_length": 1.0}
    ]

    assert _is_seam_pair(face_i, face_j, edges) == True


def test_seam_pair_different_radius():
    """Test that faces with different radii are not merged."""
    face_i = {
        "surface_type": "CYLINDER",
        "cylinder_radius": 5.0,
        "cylinder_axis": [0, 0, 1],
        "area": 10.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "centroid_z": 0.0,
    }
    face_j = {
        "surface_type": "CYLINDER",
        "cylinder_radius": 10.0,  # Different radius
        "cylinder_axis": [0, 0, 1],
        "area": 10.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "centroid_z": 0.0,
    }
    edges = [
        {"src": 0, "dst": 1, "dihedral_angle": 0.005, "convexity": 0.0, "edge_length": 1.0}
    ]

    assert _is_seam_pair(face_i, face_j, edges) == False


def test_seam_merge_increases_area():
    """Test that merging increases area correctly."""
    faces = [
        {
            "surface_type": "CYLINDER",
            "area": 10.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_z": 0.0,
            "cylinder_radius": 5.0,
            "cylinder_axis": [0, 0, 1],
        },
        {
            "surface_type": "CYLINDER",
            "area": 10.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_z": 0.0,
            "cylinder_radius": 5.0,
            "cylinder_axis": [0, 0, 1],
        },
    ]
    edges = [
        {"src": 0, "dst": 1, "dihedral_angle": 0.005, "convexity": 0.0, "edge_length": 1.0}
    ]

    merged_faces, merged_edges = merge_seam_faces(faces, edges)

    assert len(merged_faces) == 1
    assert merged_faces[0]["area"] == 20.0


def test_non_cylinder_not_merged():
    """Test that non-cylinder faces are not merged."""
    faces = [
        {
            "surface_type": "PLANE",  # Not a cylinder
            "area": 10.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_z": 0.0,
            "normal_x": 0.0,
            "normal_y": 0.0,
            "normal_z": 1.0,
        },
        {
            "surface_type": "PLANE",  # Not a cylinder
            "area": 10.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_z": 0.0,
            "normal_x": 0.0,
            "normal_y": 0.0,
            "normal_z": 1.0,
        },
    ]
    edges = [
        {"src": 0, "dst": 1, "dihedral_angle": 0.005, "convexity": 0.0, "edge_length": 1.0}
    ]

    merged_faces, merged_edges = merge_seam_faces(faces, edges)

    # Should not merge non-cylinders
    assert len(merged_faces) == 2
