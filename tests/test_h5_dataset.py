"""Tests for the MFCAD++ H5 dataset reader.

Run:  pytest tests/test_h5_dataset.py -v

These tests validate:
  - H5 file opens correctly
  - At least one batch group is parseable
  - Resulting Data objects have correct shapes
  - Feature-label filtering works
"""

from __future__ import annotations

import pytest
from pathlib import Path

# Path to H5 file (relative to repo root)
VAL_H5 = Path("MFCAD++_dataset/hierarchical_graphs/val_MFCAD++.h5")

# Skip all tests if H5 file not present
skip_if_no_h5 = pytest.mark.skipif(
    not VAL_H5.exists(),
    reason=f"H5 dataset not found at {VAL_H5}",
)


@skip_if_no_h5
def test_h5_file_opens():
    import h5py
    with h5py.File(VAL_H5, "r") as f:
        assert len(f.keys()) > 0, "H5 file is empty"


@skip_if_no_h5
def test_dataset_loads_graphs():
    from src.data.h5_dataset import MFCADPlusPlusDataset
    ds = MFCADPlusPlusDataset(h5_path=VAL_H5, feature_labels=None, cache=True)
    assert len(ds) > 0, "Dataset is empty"


@skip_if_no_h5
def test_graph_shapes():
    from src.data.h5_dataset import MFCADPlusPlusDataset
    ds = MFCADPlusPlusDataset(h5_path=VAL_H5, feature_labels=None, cache=True)
    g  = ds[0]
    assert g.x.ndim == 2,        "x should be 2D [num_faces, features]"
    assert g.x.shape[1] == 8,   "H5 path: 8-dim node features"
    assert g.edge_index.shape[0] == 2, "edge_index should be [2, num_edges]"
    assert g.edge_attr.shape[1] == 3,  "3-dim edge attributes"
    assert g.y.ndim == 1,        "y should be 1D [num_faces]"
    assert g.x.shape[0] == g.y.shape[0], "Nodes in x and y must match"


@skip_if_no_h5
def test_feature_label_filter():
    from src.data.h5_dataset import MFCADPlusPlusDataset, LABEL_IDS
    # Filter to through-hole only (label 1)
    ds_all     = MFCADPlusPlusDataset(h5_path=VAL_H5, feature_labels=None)
    ds_through = MFCADPlusPlusDataset(h5_path=VAL_H5, feature_labels=[1])
    assert len(ds_through) <= len(ds_all), "Filtered dataset must be smaller"
    assert len(ds_through) > 0, "Expected some through-hole models"
    # Every graph in filtered set must contain label 1
    for g in ds_through:
        assert 1 in g.y.unique().tolist(), "Filtered graph missing target label"


@skip_if_no_h5
def test_label_names_for():
    from src.data.h5_dataset import MFCADPlusPlusDataset
    ids = MFCADPlusPlusDataset.label_names_for(["through_hole", "blind_hole"])
    assert ids == [1, 12], f"Expected [1, 12], got {ids}"


@skip_if_no_h5
def test_edge_convexity_range():
    from src.data.h5_dataset import MFCADPlusPlusDataset
    ds = MFCADPlusPlusDataset(h5_path=VAL_H5, feature_labels=[1])
    g  = ds[0]
    # convexity column 0: values in {-1, 0, 1}
    convexities = g.edge_attr[:, 0].unique().tolist()
    for v in convexities:
        assert v in (-1.0, 0.0, 1.0), f"Unexpected convexity value {v}"
