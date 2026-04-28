"""Tests for inference algorithms (seed-and-expand, nms)."""

import pytest
import torch
from torch_geometric.data import Data
import torch.nn as nn

from src.models.feature_recognizer import FeatureRecognizer
from src.inference.seed_expand import find_feature_instances
from src.inference.confidence import compute_confidence
from src.inference.nms import non_max_suppression

@pytest.fixture
def mock_model():
    """Returns an uninitialized FeatureRecognizer for dimension testing."""
    import yaml
    cfg_raw = """
    feature_types: [counterbored_hole]
    encoder:
      node_in_dim: 9
      edge_in_dim: 3
      hidden_dim: 16
      out_dim: 16
      num_layers: 2
      dropout: 0.0
    seg_head:
      in_dim: 16
      num_classes: 25
    pooling:
      dim: 16
      mode: mean
    """
    cfg = yaml.safe_load(cfg_raw)
    
    # FeatureRecognizer supports dict configs natively
    model = FeatureRecognizer(cfg)
    model.eval()
    return model

def test_nms_suppression():
    """Verify IoU overlap successfully removes lower confidence predictions."""
    predictions = [
        {"face_ids": [1, 2, 3], "confidence": 0.9},
        {"face_ids": [2, 3, 4], "confidence": 0.7},  # 2/4 IoU = 0.5 (should be removed if threshold=0.49)
        {"face_ids": [10, 11], "confidence": 0.6}    # No overlap (kept)
    ]
    
    result = non_max_suppression(predictions, iou_threshold=0.49)
    assert len(result) == 2
    assert result[0]["confidence"] == 0.9
    assert result[1]["confidence"] == 0.6

def test_compute_confidence():
    """Verify bounds of cosine similarity mapping."""
    ref_emb = torch.tensor([1.0, 0.0, 0.0])
    
    # Exact match
    conf_exact = compute_confidence(torch.tensor([1.0, 0.0, 0.0]), ref_emb)
    assert conf_exact == 1.0
    
    # Orthogonal
    conf_ortho = compute_confidence(torch.tensor([0.0, 1.0, 0.0]), ref_emb)
    assert conf_ortho == 0.0
    
    # Opposite (should clip to 0)
    conf_opposite = compute_confidence(torch.tensor([-1.0, 0.0, 0.0]), ref_emb)
    assert conf_opposite == 0.0

def test_find_feature_instances(mock_model):
    """Integration test wrapper over the find_feature_instances pipeline."""
    # Build 3-node reference graph
    ref_x = torch.randn((3, 9))
    ref_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    ref_mask = torch.tensor([True, True, True])
    ref_data = Data(x=ref_x, edge_index=ref_edge_index, edge_attr=torch.randn((4, 3)))
    
    # Build 6-node query (nodes 0,1,2 duplicate reference)
    query_x = torch.cat([ref_x, torch.randn((3, 9))])
    query_edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    query_data = Data(x=query_x, edge_index=query_edge_index, edge_attr=torch.randn((8, 3)))
    
    # We are using an untrained model with random weights, so we lower the thresholds
    # immensely just to execute the topological traversal paths.
    instances = find_feature_instances(
        model=mock_model,
        reference_graph=ref_data,
        reference_mask=ref_mask,
        query_graph=query_data,
        tau_seed=-1.0,         # bypass embedding checks
        tau_expand=-1.0,       # allow full bfs
        tau_confidence=-1.0,
        nms_iou_threshold=0.5
    )
    
    assert type(instances) == list
    if len(instances) > 0:
        assert "face_ids" in instances[0]
        assert "confidence" in instances[0]
