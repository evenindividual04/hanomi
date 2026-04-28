"""Tests for model forward pass — no H5 data required, uses synthetic graphs."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, Batch


def _make_synthetic_graph(n_faces: int = 12, n_edges: int = 20) -> Data:
    """Create a small synthetic B-Rep graph for unit testing."""
    x          = torch.randn(n_faces, 8)
    src        = torch.randint(0, n_faces, (n_edges,))
    dst        = torch.randint(0, n_faces, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr  = torch.randn(n_edges, 3)
    y          = torch.randint(0, 25, (n_faces,))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=n_faces)


def _make_config() -> dict:
    return {
        "encoder":  {"node_in_dim": 8, "hidden_dim": 32, "out_dim": 16, "num_layers": 2, "dropout": 0.0},
        "seg_head": {"in_dim": 16, "num_classes": 25},
        "pooling":  {"dim": 16, "mode": "attention"},
    }


def test_encoder_forward():
    from src.models.encoder import BRepEncoder
    cfg = _make_config()["encoder"]
    enc = BRepEncoder(**cfg)
    g   = _make_synthetic_graph()
    out = enc(g.x, g.edge_index, g.edge_attr)
    assert out.shape == (g.num_nodes, cfg["out_dim"]), f"Wrong shape: {out.shape}"


def test_seg_head_forward():
    from src.models.seg_head import SegmentationHead
    head = SegmentationHead(in_dim=16, num_classes=25)
    emb  = torch.randn(12, 16)
    logits = head(emb)
    assert logits.shape == (12, 25)


def test_pooling_mean():
    from src.models.pooling import SubgraphPooling
    pool = SubgraphPooling(dim=16, mode="mean")
    emb  = torch.randn(12, 16)
    mask = torch.zeros(12, dtype=torch.bool)
    mask[:4] = True
    out  = pool(emb, mask)
    assert out.shape == (16,)
    assert torch.allclose(out, emb[:4].mean(0))


def test_pooling_attention():
    from src.models.pooling import SubgraphPooling
    pool = SubgraphPooling(dim=16, mode="attention")
    emb  = torch.randn(12, 16)
    mask = torch.zeros(12, dtype=torch.bool)
    mask[2:6] = True
    out  = pool(emb, mask)
    assert out.shape == (16,)


def test_feature_recognizer_forward():
    from src.models.feature_recognizer import FeatureRecognizer
    cfg   = _make_config()
    model = FeatureRecognizer(cfg)
    g     = _make_synthetic_graph()
    face_emb, seg_logits = model(g)
    assert face_emb.shape   == (g.num_nodes, 16)
    assert seg_logits.shape == (g.num_nodes, 25)


def test_feature_recognizer_embed_subgraph():
    from src.models.feature_recognizer import FeatureRecognizer
    cfg   = _make_config()
    model = FeatureRecognizer(cfg)
    g     = _make_synthetic_graph(n_faces=12)
    mask  = torch.zeros(12, dtype=torch.bool)
    mask[:3] = True
    emb   = model.embed_subgraph(g, mask)
    assert emb.shape == (16,)


def test_hybrid_loss():
    from src.losses.hybrid import HybridLoss
    loss_fn    = HybridLoss(seg_weight=1.0, contrastive_weight=0.5, temperature=0.07)
    seg_logits = torch.randn(10, 25)
    seg_labels = torch.randint(0, 25, (10,))
    anc  = torch.randn(4, 16)
    pos  = torch.randn(4, 16)
    neg  = torch.randn(4, 16)
    loss, log = loss_fn(seg_logits, seg_labels, anc, pos, neg)
    assert loss.item() > 0
    assert "seg" in log and "contrastive" in log


def test_nms():
    from src.inference.nms import non_max_suppression
    instances = [
        {"face_ids": [1, 2, 3], "confidence": 0.9, "_cluster_set": {1, 2, 3}},
        {"face_ids": [1, 2, 4], "confidence": 0.7, "_cluster_set": {1, 2, 4}},   # IoU = 2/4 = 0.5
        {"face_ids": [7, 8, 9], "confidence": 0.8, "_cluster_set": {7, 8, 9}},   # no overlap
    ]
    # Use threshold < 0.5 to ensure the 0.5 IoU overlap gets suppressed
    kept = non_max_suppression(instances, iou_threshold=0.49)
    assert len(kept) == 2, f"Expected 2 after NMS, got {len(kept)}"
    assert kept[0]["confidence"] == 0.9
