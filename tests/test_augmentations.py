"""Tests for geometric augmentations."""

import pytest
import torch
from torch_geometric.data import Data
from src.data.transforms import (
    RandomScaleFeatures, RandomFlipNormals, RandomJitterPosition, Compose,
    RandomFeatureMask, RandomEdgeDrop,
)


def test_random_scale_features():
    """Test that scaling affects area (dim 0) and cylinder radius (dim 6) for 9-dim STEP schema."""
    # 9-dim STEP schema: RandomScaleFeatures scales dim 0 (area) and dim 6 (radius, when shape[1]>6)
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]]),
        edge_index=torch.tensor([[0], [0]]),
    )

    transform = RandomScaleFeatures(scale_range=(2.0, 2.0), p=1.0)  # Always scale by 2.0
    augmented = transform(data)

    assert augmented.x[0, 0] == 2.0   # area (dim 0) scaled
    assert augmented.x[0, 6] == 10.0  # radius (dim 6) scaled


def test_random_scale_features_probability():
    """Test that scaling respects probability."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]]),
        edge_index=torch.tensor([[0], [0]]),
    )

    # With p=0.0, should never scale
    transform = RandomScaleFeatures(scale_range=(2.0, 2.0), p=0.0)
    augmented = transform(data)

    assert augmented.x[0, 0] == 1.0  # area unchanged


def test_random_flip_normals():
    """Test that normal direction flips."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0]]),  # normal = [1, 0, 0]
        edge_index=torch.tensor([[0], [0]]),
    )

    transform = RandomFlipNormals(p=1.0)  # Always flip
    augmented = transform(data)

    assert augmented.x[0, 2] == -1.0  # normal_x flipped


def test_random_jitter_position():
    """Test that position is jittered."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]]),  # centroid = [0, 0, 0]
        edge_index=torch.tensor([[0], [0]]),
    )

    transform = RandomJitterPosition(jitter_std=0.1, p=1.0)  # Always jitter
    augmented = transform(data)

    # Position should have changed (very unlikely to stay exactly the same)
    assert not torch.allclose(augmented.x[0, 1:4], torch.tensor([0.0, 0.0, 0.0]))


def test_compose():
    """Test that compose applies multiple transforms."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0]]),
        edge_index=torch.tensor([[0], [0]]),
    )

    transform = Compose([
        RandomScaleFeatures(scale_range=(2.0, 2.0), p=1.0),
        RandomFlipNormals(p=1.0),
    ])
    augmented = transform(data)

    assert augmented.x[0, 0] == 2.0  # area scaled
    assert augmented.x[0, 2] == -1.0  # normal flipped


def test_transform_leaves_other_data_unchanged():
    """Test that transforms don't modify edge_index or other attributes."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]]),
        edge_index=torch.tensor([[0], [0]]),
        y=torch.tensor([1]),
    )

    transform = RandomScaleFeatures(scale_range=(2.0, 2.0), p=1.0)
    augmented = transform(data)

    assert torch.equal(augmented.edge_index, data.edge_index)
    assert torch.equal(augmented.y, data.y)


def test_random_feature_mask_zeros_columns():
    """Test that RandomFeatureMask zeros out a subset of feature columns."""
    data = Data(
        x=torch.ones(4, 8),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    transform = RandomFeatureMask(mask_ratio=0.5, p=1.0)
    augmented = transform(data)

    # Original data must not be mutated
    assert data.x.sum() == 32.0
    # At least one column should be zeroed (mask_ratio=0.5 → 4 cols out of 8)
    assert augmented.x.sum() < 32.0
    # Masked positions should be exactly 0.0
    zero_cols = (augmented.x == 0.0).all(dim=0)
    assert zero_cols.any()


def test_random_feature_mask_respects_probability():
    """Test that RandomFeatureMask respects p=0.0 (never mask)."""
    data = Data(
        x=torch.ones(4, 8),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    transform = RandomFeatureMask(mask_ratio=0.5, p=0.0)
    augmented = transform(data)

    assert augmented.x.sum() == 32.0  # nothing masked


def test_random_edge_drop_reduces_edges():
    """Test that RandomEdgeDrop removes edges and matching edge_attr rows."""
    n_edges = 20
    edge_index = torch.randint(0, 5, (2, n_edges))
    edge_attr = torch.ones(n_edges, 3)
    data = Data(x=torch.ones(5, 8), edge_index=edge_index, edge_attr=edge_attr)

    transform = RandomEdgeDrop(drop_ratio=0.5, p=1.0)
    augmented = transform(data)

    # Some edges should have been dropped
    assert augmented.edge_index.size(1) < n_edges
    # edge_attr rows must match edge_index columns
    assert augmented.edge_attr.size(0) == augmented.edge_index.size(1)


def test_random_edge_drop_no_edge_attr():
    """Test that RandomEdgeDrop works when edge_attr is None."""
    data = Data(
        x=torch.ones(4, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        edge_attr=None,
    )
    transform = RandomEdgeDrop(drop_ratio=0.5, p=1.0)
    augmented = transform(data)

    assert augmented.edge_index.size(1) <= 4
    assert augmented.edge_attr is None


def test_random_edge_drop_respects_probability():
    """Test that RandomEdgeDrop respects p=0.0 (never drop)."""
    data = Data(
        x=torch.ones(4, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
    )
    transform = RandomEdgeDrop(drop_ratio=0.5, p=0.0)
    augmented = transform(data)

    assert augmented.edge_index.size(1) == 4
