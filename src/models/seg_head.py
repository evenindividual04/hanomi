"""Per-face segmentation head.

Takes per-face embeddings from BRepEncoder and produces 25-class logits
(24 machining feature types + background/stock).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """Two-layer MLP mapping face embeddings → class logits.

    Args:
        in_dim      : embedding dimension (must match BRepEncoder.out_dim)
        num_classes : number of output classes (25 for MFCAD++)
        dropout     : dropout on the hidden layer
    """

    def __init__(
        self,
        in_dim: int = 64,
        num_classes: int = 25,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [num_faces, in_dim]
        Returns:
            logits : [num_faces, num_classes]
        """
        return self.mlp(x)
