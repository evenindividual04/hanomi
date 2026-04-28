"""Pairwise contrastive loss for metric learning.

Used to push subgraph embeddings of the same feature type together and
push embeddings of different types apart, regardless of model context.

Note: This is NOT true NT-Xent (which uses in-batch negatives). This is
a simple positive-vs-negative pairwise contrastive loss with explicit triplets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseContrastiveLoss(nn.Module):
    """Simple positive-vs-negative pairwise contrastive loss.

    For a batch of triplets:
      - anchor:   subgraph embedding of feature type A in model X
      - positive: subgraph embedding of feature type A in model Y (≠ X)
      - negative: subgraph embedding of a different feature type in any model

    Args:
        temperature : softmax temperature τ (lower → sharper separation)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temp = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor   : [B, dim]  anchor subgraph embeddings
            positive : [B, dim]  positive (same type, different model)
            negative : [B, dim]  negative (different type)

        Returns:
            loss : scalar
        """
        anchor   = F.normalize(anchor,   dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_sim = (anchor * positive).sum(-1) / self.temp   # [B]
        neg_sim = (anchor * negative).sum(-1) / self.temp   # [B]

        logits = torch.stack([pos_sim, neg_sim], dim=1)     # [B, 2]
        labels = torch.zeros(
            anchor.size(0), dtype=torch.long, device=anchor.device
        )
        return F.cross_entropy(logits, labels)
