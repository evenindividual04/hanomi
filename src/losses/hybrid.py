"""Hybrid loss: weighted segmentation + contrastive.

L = w_seg * L_CE + w_contrastive * L_pairwise

The segmentation loss provides dense supervision from per-face MFCAD++ labels.
The contrastive loss forces same-type subgraph embeddings to cluster together
in embedding space regardless of surrounding body geometry.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import PairwiseContrastiveLoss


class HybridLoss(nn.Module):
    """Combined segmentation + contrastive loss.

    Args:
        seg_weight         : weight for cross-entropy segmentation loss
        contrastive_weight : weight for pairwise contrastive loss
        temperature        : contrastive loss temperature τ
        ignore_index       : label index to ignore in cross-entropy (-1 = none)
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        ignore_index: int = -1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.seg_weight         = seg_weight
        self.contrastive_weight = contrastive_weight
        self.ignore_index       = ignore_index
        self.pairwise           = PairwiseContrastiveLoss(temperature)

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_labels: torch.Tensor,
        anchor_emb: torch.Tensor | None = None,
        positive_emb: torch.Tensor | None = None,
        negative_emb: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute hybrid loss.

        Args:
            seg_logits   : [N, num_classes]  face segmentation logits
            seg_labels   : [N]               ground-truth face labels
            anchor_emb   : [B, dim]          anchor subgraph embeddings
            positive_emb : [B, dim]          positive subgraph embeddings
            negative_emb : [B, dim]          negative subgraph embeddings

        Returns:
            total_loss : scalar
            log_dict   : {"seg": float, "contrastive": float}
        """
        # ── Segmentation loss ────────────────────────────────────────────
        seg_loss = F.cross_entropy(seg_logits, seg_labels, ignore_index=self.ignore_index)

        log_dict = {"seg": seg_loss.item()}
        total    = self.seg_weight * seg_loss

        # ── Contrastive loss (optional — requires triplets) ───────────────
        if (anchor_emb is not None
                and positive_emb is not None
                and negative_emb is not None
                and anchor_emb.size(0) > 0):
            c_loss = self.pairwise(anchor_emb, positive_emb, negative_emb)
            log_dict["contrastive"] = c_loss.item()
            total = total + self.contrastive_weight * c_loss
        else:
            log_dict["contrastive"] = 0.0

        log_dict["total"] = total.item()
        return total, log_dict
