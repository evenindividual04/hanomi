"""Subgraph pooling: face embeddings + mask → single subgraph embedding.

Two modes:
  mean      — simple mean of face embeddings in the subgraph (fast)
  attention — learned attention weights over the subgraph faces (better)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphPooling(nn.Module):
    """Aggregate per-face embeddings of a subgraph into one vector.

    Args:
        dim  : embedding dimension (must match BRepEncoder.out_dim)
        mode : ``"mean"`` or ``"attention"``
    """

    def __init__(self, dim: int = 64, mode: str = "attention", **kwargs) -> None:
        super().__init__()
        if mode not in ("mean", "attention"):
            raise ValueError(f"mode must be 'mean' or 'attention', got '{mode}'")
        self.mode = mode
        if mode == "attention":
            self.attn = nn.Linear(dim, 1)

    def forward(
        self,
        face_embeddings: torch.Tensor,
        subgraph_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool a subgraph defined by a boolean mask.

        Args:
            face_embeddings : [num_faces, dim]  per-face embeddings
            subgraph_mask   : [num_faces]        True for faces in subgraph

        Returns:
            embedding : [dim]
        """
        sub = face_embeddings[subgraph_mask]   # [k, dim]
        if sub.size(0) == 0:
            # Empty subgraph — return zero vector
            return torch.zeros(face_embeddings.size(1), device=face_embeddings.device)
        if self.mode == "mean":
            return sub.mean(dim=0)
        else:
            weights = F.softmax(self.attn(sub), dim=0)   # [k, 1]
            return (weights * sub).sum(dim=0)              # [dim]

    def pool_indices(
        self,
        face_embeddings: torch.Tensor,
        face_indices: list[int],
    ) -> torch.Tensor:
        """Pool using a list of face indices rather than a boolean mask.

        Convenience wrapper used in inference.
        """
        n = face_embeddings.size(0)
        mask = torch.zeros(n, dtype=torch.bool, device=face_embeddings.device)
        if face_indices:
            idx = torch.tensor(face_indices, dtype=torch.long, device=face_embeddings.device)
            mask[idx] = True
        return self.forward(face_embeddings, mask)
