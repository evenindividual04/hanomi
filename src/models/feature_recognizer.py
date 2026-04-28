"""Full feature recognition model.

Combines:
  BRepEncoder      → per-face embeddings
  SegmentationHead → 25-class face logits
  SubgraphPooling  → subgraph-level embedding for metric learning
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .encoder import BRepEncoder
from .seg_head import SegmentationHead
from .pooling import SubgraphPooling


class FeatureRecognizer(nn.Module):
    """End-to-end B-Rep feature recognition model.

    Args:
        config : object with sub-configs ``encoder``, ``seg_head``, ``pooling``
                 (each is a dict or namespace passed as **kwargs).
    """

    def __init__(self, config) -> None:
        super().__init__()
        enc_cfg  = config.get("encoder",  {}) if isinstance(config, dict) else vars(config.encoder)
        seg_cfg  = config.get("seg_head", {}) if isinstance(config, dict) else vars(config.seg_head)
        pool_cfg = config.get("pooling",  {}) if isinstance(config, dict) else vars(config.pooling)

        self.encoder  = BRepEncoder(**enc_cfg)
        self.seg_head = SegmentationHead(**seg_cfg)
        self.pooling  = SubgraphPooling(**pool_cfg)

        out_dim = enc_cfg.get("out_dim", 64)
        # SimCLR projection head: absorbs contrastive uniformity cost during training.
        # Representations before this head are better for downstream retrieval —
        # use_proj=False at inference keeps existing checkpoints compatible.
        self.proj_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, data: Data):
        """Run encoder + segmentation head over a batched graph.

        Returns:
            face_emb   : [total_faces, out_dim]   per-face embeddings
            seg_logits : [total_faces, num_classes] segmentation logits
        """
        face_emb   = self.encoder(data.x, data.edge_index, data.edge_attr)
        seg_logits = self.seg_head(face_emb)
        return face_emb, seg_logits

    def embed_subgraph(
        self,
        data: Data,
        subgraph_mask: torch.Tensor,
        use_proj: bool = False,
    ) -> torch.Tensor:
        """Embed a subgraph defined by a boolean face mask.

        Args:
            data           : single-graph PyG Data (not batched)
            subgraph_mask  : [num_faces] bool, True for faces in the subgraph
            use_proj       : if True, pass through SimCLR projection head
                             (only during contrastive training; False at inference)

        Returns:
            embedding : [out_dim]
        """
        face_emb, _ = self.forward(data)
        emb = self.pooling(face_emb, subgraph_mask)
        return self.proj_head(emb) if use_proj else emb

    def embed_subgraph_indices(
        self,
        data: Data,
        face_indices: list[int],
    ) -> torch.Tensor:
        """Embed a subgraph defined by a list of face indices."""
        face_emb, _ = self.forward(data)
        return self.pooling.pool_indices(face_emb, face_indices)
