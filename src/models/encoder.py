"""GINEConv encoder for B-Rep face graphs.

Input : per-face feature vectors + adjacency with edge attributes
Output: per-face embeddings in a latent space
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm


class BRepEncoder(nn.Module):
    """Inductive GNN encoder over B-Rep attributed adjacency graphs.

    Uses GINEConv (Graph Isomorphism Network with Edge features) because:
      - Uses edge features (convexity, dihedral_angle, edge_length) critical for feature boundaries
      - Inductive: generalises to new STEP files at inference without retraining
      - Universal approximation for graph functions
      - Proven on molecular GNNs (similar topological patterns)

    Args:
        node_in_dim : input node feature dimension (8 for H5 path, 9 for STEP)
        edge_in_dim : input edge feature dimension (3 for dihedral, convexity, length)
        hidden_dim  : hidden width in each GINE layer
        out_dim     : output embedding dimension per face
        num_layers  : number of GINEConv layers (≥2)
        dropout     : dropout probability applied after each layer
    """

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.edge_in_dim = edge_in_dim
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=edge_in_dim,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList(
            [BatchNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout  = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x          : [num_faces, node_in_dim]  node features
            edge_index : [2, num_edges]             adjacency
            edge_attr  : [num_edges, edge_in_dim]  edge attributes (now used by GINEConv)

        Returns:
            face_emb : [num_faces, out_dim]
        """
        x = self.node_proj(x).relu()
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = x.relu()
            x = self.dropout(x)
        return self.out_proj(x)   # [num_faces, out_dim]
