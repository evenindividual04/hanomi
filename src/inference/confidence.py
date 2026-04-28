"""Confidence scoring logic for matched feature candidate clusters.

Maps the contrastive subgraph embedding space distance (cosine similarity)
to a normalized confidence score bounded [0, 1].
"""

import torch
import torch.nn.functional as F
from typing import Union

def compute_confidence(query_subgraph_emb: torch.Tensor, ref_subgraph_emb: torch.Tensor) -> float:
    """Computes confidence score from two feature subgraph embeddings.
    
    In the metric learning framework, embedding distance dictates semantic
    feature similarity. We use Cosine Similarity wrapped to a [0,1] confidence.
    
    Args:
        query_subgraph_emb: [dim] pooled tensor for candidate query cluster
        ref_subgraph_emb:   [dim] pooled tensor for anchor reference feature
        
    Returns:
        Confidence float in range [0, 1].
    """
    if query_subgraph_emb.dim() == 1:
        query_subgraph_emb = query_subgraph_emb.unsqueeze(0)
    if ref_subgraph_emb.dim() == 1:
        ref_subgraph_emb = ref_subgraph_emb.unsqueeze(0)
        
    # Raw cosine similarity [-1, 1]
    sim = F.cosine_similarity(query_subgraph_emb, ref_subgraph_emb, dim=-1)
    
    # Scale to [0, 1] — negative correlations are clipped to 0 since they indicate
    # orthogonal or opposite geometric types.
    sim = float(sim.squeeze())
    confidence = max(0.0, sim)
    
    return round(confidence, 4)
