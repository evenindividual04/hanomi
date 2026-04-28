"""Neural seed-and-expand subgraph search for inference.

Algorithm (3-stage):
  1. Heuristic + neural seed filtering: surface-type match AND per-face
     cosine similarity >= tau_seed against reference face prototype.
  2. Neural BFS expansion: k-hop BFS where a neighbor is only added if its
     per-face similarity >= tau_expand (prunes stock/unrelated faces early).
  3. Subgraph confidence: pool surviving candidate → cosine similarity against
     reference subgraph embedding >= tau_confidence.

A single GNN forward pass on both the reference and query graphs is run at the
top of find_feature_instances. Stages 1 and 2 reuse the resulting per-face
embeddings — no extra GNN calls.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .nms import non_max_suppression


# ── Adjacency helper ──────────────────────────────────────────────────────────

def _build_adj_dict(edge_index: torch.Tensor, n_edges: int) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for col in range(n_edges):
        src = int(edge_index[0, col].item())
        dst = int(edge_index[1, col].item())
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)
    return adj


# ── Public helpers (also used in tests) ──────────────────────────────────────

def get_heuristic_seeds(
    graph: Data,
    reference_surface_types: List[int],
) -> List[int]:
    """Filter faces by surface type matching reference (CPU, no GNN).

    Surface type is stored pre-divided by 11 in H5 node features (col 4).
    """
    if not reference_surface_types:
        raise ValueError("Empty reference_surface_types")
    ref_set = set(reference_surface_types)
    return [
        i for i in range(graph.num_nodes)
        if int(graph.x[i, 4].item() * 11) in ref_set
    ]


def khop_expand(
    seed: int,
    edge_index: torch.Tensor,
    k: int = 2,
) -> List[int]:
    """Blind BFS expansion k hops from seed (no neural filtering).

    Kept as a public function for ablations and tests. Production inference
    uses _khop_expand_neural which prunes via tau_expand.
    """
    adj = _build_adj_dict(edge_index, edge_index.shape[1])
    visited = {seed}
    frontier = {seed}
    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            for nbr in adj.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


# ── Private neural helpers ────────────────────────────────────────────────────

def _khop_expand_neural(
    seed: int,
    adj: Dict[int, List[int]],
    query_emb_norm: torch.Tensor,
    ref_face_proto: torch.Tensor,
    k: int,
    tau: float,
) -> List[int]:
    """BFS expansion pruned by per-face cosine similarity >= tau.

    Args:
        seed           : starting face index
        adj            : prebuilt adjacency dict
        query_emb_norm : [N, D] L2-normalised query face embeddings
        ref_face_proto : [D]    L2-normalised reference face prototype
        k              : max hops
        tau            : minimum cosine similarity to include a neighbour

    Returns:
        Sorted list of face indices in the pruned k-hop neighbourhood.
    """
    visited = {seed}
    frontier = {seed}
    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            for nbr in adj.get(node, []):
                if nbr not in visited:
                    sim = float(query_emb_norm[nbr] @ ref_face_proto)
                    if sim >= tau:
                        visited.add(nbr)
                        next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


def _ref_embeddings(
    model,
    reference_graph: Data,
    reference_mask: torch.Tensor,
    device: torch.device,
):
    """Return (subgraph_emb [D], face_proto [D]) for a single reference graph.

    subgraph_emb  — attention-pooled embedding of all feature faces (used for
                    the final tau_confidence comparison)
    face_proto    — L2-normalised mean of per-face embeddings (used for
                    per-face tau_seed / tau_expand comparisons)
    """
    ref_graph = reference_graph.to(device)
    mask_t = reference_mask.to(device)
    with torch.no_grad():
        face_emb, _ = model(ref_graph)
    subgraph_emb = model.pooling(face_emb, mask_t)
    face_norm = F.normalize(face_emb[mask_t], dim=1)       # [k, D]
    face_proto = F.normalize(face_norm.mean(0), dim=0)     # [D]
    return subgraph_emb, face_proto


def _knn_embeddings(
    model,
    reference_graphs: List[Data],
    reference_masks: List[torch.Tensor],
    k: int,
    device: torch.device,
):
    """Return (subgraph_proto [D], face_proto [D]) averaged over k references."""
    subgraph_embs: List[torch.Tensor] = []
    face_protos: List[torch.Tensor] = []

    for g, m in zip(reference_graphs, reference_masks):
        if m.sum() == 0:
            continue
        sub_emb, face_proto = _ref_embeddings(model, g, m, device)
        subgraph_embs.append(sub_emb)
        face_protos.append(face_proto)
        if len(subgraph_embs) >= k:
            break

    if not subgraph_embs:
        raise ValueError("No valid reference graphs with non-empty masks")

    subgraph_proto = F.normalize(torch.stack(subgraph_embs).mean(0), dim=0)
    face_proto     = F.normalize(torch.stack(face_protos).mean(0),    dim=0)
    return subgraph_proto, face_proto


# ── Main entry point ──────────────────────────────────────────────────────────

def find_feature_instances(
    model,
    reference_graph: Data | List[Data],
    reference_mask: torch.Tensor | List[torch.Tensor],
    query_graph: Data,
    reference_surface_types: List[int] = None,
    k_hop: int = 2,
    k_neighbors: int = 1,
    tau_seed: float = 0.6,
    tau_expand: float = 0.4,
    tau_confidence: float = 0.5,
    nms_iou_threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Find all instances of a reference feature in a query model.

    Args:
        model                  : Trained FeatureRecognizer
        reference_graph        : PyG Data (or list for k-NN)
        reference_mask         : [num_faces] bool mask for feature faces (or list)
        query_graph            : PyG Data for the query model
        reference_surface_types: Surface type ints to use as seed candidates.
                                 Derived from reference mask if None.
        k_hop                  : BFS expansion depth
        k_neighbors            : Number of references to average for k-NN
        tau_seed               : Minimum per-face cosine similarity to be a seed
        tau_expand             : Minimum per-face cosine similarity to expand into
        tau_confidence         : Minimum subgraph-level cosine similarity to emit
        nms_iou_threshold      : IoU threshold for non-max suppression
        device                 : Torch device (inferred if None)

    Returns:
        List of {face_ids, occ_face_ids, confidence}, sorted by descending confidence.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    query_graph = query_graph.to(device)

    # ── Reference embeddings ──────────────────────────────────────────────────
    if isinstance(reference_graph, list):
        if reference_surface_types is None:
            m0 = reference_mask[0].cpu()
            rx = reference_graph[0].x[m0]
            reference_surface_types = list({int(rx[i, 4].item() * 11) for i in range(rx.shape[0])})
        ref_subgraph_emb, ref_face_proto = _knn_embeddings(
            model, reference_graph, reference_mask, k_neighbors, device
        )
    else:
        if reference_surface_types is None:
            m0 = reference_mask.cpu()
            rx = reference_graph.x[m0]
            reference_surface_types = list({int(rx[i, 4].item() * 11) for i in range(rx.shape[0])})
        ref_subgraph_emb, ref_face_proto = _ref_embeddings(
            model, reference_graph, reference_mask, device
        )

    # ── Query GNN forward (once, upfront) ────────────────────────────────────
    with torch.no_grad():
        query_face_emb, _ = model(query_graph)
    query_emb_norm = F.normalize(query_face_emb, dim=1)    # [N, D]

    # ── Stage 1: heuristic + neural seed filtering ────────────────────────────
    heuristic_seeds = get_heuristic_seeds(query_graph, reference_surface_types)
    if not heuristic_seeds:
        return []

    seeds = [
        i for i in heuristic_seeds
        if float(query_emb_norm[i] @ ref_face_proto) >= tau_seed
    ]
    if not seeds:
        return []

    # ── Stage 2: neural BFS expansion ────────────────────────────────────────
    adj = _build_adj_dict(query_graph.edge_index, query_graph.edge_index.shape[1])
    candidate_subgraphs = [
        _khop_expand_neural(seed, adj, query_emb_norm, ref_face_proto, k_hop, tau_expand)
        for seed in seeds
    ]
    unique_subgraphs = list({frozenset(s): s for s in candidate_subgraphs}.values())

    # ── Stage 3: subgraph pooling + confidence threshold ─────────────────────
    instances = []
    for candidate_faces in unique_subgraphs:
        candidate_emb = model.pooling.pool_indices(query_face_emb, candidate_faces)
        confidence = float(F.cosine_similarity(
            candidate_emb.unsqueeze(0),
            ref_subgraph_emb.unsqueeze(0),
        ))
        if confidence < tau_confidence:
            continue
        instances.append({
            "face_ids": (
                [query_graph.face_ids[i] for i in candidate_faces]
                if hasattr(query_graph, "face_ids") else candidate_faces
            ),
            "occ_face_ids": (
                [query_graph.occ_face_ids[i] for i in candidate_faces]
                if hasattr(query_graph, "occ_face_ids")
                else [f"#{i}" for i in candidate_faces]
            ),
            "confidence": round(confidence, 4),
            "_cluster_set": set(candidate_faces),
        })

    instances = non_max_suppression(instances, iou_threshold=nms_iou_threshold)
    for inst in instances:
        inst.pop("_cluster_set", None)
    instances.sort(key=lambda x: -x["confidence"])
    return instances


# ── Kept for backward compatibility ──────────────────────────────────────────

def knn_prototype(
    model,
    reference_graphs: List[Data],
    reference_masks: List[torch.Tensor],
    k: int = 3,
    device: torch.device = None,
) -> torch.Tensor:
    """Return averaged subgraph prototype over k references.

    Deprecated in favour of _knn_embeddings (which also returns the face
    prototype needed for tau_seed / tau_expand). Kept for external callers.
    """
    if device is None:
        device = next(model.parameters()).device
    subgraph_proto, _ = _knn_embeddings(model, reference_graphs, reference_masks, k, device)
    return subgraph_proto
