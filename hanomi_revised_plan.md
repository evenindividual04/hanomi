# Hanomi — Revised Implementation Plan
## From Near-Complete Codebase to Assignment-Ready Submission

> **This document picks up where the original plan left off.**
> It does not repeat the original structure — it lists only what changes, why, and how.
> Read alongside the original plan.

---

## Step 0: Preserve the Existing Work (Do This First)

Before touching a single file:

```bash
# In your project root
cp -r hanomi-feature-recognition hanomi-feature-recognition-v1-backup
```

The old codebase is not throwaway. It becomes:
- Your "GNN segmentation only, no contrastive" ablation baseline (Ablation B in the evaluation matrix)
- Your fallback if any new change breaks the pipeline mid-week
- Proof of iteration in the repo history

Do not delete or modify `hanomi-feature-recognition-v1-backup` after this point.

---

## Priority 1: Fix Edge Features Actually Reaching the Model

**The bug:** `SAGEConv` does not accept `edge_attr`. Your forward pass signature takes it but drops it silently. The entire "AAG with geometric edge attributes" claim is currently cosmetic.

**The fix:** Replace `SAGEConv` with `GINEConv`, which is designed for edge-attributed graphs and has minimal additional complexity.

```python
# src/models/encoder.py — REPLACE THIS

# OLD (broken)
from torch_geometric.nn import SAGEConv, BatchNorm
self.convs = nn.ModuleList([
    SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
])

# NEW (correct)
from torch_geometric.nn import GINEConv, BatchNorm
import torch.nn as nn

self.convs = nn.ModuleList([
    GINEConv(
        nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ),
        edge_dim=edge_in_dim,   # 3-dim edge features now actually used
    )
    for _ in range(num_layers)
])
```

Update the forward pass to pass `edge_attr` into each conv:

```python
def forward(self, x, edge_index, edge_attr=None):
    x = self.node_proj(x).relu()
    for conv, norm in zip(self.convs, self.norms):
        x = conv(x, edge_index, edge_attr)   # edge_attr actually consumed now
        x = norm(x)
        x = x.relu()
        x = self.dropout(x)
    return self.out_proj(x)
```

**Why this matters for the walkthrough:** The convexity flag and dihedral angle between adjacent faces are precisely what differentiate a hole boundary (concave transitions) from a random planar junction. If these never reach the model, your "edge-aware AAG" framing is a lie waiting to be called out.

**Alternative if GINEConv causes dependency issues:** Manually concatenate edge features to source node features before each message pass. More verbose but equally correct.

---

## Priority 2: Fix the NT-Xent Implementation

**The bug:** Your current contrastive loss is a 2-logit binary cross-entropy (one positive vs. one hard negative). Real NT-Xent normalizes over all in-batch negatives. Calling it NT-Xent will be caught immediately.

**The fix:** Two options. Pick one and commit.

### Option A — Rename to match what it does (faster, safe)

```python
# src/losses/contrastive.py

class PairwiseContrastiveLoss(nn.Module):
    """
    Simple positive-vs-negative pairwise contrastive loss.
    NOT NT-Xent (which uses in-batch negatives).
    One anchor, one positive, one hard negative per sample.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, anchor, positive, negative):
        anchor   = F.normalize(anchor,   dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        pos_sim  = (anchor * positive).sum(-1) / self.temp
        neg_sim  = (anchor * negative).sum(-1) / self.temp
        logits   = torch.stack([pos_sim, neg_sim], dim=1)
        labels   = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)
```

Update all references in `hybrid.py`, configs, and the write-up.

### Option B — Implement real NT-Xent with in-batch negatives (better signal, ~30 min extra work)

```python
class NTXentLoss(nn.Module):
    """
    Proper NT-Xent: each sample's positive is its augmented pair,
    all other samples in the batch are negatives.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        # z1, z2: [batch, dim] — two views/augmentations of the same subgraph
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        batch = z1.size(0)

        # Concatenate both views: [2*batch, dim]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temp   # [2*batch, 2*batch]

        # Mask out self-similarity
        mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # Labels: for sample i in z1, its positive is sample i+batch in z2
        labels = torch.cat([
            torch.arange(batch, 2*batch, device=z.device),
            torch.arange(0,     batch,   device=z.device),
        ])
        return F.cross_entropy(sim, labels)
```

**For Option B, "two views" of a subgraph means:** the same feature subgraph from two different CAD models of the same type (e.g., two different counterbored holes). Generate these pairs during triplet construction in the dataloader.

**Recommendation:** Start with Option A if you are under time pressure. Option B is better signal but requires dataloader changes. Either is defensible; Option A just needs to be named correctly.

---

## Priority 3: Invert the Candidate Generation Order

**The bug:** Your current inference pipeline runs `O(N)` full GNN passes on all faces first, then uses embedding similarity to find seeds. This is computationally backward. Cheap geometric filtering should happen before expensive neural inference.

**The fix:** Replace `find_feature_instances` with the following pipeline:

```python
# src/inference/seed_expand.py — FULL REWRITE

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus

# Step 1: Heuristic seed filtering (OCC, CPU-only, sub-millisecond)
def get_heuristic_seeds(
    graph: Data,
    reference_surface_types: list[int],   # surface types present in reference feature
) -> list[int]:
    """
    Filter faces to only those matching surface types in the reference.
    For a cylindrical hole: keep only cylinder faces.
    This runs on the graph node features, no GNN needed.
    """
    seed_indices = []
    for i in range(graph.num_nodes):
        face_surface_type = int(graph.x[i, 0].item())
        if face_surface_type in reference_surface_types:
            seed_indices.append(i)
    return seed_indices


# Step 2: k-hop topological expansion (graph traversal, no GNN)
def khop_expand(
    seed_face: int,
    edge_index: torch.LongTensor,
    k: int = 2,
) -> list[int]:
    """BFS expansion k hops from seed face. Returns all face indices in neighborhood."""
    adj = build_adjacency_dict(edge_index)
    visited = {seed_face}
    frontier = {seed_face}
    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return list(visited)


# Step 3: Neural pass only on candidate subgraphs
def find_feature_instances(
    model: FeatureRecognizer,
    reference_graph: Data,
    reference_mask: torch.BoolTensor,
    query_graph: Data,
    reference_surface_types: list[int],
    k_hop: int = 2,
    tau_confidence: float = 0.5,
) -> list[dict]:

    # --- Heuristic stage: no GNN, pure graph traversal ---
    seed_faces = get_heuristic_seeds(query_graph, reference_surface_types)
    if not seed_faces:
        return []

    candidate_subgraphs = []
    for seed in seed_faces:
        subgraph_faces = khop_expand(seed, query_graph.edge_index, k=k_hop)
        candidate_subgraphs.append(subgraph_faces)

    # --- Neural stage: GNN only on candidates ---
    ref_emb = model.embed_subgraph(reference_graph, reference_mask)

    instances = []
    for candidate_faces in candidate_subgraphs:
        mask = torch.zeros(query_graph.num_nodes, dtype=torch.bool)
        mask[candidate_faces] = True

        # Run GNN only on the subgraph, not full query graph
        candidate_emb = model.embed_subgraph(query_graph, mask)
        confidence = F.cosine_similarity(
            candidate_emb.unsqueeze(0), ref_emb.unsqueeze(0)
        ).item()

        if confidence > tau_confidence:
            instances.append({
                "face_ids":     [query_graph.face_ids[i]     for i in candidate_faces],
                "occ_face_ids": [query_graph.occ_face_ids[i] for i in candidate_faces],
                "confidence":   round(confidence, 4),
            })

    instances = non_max_suppression(instances, iou_threshold=0.5)
    return sorted(instances, key=lambda x: -x["confidence"])
```

**On the "run GNN only on the subgraph" point:** You still run the full forward pass (the GNN needs global context from message passing to produce good face embeddings). What changes is that pooling only happens over the candidate subgraph faces. The full graph forward pass is still `O(N)` once — but you only do it once per query model, not once per candidate. Then you do `O(S)` subgraph poolings where S << N.

**Add this to the inference config:**
```yaml
# configs/base.yaml
inference:
  k_hop: 2                     # expansion depth — ablate this on Day 4
  tau_confidence: 0.5
  nms_iou_threshold: 0.5
  reference_surface_types:     # derived automatically from reference feature at runtime
    counterbored_hole: [1]     # CYLINDER = 1
    through_hole: [1]
    pocket: [0]                # PLANE = 0
    slot: [0, 1]
```

---

## Priority 4: Backbone Decision and Consistent Justification

You need to make one concrete choice and defend it consistently everywhere — in code, write-up, and walkthrough prep. Here is the honest framing for both options.

### If you keep GraphSAGE (V1 path, less risk)

The correct justification — do not use the OOM argument:

> "UV-Net's UV-grid sampling pipeline introduces OpenCascade surface parameterization calls that add ~2 hours of data pipeline debugging risk on a 5-day timeline. We use edge-aware face-level AAG (GINEConv) as V1, accepting reduced robustness to CAD kernel split anomalies. UV-Net is the validated V2 upgrade path with identical contrastive architecture on top."

Specifically note the one concrete failure mode you're accepting: a 360° cylinder exported as two 180° half-surfaces by some CAD kernels will be split into two nodes in your graph. UV-Net's absolute 3D coordinate sampling is immune to this; your model is not. State this explicitly in the failure mode table — it shows you understand the tradeoff, which is the signal they want.

### If you switch to UV-Net (stronger, more risk)

Day 1 becomes: clone the Autodesk UV-Net repo, run their preprocessing on 5 MFCAD++ STEP files, verify the DGL graph output. If it works cleanly in under 2 hours, proceed. If not, fall back to GINEConv immediately.

What changes in the code:
```python
# src/parsing/graph_builder.py
# Replace custom feature extraction with UV-Net preprocessing scripts
# Their repo: https://github.com/AutodeskAILab/UV-Net

# Key output difference: node features become [num_faces, 10, 10, 7] UV-grids
# not [num_faces, 9] scalars
# The UV-Net encoder handles this internally via 2D CNN before message passing
```

The contrastive head, pooling, and inference pipeline stay identical. Only the encoder and feature extractor change.

---

## Priority 5: Fix the Walkthrough Prep Answers

These three answers need to be rewritten regardless of which backbone you choose.

### "Why not UV-Net?" (currently broken)

**If you kept GraphSAGE:**
> "UV-Net is the architecturally superior choice and our target for V2. The trade-off we made is engineering risk on the data pipeline: UV-grid generation requires parameterizing B-Rep surfaces via OpenCascade's UV domain, which has known edge cases with degenerate or trimmed surfaces. GINEConv on face-level AAG gets us 80% of the expressiveness with significantly less pipeline risk in 5 days. The concrete cost is reduced robustness to the CAD kernel seam anomaly — a 360° cylinder becoming two 180° half-surfaces."

**If you switched to UV-Net:**
> "We did use UV-Net. We stripped the classification head and use it purely as an encoder for the contrastive prototype network."

### "What is self-supervised path?" (currently vague)

> "BRepMAE: download the ABC dataset (1M unlabeled STEP files), parse to B-Rep graphs, mask 30% of face attributes — surface type, area, radius — and train the encoder to reconstruct them from topological context alone. This is directly analogous to BERT's masked language modeling but on geometric graphs. The benefit: fine-tuning on MFCAD++ reaches equivalent F1 with 5–10× fewer labeled examples. At 50 feature types where some are rare, this is the only scalable path."

### "What happens when you query an unseen feature type?" (not in original plan)

Prepare this — it will be asked:
> "The encoder still runs. The reference subgraph still gets pooled into an embedding. Cosine similarity still fires. The model returns the topologically closest candidates in embedding space. Whether that's correct depends on whether geometric similarity correlates with semantic similarity for the unseen type. It partially works — a never-seen keyway will cluster closer to a slot than to a hole. The self-supervised pretrained encoder (BRepMAE) improves this further by giving the latent space a richer prior over all B-Rep geometry before any task-specific supervision."

---

## Priority 6: Add Confidence Calibration

The assignment explicitly asks for a confidence score per instance. Outputting raw cosine similarity is not calibrated — it's an arbitrary distance, not a probability. Add this to `src/evaluation/metrics.py`:

```python
# src/evaluation/metrics.py — ADD THIS SECTION

from sklearn.calibration import calibration_curve
import numpy as np

def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    """
    Brier score: mean squared error between predicted confidence
    and actual match outcome (1 = correct instance, 0 = false positive).
    Lower is better. Perfect calibration = 0.0.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted average deviation between confidence and accuracy across bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def compute_calibration_metrics(results: list[dict]) -> dict:
    """
    results: list of {confidence: float, is_true_positive: bool}
    Returns Brier score and ECE.
    """
    y_true = [int(r["is_true_positive"]) for r in results]
    y_prob = [r["confidence"] for r in results]
    return {
        "brier_score": brier_score(y_true, y_prob),
        "ece":         expected_calibration_error(np.array(y_true), np.array(y_prob)),
    }
```

Add to `ResultsLogger.save()`:
```python
calibration = compute_calibration_metrics(self.calibration_results)
agg["brier_score"] = calibration["brier_score"]
agg["ece"]         = calibration["ece"]
```

Add to the benchmark comparison table in the README.

---

## Priority 7: Fix the "Non-Negotiable" Language

In the write-up and README, replace every instance of "B-Rep is non-negotiable" or "the only valid representation" with:

> "B-Rep / face adjacency graph is the best-fit representation for this task given two specific constraints: (1) the output requires exact Face ID mapping, which the 1:1 node-to-face correspondence in the graph provides without any post-processing, and (2) CAD feature semantics are topological and surface-type based, which the B-Rep structure preserves exactly. Mesh, point cloud, and multi-view representations lose one or both of these properties."

This is stronger because it's argued, not asserted.

---

## Additional Changes for "Ace" Level

These go beyond fixing bugs and address the specific signals the assignment evaluates.

### A. Handle the CAD Kernel Split Anomaly in Preprocessing

Add this to `src/parsing/graph_builder.py`. It takes 20 minutes and directly addresses a failure mode the deep dive calls out:

```python
def merge_seam_faces(faces: list[dict], adjacency: list[tuple]) -> tuple:
    """
    Detect and merge co-planar or co-cylindrical faces that share a seam edge
    (artifact of some CAD kernels splitting 360° cylinders into two 180° halves).
    Detection: two cylinder faces with identical radius, identical axis direction,
    and a shared edge with dihedral angle < 1 degree.
    """
    merged_faces = []
    skip_indices = set()
    for i, face_i in enumerate(faces):
        if i in skip_indices:
            continue
        for j, face_j in enumerate(faces[i+1:], start=i+1):
            if j in skip_indices:
                continue
            if _is_seam_pair(face_i, face_j, adjacency):
                merged = _merge_face_pair(face_i, face_j)
                merged_faces.append(merged)
                skip_indices.update([i, j])
                break
        else:
            merged_faces.append(face_i)
    return merged_faces
```

Log how many merges happen per model in preprocessing — that number is a useful data point in the failure mode analysis.

### B. k-NN Prototype Voting for Few-Shot Robustness

The deep dive §6.3 mentions that using multiple reference examples (k-NN voting) is more robust than a single 1-shot prototype. Add this to the inference pipeline with minimal effort:

```python
# src/inference/seed_expand.py — ADD

def knn_prototype(
    model: FeatureRecognizer,
    reference_graphs: list[Data],       # multiple reference examples of same feature type
    reference_masks:  list[torch.BoolTensor],
    k: int = 3,
) -> torch.Tensor:
    """
    Compute k-NN prototype by averaging k reference embeddings.
    More robust than single 1-shot prototype.
    Matches the assignment's "single reference" constraint when k=1.
    """
    embeddings = [
        model.embed_subgraph(g, m)
        for g, m in zip(reference_graphs[:k], reference_masks[:k])
    ]
    return torch.stack(embeddings).mean(dim=0)
```

This is essentially free to implement and gives you a richer story: "The system works from a single reference (1-shot) and improves gracefully as more examples are provided. At k=3, embedding space noise averages out and false positives drop."

### C. t-SNE Visualization (Notebook 04)

This is your most powerful visual for the walkthrough and the extensibility demo. It directly shows the embedding space working.

```python
# notebooks/04_extensibility_demo.ipynb

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_embedding_space(embeddings: dict[str, np.ndarray], title: str):
    """
    embeddings: {"counterbored_hole": array[N,64], "through_hole": array[M,64], "background": array[K,64]}
    """
    all_emb = np.vstack(list(embeddings.values()))
    labels  = np.concatenate([
        np.full(len(v), i) for i, v in enumerate(embeddings.values())
    ])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(all_emb)

    colors = ['#e74c3c', '#3498db', '#95a5a6']
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(embeddings.keys(), colors)):
        mask = labels == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=color, label=name, alpha=0.7, s=30)
    ax.legend(); ax.set_title(title)
    plt.tight_layout()
    return fig
```

Run this for Phase 1 (1 type) and Phase 2 (2 types) and put both images side by side in the extensibility section of the write-up. The visual of two well-separated clusters is worth more in a walkthrough than any F1 number.

### D. Augmentation for Robustness

Add these two geometric augmentations to `src/data/transforms.py`. They are cheap and directly address the size/orientation invariance the assignment requires:

```python
class RandomScaleFeatures:
    """Scale area and radius features by random factor. Tests size invariance."""
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, data: Data) -> Data:
        scale = random.uniform(*self.scale_range)
        data = data.clone()
        data.x[:, 1] *= scale   # area
        data.x[:, 6] *= scale   # cylinder radius
        return data


class RandomFlipNormals:
    """Flip face normal direction. Tests orientation invariance."""
    def __call__(self, data: Data) -> Data:
        data = data.clone()
        if random.random() > 0.5:
            data.x[:, 2:5] *= -1   # normal_x, normal_y, normal_z
        return data
```

These augmentations specifically test the invariances stated in the assignment ("same feature regardless of size or orientation"). Mention them in the write-up.

### E. Hard Negative Mining — Make It Concrete

The original plan mentions hard negative mining after epoch 5 but doesn't implement it. Add this to the training loop:

```python
# src/data/dataloader.py — ADD

class HardNegativeTripletSampler:
    """
    After warm-up epochs, mine hard negatives: negatives closest in
    embedding space to the anchor (hardest to distinguish).
    """
    def __init__(self, model, dataset, warm_up_epochs=5):
        self.model = model
        self.dataset = dataset
        self.warm_up_epochs = warm_up_epochs
        self._embedding_cache = {}

    def get_triplet(self, anchor_idx: int, epoch: int) -> tuple:
        anchor_type  = self.dataset[anchor_idx].feature_type
        positive_idx = self._sample_positive(anchor_type, anchor_idx)

        if epoch < self.warm_up_epochs:
            # Random negative during warm-up — prevents collapse from hard negatives too early
            negative_idx = self._sample_random_negative(anchor_type)
        else:
            # Hard negative: closest embedding that is NOT the same feature type
            negative_idx = self._sample_hard_negative(anchor_idx, anchor_type)

        return anchor_idx, positive_idx, negative_idx
```

State in the write-up: "Random negatives for first 5 epochs to avoid early embedding collapse; hard negatives thereafter. This directly addresses the known failure mode of contrastive loss saturation on easy negatives."

### F. Failure Mode Table — Complete It

Fill this in before the walkthrough. Every empty cell is a vulnerability.

| Scenario | Your model's behaviour | Severity | Mitigation |
|---|---|---|---|
| Feature intersects pocket wall | Adjacent face topology breaks; unexpected concave edges appear | High | GNN's receptive field aggregates past the intersection; metric learning still clusters correctly |
| 360° cylinder split into 2×180° (CAD kernel) | Two nodes instead of one; subgraph size doubles | Medium | Seam merge in preprocessing (§A above); UV-Net immune if adopted |
| Through-hole in curved body | Normal features lose meaning; cylinder axis misaligned | Medium | Augment with random normal perturbation during training |
| Symmetric model (4 identical holes) | All 4 detected — correct! But NMS may accidentally merge nearby instances | Low | Tune NMS IoU threshold; log per-model instance count |
| Counterbored hole with added boss | Extra adjacent face breaks rule-based; GNN handles it | Low (for GNN) | Rule-based baseline will fail — document this contrast |
| Very small feature (M2 thread, area → 0) | Area normalization loses discriminative power | Medium | Adaptive normalization per model bounding box diagonal |
| Highly filleted/blended edges | Convexity flag becomes SMOOTH; dihedral angle → π | Medium | Add SMOOTH as valid edge class; don't rely on convexity alone for seeds |
| Degenerate face (area = 0) | Parser exception or zero-division | Low | Skip degenerate faces; log count in `failed_parses.txt` |

### G. Cost Analysis — Make the Numbers Concrete

Add this table to the README and write-up. The assignment explicitly asks for $/query at production scale.

| Method | Inference path | ms/model | GPU cost/query | At 10k models/day |
|---|---|---|---|---|
| Rule-based heuristic | CPU only | < 10ms | $0.000 | $0.00 |
| LLM serialization (Claude Sonnet) | API call, ~800 tokens | 500–2000ms | ~$0.002 | ~$20/day |
| LLM serialization (GPT-4o) | API call, ~800 tokens | 500–2000ms | ~$0.003 | ~$30/day |
| GNN (ours, Colab T4) | GNN forward + subgraph pool | 15–30ms | ~$0.00005 | ~$0.50/day |
| GNN (ours, AWS g4dn.xlarge @ $0.526/hr) | GNN forward + subgraph pool | 15–30ms | ~$0.000004 | ~$0.04/day |

**The punchline:** GNN approach is ~500× cheaper than LLM at production scale. The LLM baseline exists to demonstrate understanding of its trade-offs, not because it is competitive.

---

## Updated Ablation Matrix

Run these five experiments and put results in a table. They directly answer "how do you know each component contributes?"

| Exp | Edge features (GINEConv) | Contrastive loss | k-hop expansion | Expected F1 | What it tests |
|---|---|---|---|---|---|
| A — Full model | ✅ | ✅ | 2-hop | Best | Baseline |
| B — No contrastive | ✅ | ❌ (seg only) | 2-hop | Lower | Value of metric learning |
| C — No edge features | ❌ (node only) | ✅ | 2-hop | Lower | Value of edge attributes |
| D — 1-hop expansion | ✅ | ✅ | 1-hop | Lower | Sufficient context depth |
| E — 3-hop expansion | ✅ | ✅ | 3-hop | A or lower | Context vs noise tradeoff |

Note: Experiment B is essentially your old GraphSAGE codebase. Run it on the old backup to save time.

---

## Write-Up Section Order (Match Assignment Criteria Exactly)

The assignment weights criteria in this order. Mirror it:

1. **Problem framing** — metric learning on B-Rep subgraphs, not classification. The three invariances and how each is addressed.
2. **Representation choice** — B-Rep/AAG argued (not asserted), comparison table with alternatives, exact node/edge feature schema with justification per feature.
3. **Context / token / compute budgeting** — the cost table above, token overflow handling for LLM baseline, GNN inference complexity analysis.
4. **Benchmarking** — benchmark table (3 methods × 2 feature types), ablation table (5 experiments), calibration metrics.
5. **Extensibility demo** — Phase 1 → Phase 2 delta table, "what breaks at 10th/50th type" numbered analysis.
6. **Engineering taste** — clean README, reproducible with one command, tests pass, programmatic results logging.
7. **Bonus** — molecular similarity analogy, BRepMAE self-supervised path, cost analysis, failure mode table.

Keep each section to one page maximum. The assignment says "a thoughtful 60% beats a black-box 90%" — density of insight per word matters more than length.

---

## Remaining Day-by-Day (From Current State)

Assumes you are nearly done with the old plan. This is the delta work only.

### Today / Day N: Apply Priority Fixes

- [ ] Copy codebase: `cp -r hanomi-feature-recognition hanomi-feature-recognition-v1-backup`
- [ ] Replace `SAGEConv` with `GINEConv`, verify edge_attr flows through (Priority 1)
- [ ] Rename or rewrite contrastive loss (Priority 2, Option A is 10 min)
- [ ] Rewrite `find_feature_instances` with heuristic-first order (Priority 3)
- [ ] Decide backbone — commit to GraphSAGE or UV-Net, update walkthrough prep (Priority 4+5)
- [ ] Run existing tests — `pytest tests/` — confirm nothing broke

### Day N+1: Additional Improvements

- [ ] Add seam merge function to preprocessing (§A, ~30 min)
- [ ] Add `RandomScaleFeatures` and `RandomFlipNormals` augmentations (§D, ~20 min)
- [ ] Add `knn_prototype` function to inference (§B, ~20 min)
- [ ] Add `compute_calibration_metrics` to evaluation (Priority 6, ~30 min)

### Day N+2: Run Experiments and Collect Numbers

- [ ] Run full training on counterbored holes (Phase 1)
- [ ] Run ablation experiments B and C (the two most important ones)
- [ ] Run all three baselines on test set
- [ ] Run `scripts/evaluate.py`, confirm per_model_results.json + metrics.json are populated
- [ ] Run old v1 backup as Ablation B (segmentation-only, no contrastive)

### Day N+3: Extensibility + Visualization

- [ ] Train Phase 2 (add through-holes), record delta table
- [ ] Generate t-SNE plots Phase 1 and Phase 2 (§C, ~45 min)
- [ ] Complete failure mode table (§F)
- [ ] Run ablations D and E (k-hop depth)

### Final Day: Write-up + Polish

- [ ] Write-up in section order matching assignment criteria
- [ ] Populate cost analysis table with real numbers from your runs
- [ ] README: one-command setup and run, architecture diagram if time permits
- [ ] Verify `pytest` passes clean
- [ ] Prepare walkthrough talking points — especially the rewritten answers above
- [ ] Final end-to-end run from scratch

---

## Quick Reference: What Changed vs Original Plan

| Component | Original Plan | Revised Plan | Reason |
|---|---|---|---|
| GNN convolution | SAGEConv (drops edge_attr) | GINEConv (uses edge_attr) | Bug fix |
| Contrastive loss | Mislabeled NT-Xent | PairwiseContrastive or real NT-Xent | Correctness |
| Candidate generation | Neural-first, then heuristic | Heuristic-first, then neural | Compute efficiency |
| Backbone justification | Incorrect OOM argument | Honest engineering risk tradeoff | Walkthrough credibility |
| Confidence scoring | Raw cosine similarity | Cosine + Brier/ECE calibration | Assignment requirement |
| B-Rep framing | "Non-negotiable" | "Best fit, argued" | Walkthrough robustness |
| Seam handling | Not present | Seam merge in preprocessing | Failure mode coverage |
| Augmentations | None | Scale + normal flip | Invariance demonstration |
| k-NN prototype | Not present | k reference embeddings averaged | Few-shot robustness |
| Old codebase | Would be deleted | Preserved as Ablation B | Evaluation data point |

---

*Plan version: 2.0 | Revised from v1 implementation | Hanomi ML Engineer Take-Home*
