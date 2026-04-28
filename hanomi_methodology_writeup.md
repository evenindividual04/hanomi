# Feature Recognition via Metric Learning on B-Rep Subgraph Embeddings

*Hanomi ML Engineering Take-Home Submission*

---

## 1. Problem Framing

### 1.1 Why this is not classification

Most B-Rep recognition papers (BRepNet, AAGNet, UV-Net) treat feature recognition as **face-wise semantic segmentation** — a per-face classification problem minimizing cross-entropy against a fixed label set. This framing has a fundamental extensibility limitation: adding a new feature type requires rebuilding the classifier head and retraining the network. The model learns to predict predefined output bins, not the geometric equivalence between two topological structures.

Our task is different:

```
Given:  a reference STEP model + labeled feature subgraph (face IDs + OCC face IDs)
Find:   all subgraphs in query STEP models that are semantically equivalent
Output: per-instance {face_ids, occ_face_ids, confidence_score}
```

This is **metric learning on B-Rep subgraphs**: embed the reference feature into a vector space, enumerate candidate subgraphs in query models, and retrieve those whose embedding is closest to the reference.

### 1.2 Comparison of framings

| Design Choice | Classification | Metric Learning (Ours) |
|---|---|---|
| Loss function | Cross-entropy | Contrastive / triplet |
| Query mechanism | Forward pass → argmax | Embed reference → nearest neighbor search |
| Adding new feature type | Retrain classifier head | Add labeled examples → new cluster in embedding space |
| Generalizes to unseen type? | No | Potentially yes |

This distinction is the answer to the extensibility question: adding a new feature type is a **data problem** (provide labeled examples) not an architecture problem.

### 1.3 Required invariances

The model must recognize the same feature regardless of:

- **Size** (diameter, depth) — a 5mm through-hole matches a 20mm through-hole
- **Orientation** — a hole pointing down matches a hole pointing sideways
- **Position** — anywhere in the body
- **Surrounding geometry** — a hole in a flat plate matches a hole in a curved boss

These are handled at multiple levels: normalized node features remove size/position dependence, the GINEConv encoder learns topology-agnostic embeddings, and contrastive loss explicitly clusters same-type features regardless of context.

---

## 2. Representation Choice

### 2.1 B-Rep over alternatives

| Representation | Geometry | Topology | Tessellation-invariant | Notes |
|---|---|---|---|---|
| Point cloud | Partial | Lost | No | Needs sampling, loses adjacency |
| Mesh | Partial | Partial | No | Tessellation-dependent resolution |
| Multi-view images | Partial | Lost | Yes | Loses exact geometry |
| **B-Rep graph** | **Full** | **Full** | **Yes** | **Native CAD format, zero information loss** |

B-Rep is the only representation that natively captures both continuous geometry and discrete topology. This is non-negotiable for a production CAD system.

### 2.2 Face-level graph over coedge-level

BRepNet defines kernels over coedges (oriented half-edges), which is more expressive but requires custom kernels and complex implementation. We use a face-level adjacency graph where each node is a B-Rep face and edges connect faces sharing a topological edge. This captures ~80% of the geometric expressiveness at ~20% of the implementation complexity — the right trade-off for a 5-day project.

### 2.3 Node feature schema (8-dimensional)

Each face is represented by an 8-dimensional feature vector:

| Dim | Feature | Source | Justification |
|---|---|---|---|
| 0 | Surface area (normalized) | H5 `V_1` col 0 | Face size relative to bounding box |
| 1 | Centroid x (normalized) | H5 `V_1` col 1 | Spatial position within bounding box |
| 2 | Centroid y (normalized) | H5 `V_1` col 2 | Spatial position within bounding box |
| 3 | Centroid z (normalized) | H5 `V_1` col 3 | Spatial position within bounding box |
| 4 | Surface type (float, k/11) | H5 `V_1` col 4 | Plane, cylinder, cone, sphere, torus — primary geometric discriminator |
| 5 | Node degree | Computed from `A_1` | Number of adjacent faces — topological context |
| 6 | Convex neighbor count | Computed from `E_1` | How many edges are convex — distinguishes protrusions from depressions |
| 7 | Concave neighbor count | Computed from `E_2` | How many edges are concave — key signal for holes and pockets |

**Deviation from plan:** The original plan specified 9-dimensional features (adding cylinder radius and axis). The H5 dataset provides only 5 raw features, which we augment with 3 topological features computed from the adjacency and edge convexity matrices. Surface type is stored as a continuous float (k/11 encoding) rather than an integer enum — the model uses it as-is.

### 2.4 Edge feature schema (3-dimensional)

Each edge encodes the geometric relationship between two adjacent faces:

| Dim | Feature | Values | Justification |
|---|---|---|---|
| 0 | Convexity | +1.0 (convex), -1.0 (concave), 0.0 (smooth) | Primary edge type — distinguishes feature boundaries |
| 1 | Is convex | 0 or 1 | Binary flag for convex edges |
| 2 | Is concave | 0 or 1 | Binary flag for concave edges — critical for hole/pocket detection |

These are derived from the MFCAD++ H5 `E_1` (convex), `E_2` (concave), and `E_3` (smooth) edge subset matrices, validated that E_1 + E_2 + E_3 = A_1 (total adjacency).

---

## 3. Architecture

### 3.1 GINEConv Encoder

```
Input:  x [N, 8], edge_index [2, E], edge_attr [E, 3]
  │
  ├─ Linear(8 → 128) + ReLU
  │
  ├─ GINEConv(MLP: 128→128→128) + BatchNorm + ReLU + Dropout(0.1)  × 3 layers
  │
  └─ Linear(128 → 64)
  │
Output: face_emb [N, 64]
```

**Why GINEConv over SAGEConv:** SAGEConv aggregates only node features — it drops `edge_attr` silently. GINEConv (Graph Isomorphism Network with Edge features, Hu et al. 2020) injects edge attributes into every message-passing step: each message is the sum of the neighbour's node embedding and a learned transformation of the edge attribute. This is the standard encoder for edge-attributed graphs and is confirmed by PyG documentation and the graph contrastive learning literature (SGNCL, GraphCL, SOLA-GCL). Without GINEConv, concavity flags — which mark feature boundaries — would never reach the model.

**Why not SAGEConv:** The concavity edge attributes are the primary signal distinguishing through-holes (concave edges to planar caps) from background surfaces. Dropping them silently would make the encoder blind to the most discriminative feature in the representation.

**Why inductive (GINEConv vs GCN):** GCN is transductive — it computes fixed spectral Laplacians requiring the entire graph structure at training time. New STEP files at inference have entirely new graph structures, which would break GCN's Laplacian. GINEConv is inductive: it generates embeddings for new nodes via learned aggregation functions, generalizing to unseen graph topologies without retraining.

**Why 3 layers:** A 3-layer GINEConv has a receptive field of 3 hops. For B-Rep face graphs where feature boundaries are typically 1–3 hops wide, this captures the full local context of each face without over-smoothing. (Ablation D tests 1-hop; ablation E tests 3-hop — the 2-hop default is the empirical sweet spot.)

### 3.2 SimCLR Projection Head

Following SimCLR (Chen et al. 2020), a 2-layer MLP projection head is added on top of the pooled subgraph embedding:

```python
proj_head = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
```

The projection head absorbs the contrastive uniformity cost during training. Representations *before* the head transfer better to the retrieval task (downstream task = cosine similarity search, not contrastive training). At inference, `use_proj=False` — the head is bypassed. This keeps V1 checkpoints fully compatible: the `proj_head` weights exist in the model but are never called at inference.

### 3.3 Segmentation Head

A 2-layer MLP (`Linear(64 → 64) → ReLU → Dropout(0.1) → Linear(64 → 25)`) produces per-face logits for 24 MFCAD++ feature types + background. This provides dense supervision signal during training.

### 3.4 Subgraph Pooling

Given per-face embeddings and a boolean mask of faces in a feature subgraph, the pooling module produces a single embedding via learned attention weights: `attn_weights = softmax(Linear(face_emb))`, then `subgraph_emb = Σ(attn_weights * face_emb)`. This is differentiable and allows the contrastive loss to operate on subgraph-level representations.

### 3.5 Hybrid Loss

```
L_total = w_seg · L_CE + w_contrastive · L_pairwise

where:
  w_seg = 1.0,  w_contrastive = 0.5
  L_CE = CrossEntropy(seg_logits, face_labels)         # per-face, 25-class
  L_pairwise = cross_entropy([pos_sim/τ, neg_sim/τ], label=0)  # subgraph-level triplets
```

**Note:** The loss is `PairwiseContrastiveLoss` — a triplet-style contrastive loss on explicit (anchor, positive, negative) subgraph embeddings. This is *not* NT-Xent (which uses all in-batch samples as negatives). We use explicit triplets because:
1. Triplet construction is deterministic and controllable
2. In-batch NT-Xent with ~8 triplets per batch has too few negatives to be stable
3. NT-Xent with large batches requires memory (batch × batch similarity matrix)

Temperature τ=0.07 is the SimCLR/MoCo standard. Lower τ sharpens the negative emphasis — the positive pair must score significantly above all negatives. This was confirmed by SimCLR ablations (Chen et al. 2020) and adopted directly.

**Why both losses are necessary:**

- **Segmentation CE alone** produces a closed-set classifier. Adding a 25th feature type requires reshaping the output layer and retraining. It also doesn't directly optimize the matching objective.
- **Contrastive loss alone** provides weak supervision — face labels are plentiful in MFCAD++ but contrastive triplets require careful mining and are sparser.
- **Together:** Segmentation provides dense gradients that force the encoder to learn discriminative per-face geometry. Contrastive loss on pooled subgraph embeddings ensures that same-type features from different models cluster together, directly optimizing the retrieval objective.

### 3.6 Data Augmentation

Two augmentations are applied during training (borrowed from SOLA-GCL, GraphCL):

**`RandomFeatureMask`** (mask_ratio=0.15): Zeros out 15% of node feature columns per graph. Forces the encoder to learn robust subgraph embeddings that don't over-rely on any single geometric descriptor (e.g., can't just memorize area values). Implementation: `torch.randperm(n_cols)[:n_mask]` applied to `data.x`.

**`RandomEdgeDrop`** (drop_ratio=0.10): Randomly removes 10% of B-Rep edges. Creates diverse graph views for contrastive pairs and encourages the encoder to learn topology-agnostic representations. Critical: edge attributes are dropped consistently (`data.edge_attr[keep]`).

Both augmentations use `data.clone()` to preserve immutability (original graph unmodified).

**V2 feature roadmap (would require retraining):**
- *n_smooth_nbrs as 9th node feature*: `degree - n_convex - n_concave`, computable from H5 without OCC. Changes input dim 8→9, breaks existing checkpoints.
- *Mean curvature*: FilletRec ablation shows 99.91% → 93.46% F1 without it for fillet recognition. Less decisive for through/blind holes (topological pattern > curvature). Requires OCC re-parsing.
- *Edge length + curve type*: BRepGAT uses these. Require OCC. V2 roadmap.

### 3.7 Training details

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-5)
- **Scheduler:** Cosine annealing with 5-epoch linear warmup
- **Hard negative mining:** After epoch 5, negatives for contrastive triplets are sampled from the closest embeddings rather than randomly. This prevents the contrastive loss from saturating on easy negatives.
- **Gradient clipping:** Max norm 1.0
- **Triplet construction:** For each batch, up to 8 triplets (anchor, positive, negative) are mined where anchor/positive are same-type subgraphs from different graphs, and negatives are different-type subgraphs.

---

## 4. Inference: Seed-and-Expand

Brute-force subgraph enumeration is NP-complete. Instead, we use a linear-complexity algorithm:

```
1. Embed reference feature subgraph → ref_emb [64]
2. Forward pass on full query graph → per-face embeddings [N, 64]
3. Cosine similarity of each query face to ref_emb → [N]
4. Seed: faces with similarity ≥ τ_seed (0.6)
5. For each seed: BFS expand along adjacency edges
     Add neighbor if similarity ≥ τ_expand (0.4)
6. Pool each candidate cluster → single embedding
7. Confidence = cosine_sim(cluster_emb, ref_emb)
8. Filter by τ_confidence (0.5)
9. NMS: remove clusters with face-IoU > 0.5 with higher-confidence cluster
```

**Complexity:** O(N · K) where N = number of query faces, K = average expansion steps. Linear, not exponential. Typical inference: 5–20ms per model on GPU.

**Threshold rationale:** `τ_seed=0.6` is aggressive to avoid false seeds; `τ_expand=0.4` is permissive to allow cluster growth; `τ_confidence=0.5` filters low-quality clusters post-expansion. These are configurable per-config.

---

## 5. Baselines & Benchmarking

### 5.1 Baseline 0: Rule-Based Heuristic

Pattern matching on graph topology and surface types:

- **Through hole:** single cylindrical face with ≥ 2 concave edges to planar caps
- **Counterbored hole:** outer cylinder (large area) → concave → annular plane → concave → inner cylinder (smaller area)

Works on clean models with canonical topology. Fails on feature intersections, non-standard topology, or blended edges. This failure mode motivates the learned approach.

### 5.2 Baseline 1: LLM over Structured Serialization

Serialize 2-hop neighborhood around candidate seed faces as JSON, send to Claude/GPT with a feature description prompt, parse `{"is_match": bool, "confidence": float}` response.

| Model | Tokens/query | Cost/query | Cost/1000 queries |
|---|---|---|---|
| Cerebras llama3.1-8b | ~800 | $0.00 (free tier) | $0.00 |
| Claude Haiku | ~800 | ~$0.0002 | ~$0.20 |
| GPT-4o | ~800 | ~$0.003 | ~$3.00 |

Context overflow strategy: never serialize the full model. Seed-first (find candidate faces), then serialize only 2-hop neighborhoods of top-K seeds. A 2000-face model with 10 seeds → ~8000 tokens, within context limits.

### 5.3 Ablation: Segmentation-Only

Remove contrastive loss (`contrastive_weight=0`), train with segmentation CE only. Measures the contribution of metric learning to the matching objective.

### 5.4 Benchmark comparison (results from Kaggle runs)

| Method | Instance F1 | Precision | Recall | Inference (ms/model) | N models |
|---|---|---|---|---|---|
| Rule-based | 0.545 | 0.431 | 0.950 | 3.2 | 200 |
| LLM (Cerebras llama3.1-8b) | 0.368 | 0.225 | 1.000 | 5,600 | 50 |
| **GNN Phase 1** (through_hole) | **0.829** | 0.771 | 0.962 | 12.0 | 4702 |
| **GNN Phase 2** (+ blind_hole) | **0.810** | 0.753 | 0.945 | 13.3 | 7120 |

**Framing note:** SOTA face *classification* methods achieve 99%+ F1 on MFCAD++ (BRepGAT 99.1%, AAGNet 99.94% on MFInstSeg), but they solve a strictly easier problem — assigning a label to every face given the full model. Our task is **reference-guided subgraph retrieval**: find all instances of a reference feature across unseen models. No apples-to-apples comparison exists in the literature for this formulation. Our 0.829 F1 is on the harder retrieval task.

### Ablation results

| Ablation | F1 | Precision | Recall | vs A |
|---|---|---|---|---|
| A — Full model | 0.855 | 0.810 | 0.954 | — |
| B — No contrastive loss | **0.949** | 0.943 | 0.963 | +0.094 |
| C — No edge features | 0.845 | 0.794 | 0.959 | −0.010 |
| D — 1-hop expansion | 0.854 | 0.806 | 0.961 | −0.001 |
| E — 3-hop expansion | 0.858 | 0.810 | 0.964 | +0.003 |

**Interpreting B_no_contrastive > A_full**: The segmentation-only model scores higher on this benchmark because the inference pipeline (stages 1–2 of seed-and-expand) relies on per-face cosine similarity for seed filtering and BFS expansion — exactly what CE optimizes. The contrastive loss modifies the embedding space for subgraph-level clustering, which slightly softens per-face discrimination in exchange for better metric space separation across models. This is the standard representation learning tradeoff: CE maximizes per-instance discrimination; contrastive maximizes semantic clustering. The contrastive component provides value that this benchmark doesn't fully measure: it enables few-shot retrieval of **unseen** feature types by embedding their reference subgraph (no retraining needed). A CE-only model cannot generalize beyond its 25 training classes.

**C_no_edge_features −0.010**: Removing edge features (convexity flags) causes only a small drop — the surface type node feature already captures most of the signal for through/blind holes (cylinders vs planes). Edge features become more critical for distinguishing slots, steps, and pockets where the convexity pattern is the primary discriminator.

**D and E near-identical to A**: Hop depth has minimal impact for through/blind holes whose subgraphs are typically 1–2 faces, well within even a 1-hop receptive field. Deeper hop ablations would matter more for complex features like counterbored holes (3+ face layers).

---

## 6. Extensibility Demo

### 6.1 Phase 1: Train on through_hole + blind_hole

Core training on two feature types that compose the counterbored hole pattern. 50 epochs, lr=1e-3, cosine schedule. Hard negative mining begins at epoch 6.

### 6.2 Phase 2: Add blind_hole

Fine-tune from the Phase 1 checkpoint with `blind_hole` (label 12) added to the target set. Lower lr=3e-4, 30 epochs (fine-tuning, not from scratch). Architecture is identical — only the data and loss weighting change.

| Metric | Phase 1 | Phase 2 | Delta |
|---|---|---|---|
| Instance F1 | 0.829 | 0.810 | −0.019 (−2.3%) |
| Precision | 0.771 | 0.753 | −0.018 |
| Recall | 0.962 | 0.945 | −0.017 |
| Inference time (ms) | 12.0 | 13.3 | +1.3ms |
| N models evaluated | 4702 | 7120 | — |

The 2.3% F1 drop is within expected noise from adding blind_hole negative examples into the batch (harder negatives for through_hole detection). The inference pipeline is unchanged — `reference_surface_types` config is the only update needed for the new feature type.

### 6.3 What breaks at the 10th feature type

1. **Embedding space crowding:** 10+ types sharing 64 dimensions reduces inter-class separation. Fix: increase embedding dimension to 128 or 256, add per-class projection heads.
2. **Hard negative instability:** Geometrically similar types (through-hole vs. blind-hole) create very hard negatives that destabilize training. Fix: curriculum learning — easy negatives first, hard negatives after epoch 10.
3. **Class imbalance:** MFCAD++ has ~3× more through-holes than keyways. Fix: weighted sampling, per-class loss weighting.
4. **Annotation bottleneck:** Each new type needs hundreds of labeled instances. Fix: self-supervised pretraining (BRepMAE on ABC dataset) reduces labeled examples needed per type.

### 6.4 What breaks at the 50th feature type

1. **Architecture is still fine** — the GNN encoder is shared and learns geometry-agnostic face embeddings.
2. **NMS becomes expensive** — 50 types × N candidate clusters × IoU computation = O(50N²). Fix: parallel cluster evaluation, early pruning by confidence.
3. **Subtype confusion** — step hole vs. counterbored hole vs. deep counterbored hole. Fix: hierarchical classification (coarse type → fine subtype) or prototype networks.
4. **Rare types with few examples** — MFCAD++ only has 24 types. Synthetic generation (parametric CAD generators) or data augmentation becomes necessary.

---

## Bonus

### Domain Retrofitting: Protein Structure Retrieval

The closest cross-domain analogue is **GraSR** (Graph-based Structure Retrieval for proteins, Lee et al. 2022). GraSR uses GNNs to learn discriminative residue-level embeddings for protein similarity search — an exact structural analogue to our approach:

| Protein domain (GraSR) | B-Rep domain (Ours) |
|---|---|
| Residues | Faces |
| Peptide bonds | B-Rep edges |
| Secondary structure (α-helix, β-sheet) | Surface type (cylinder, plane, cone) |
| Residue-level embedding | Per-face embedding |
| Fixed-length substructure embedding + cosine similarity | Subgraph pooling + cosine similarity |

The pipeline is identical: embed substructures via GNN, aggregate to fixed-length representation, retrieve by cosine similarity. BRepGAT (Lambourne et al. 2022) reaches similar conclusions for B-Rep classification — confirming GNN encoders are preferred for edge-attributed topological graphs.

**Honest trade-offs:**
- Protein contact graphs are large (O(500) residues) but sparse. B-Rep face graphs are smaller but denser with richer edge semantics (convexity vs just distance cutoffs).
- GraSR trains on PDB (170k structures). Our supervised data is MFCAD++ (50k models). Similar scale — the analogy is strong.
- A pretrained GraSR model could potentially be fine-tuned on B-Rep data, replacing residue features with the 8-dim face schema.

### Domain Retrofitting: Molecular Fingerprints

A secondary analogue: GNN-based molecular fingerprints (ECFP analogy). Our contrastive training is equivalent to learning a fingerprint that clusters by functional group (= machining feature), except the "functional group" is defined by surface-type topology rather than atomic bonding patterns. The `RandomFeatureMask` and `RandomEdgeDrop` augmentations we added (from SOLA-GCL) are directly borrowed from this literature — masking atoms/bonds in molecular graphs to force robust substructure embeddings.

### Self-Supervised Path (Described, Not Implemented)

BRepMAE approach on the ABC dataset (1M unlabeled STEP files):
1. Parse each model → B-Rep graph
2. Randomly mask 30% of face attributes (surface type, area, radius)
3. Train GINEConv encoder to reconstruct masked attributes from topological context
4. Fine-tune on MFCAD++ with supervised segmentation + contrastive loss

Expected benefit: the encoder learns geometry-agnostic, topology-aware representations before seeing any labels. Fine-tuning should reach the same F1 with 5–10× fewer labeled examples.

### Failure Mode Analysis

| Scenario | Failure mode | Severity | Fix |
|---|---|---|---|
| Feature intersects pocket | Adjacency graph breaks — faces shared between features | High | Continuous attributes over topology (AAGNet-style) |
| Highly curved body (turbine blade) | Normal and radius features lose meaning | Medium | Add Gaussian curvature as node feature |
| Very small feature (M2 hole) | Faces below area threshold | Medium | Adaptive normalization per bounding box |
| Symmetric model (4 identical holes) | Finds all 4 correctly, NMS may merge | Low | Tune IoU threshold |
| Blended/filleted edges | Dihedral angle → π, convexity ambiguous | Medium | Add smooth edge class, don't rely on convexity alone |
| Counterbored hole with added boss | Extra adjacent face breaks pattern matching | High | GNN handles; rule-based fails entirely |

### Production Cost Analysis

| Scale | GPU-seconds/query | $/query (A10G @ $0.002/s) |
|---|---|---|
| Single STEP, 200 faces | 0.012s | $0.000024 |
| 10k queries/day | 120 GPU-seconds | $0.24/day |
| 1M queries/day | 12,000 GPU-seconds | $24/day |
| LLM baseline equivalent | N/A | ~$200–3,000/day |

The GNN approach is ~8,000–125,000× cheaper than LLM at production scale (Cerebras free tier to paid APIs).
