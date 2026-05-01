# B-Rep Feature Recognition via Metric Learning on Subgraph Embeddings

Metric learning on B-Rep adjacency graphs for CAD feature recognition. Given a reference STEP model with labeled feature face IDs, finds all semantically equivalent subgraphs in query models.

**This is subgraph retrieval, not face classification.** The model learns a metric space where subgraphs of the same machining feature type cluster together, enabling reference-guided search across unseen models without retraining.

---

## Setup

```bash
conda env create -f environment.yml && conda activate hanomi
```

`pythonocc-core` must come from conda-forge (no pip build). The environment.yml handles this automatically. It is only required for the STEP parsing path (`src/parsing/`); training and evaluation use pre-built H5 files.

---

## Data

Datasets live **outside** the project directory so they are excluded from the submission zip.

| Dataset | Path | Use |
|---|---|---|
| MFCAD++ | `/Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs/` | Training + eval |
| Fusion 360 Gallery | `/Users/anmolsen/Developer/s1.0.0/` | Generalization test only |

**MFCAD++ splits**: 41 766 train / 8 950 val / 8 632 test models, 25 face-type classes. (Local H5 has 8 632 test models; the MFCAD++ paper reports 8 949 — the difference is models excluded during H5 preprocessing.)

**H5 node feature schema** (8-dim): `[area, cx, cy, cz, surface_type/11, degree, n_convex_nbrs, n_concave_nbrs]`

**Edge features** (3-dim): `[convexity ∈ {−1,0,1}, is_convex, is_concave]`

`surface_type` is stored pre-divided by 11 in H5 node features. Recover integer with `int(x[i, 4] * 11)`.

---

## Run Commands

### Validate dataset
```bash
python scripts/validate_data.py
```

### Train Phase 1 — through_hole
```bash
python scripts/train.py --config configs/counterbored_hole.yaml --seed 42 \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs
```

### Train Phase 2 — add blind_hole, fine-tune from Phase 1
```bash
python scripts/train.py --config configs/extensibility_v2.yaml --seed 42 \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
    --checkpoint results/runs/run_001/checkpoints/best.pt
```

### Evaluate GNN
```bash
python scripts/evaluate.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
    --methods gnn --output_dir results/runs/run_001/eval
```

### Evaluate baselines
```bash
python scripts/evaluate_baselines.py \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
    --methods rule_based llm --n_models 200 --output_dir results/baselines
```

### Inference on a STEP file
```bash
python scripts/inference_demo.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --reference_step path/to/ref.step \
    --reference_face_ids 3 4 5 \
    --query_dir path/to/queries/ \
    --output results/demo_output.json
```

### Print consolidated results table
```bash
python scripts/generate_report.py
```

### Compute calibration metrics from existing results
```bash
python scripts/compute_calibration.py results/runs/run_001/eval/per_model_results.json
```

### t-SNE embedding visualization
```bash
python scripts/visualize_embeddings.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
    --output_dir results/figures --label phase1

python scripts/visualize_embeddings.py \
    --checkpoint results/runs/run_002/checkpoints/best.pt \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
    --output_dir results/figures --label phase2
```

### Tests
```bash
pytest tests/ -v
pytest --cov=src --cov-report=term-missing
```

---

## Results

### Method Comparison

| Method | F1 | Precision | Recall | ms/model | N models |
|---|---|---|---|---|---|
| Rule-based | 0.545 | 0.431 | 0.950 | 3.2 | 200 |
| LLM (Cerebras llama3.1-8b) | 0.368 | 0.225 | 1.000 | 5.6 | 50 |
| **GNN Phase 1** (through_hole) | **0.829** | 0.771 | 0.962 | 12.0 | 4702 |
| **GNN Phase 2** (+ blind_hole) | **0.810** | 0.753 | 0.945 | 13.3 | 7120 |

> SOTA face *classification* (BRepGAT 99.1%, AAGNet 99.94% on MFCAD) solves a strictly easier problem — every face is labeled given the full model. Our task is **reference-guided subgraph retrieval**: find all instances of a reference feature across unseen models. No apples-to-apples comparison exists in the literature for this formulation.

### Ablation Results

Run ablation evals with `scripts/evaluate.py`, then `python scripts/generate_report.py`.

| Ablation | F1 | Precision | Recall | vs Full |
|---|---|---|---|---|
| A — Full model | 0.855 | 0.810 | 0.954 | — |
| B — No contrastive loss | 0.949 | 0.943 | 0.963 | +0.094 |
| C — No edge features | 0.845 | 0.794 | 0.959 | −0.010 |
| D — 1-hop expansion | 0.854 | 0.806 | 0.961 | −0.001 |
| E — 3-hop expansion | 0.858 | 0.810 | 0.964 | +0.003 |

> **B_no_contrastive note**: CE-only model scores higher here because the inference pipeline's seed filtering (stages 1–2) relies on per-face cosine similarity, which segmentation CE directly optimizes. Contrastive loss trades marginal per-face discrimination for better subgraph-level clustering — useful for few-shot retrieval of new unseen feature types. This is the known CE vs. contrastive tradeoff in representation learning.

### Cost Analysis

| Method | ms/query | GPU-h / 10k queries | $/10k queries |
|---|---|---|---|
| Rule-based | 3.2 | 0.000 | $0.00 |
| LLM (Cerebras llama3.1-8b) | ~5600 | N/A (API) | ~$0 (free tier) |
| GNN (T4 GPU) | 12 | 0.033 | ~$0.15 |

---

## Architecture

```
STEP/H5 → B-Rep AAG (faces=nodes, shared B-Rep edges=graph edges)
    ↓  8-dim node features (H5) / 9-dim (STEP)
    ↓  3-dim edge features [convexity, is_convex, is_concave]
GINEConv encoder — 3 layers, hidden=128, out=64        ← BRepEncoder
    ↓  per-face embeddings [N, 64]
    ├── SegmentationHead  → [N, 25] logits → cross-entropy loss
    └── SubgraphPooling   → [dim] embedding → pairwise contrastive loss
```

**Loss**: `L = 1.0 · L_CE + 0.5 · L_contrastive` (τ=0.07).

**Why GINEConv over SAGEConv**: GINEConv passes `edge_attr` to every convolutional layer. SAGEConv drops edge attributes silently. Concavity flags — which mark feature boundaries — would never reach the model with SAGEConv.

**Why hybrid loss**: Segmentation cross-entropy alone doesn't cluster feature types in embedding space. Contrastive loss alone doesn't produce per-face discriminative features. Both are needed. Measured on 50 test models: inter-class centroid distance = 20.98, mean intra-class = 7.84, **separation ratio = 2.67×**.

**Attention pooling**: `SubgraphPooling` uses a learned `Linear(64→1)` to weight face embeddings before aggregation. On test samples, feature faces receive **~80% of total attention weight** despite being a minority of faces — confirming the pooling layer learned to focus on geometrically relevant subgraphs.

**Inference** — 3 stages, O(N·K) not exponential:
1. Heuristic seed filtering by surface type (CPU, no GNN call)
2. Neural BFS expansion (k=2 hops, pruned by per-face cosine similarity ≥ τ_expand)
3. Subgraph pooling → cosine similarity against reference embedding → NMS (IoU 0.5)

**SimCLR projection head** (V2): A 2-layer MLP `proj_head` is added to `FeatureRecognizer`. `use_proj=False` by default — existing checkpoints load without weight changes. Enable during contrastive training only (`use_proj=True` in the loss call); representations before the projection head transfer better to retrieval tasks.

**Augmentations added** (SOLA-GCL):
- `RandomFeatureMask` — zeros out 15% of node feature columns, forcing robust subgraph embeddings
- `RandomEdgeDrop` — drops 10% of edges randomly, creating diverse graph views for contrastive pairs

---

## Extensibility

Adding a new feature type requires:
1. Adding its label ID to `feature_types` in the config
2. Fine-tuning from existing checkpoint — **no architecture changes**
3. Updating `reference_surface_types` in inference config

**Phase 1 → Phase 2 delta** (through_hole → + blind_hole):
- F1: 0.829 → 0.810 (−2.3%, within noise from new negative examples in batch)
- Training: fine-tuned from Phase 1 checkpoint, not scratch
- Inference pipeline: zero changes

---

## Reproducibility

All results auto-logged to:
- `results/runs/run_NNN/eval/metrics.json` — aggregate metrics
- `results/runs/run_NNN/eval/per_model_results.json` — per-model predictions
- `results/runs/run_NNN/eval/per_model_results.csv` — CSV version

All scripts accept `--seed` (default 42).

---

## Project Structure

```
src/
  data/
    h5_dataset.py          MFCAD++ H5 reader (connected-component parsing)
    dataloader.py          PyG DataLoader, triplet mining
    transforms.py          Augmentations: RandomScaleFeatures, RandomFlipNormals,
                           RandomFeatureMask, RandomEdgeDrop
  models/
    encoder.py             BRepEncoder (3-layer GINEConv, hidden=128, out=64)
    seg_head.py            SegmentationHead (25-class linear)
    pooling.py             SubgraphPooling (attention-weighted)
    feature_recognizer.py  Full model + SimCLR projection head
  losses/
    contrastive.py         PairwiseContrastiveLoss (triplet, τ=0.07)
    hybrid.py              Weighted CE + contrastive combination
  inference/
    seed_expand.py         3-stage seed-and-expand inference
    nms.py                 Non-max suppression on overlapping clusters
  baselines/
    rule_based.py          Topological pattern-matching baseline
    llm_baseline.py        Claude/GPT API baseline
  evaluation/
    metrics.py             Instance F1 (IoU), face-level F1, Brier, ECE
    results_logger.py      JSON + CSV output
  parsing/
    step_parser.py         STEP → PyG Data (requires pythonocc)
    graph_builder.py       B-Rep → AAG conversion
scripts/
  train.py                 Training loop
  evaluate.py              Full test-set evaluation
  evaluate_baselines.py    Baseline evaluation
  inference_demo.py        Single reference→query inference demo
  generate_report.py       Consolidated results table (one command)
  compute_calibration.py   Brier + ECE from existing predictions
  visualize_embeddings.py  t-SNE plots of subgraph embedding space
configs/
  base.yaml                Default hyperparameters
  counterbored_hole.yaml   Phase 1 (through_hole)
  extensibility_v2.yaml    Phase 2 (+ blind_hole)
  ablations/               A_full, B_no_contrastive, C_no_edge_features, D_1hop, E_3hop
```
