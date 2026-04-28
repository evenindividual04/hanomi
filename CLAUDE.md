# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hanomi ML Engineer take-home: **feature recognition across CAD models**. The correct framing is **metric learning on B-Rep subgraphs** — not face classification. Given a reference STEP model with labeled feature face IDs, find all semantically equivalent subgraphs in query models.

## Environment Setup

```bash
# Primary: use the lockfile
conda env create -f environment.yml
conda activate hanomi

# pythonocc-core must come from conda-forge (no pip build available)
conda install -c conda-forge pythonocc-core=7.7.2
```

## Commands

```bash
# Validate dataset integrity before any training
python scripts/validate_data.py

# Explore class distribution and graph statistics
python scripts/explore_dataset.py

# Preprocess STEP files → PyG .pt graphs (only needed for STEP path)
python scripts/preprocess.py \
    --input_dir /Users/anmolsen/Developer/MFCAD++_dataset/step \
    --output_dir data/processed \
    --feature_types counterbored_hole through_hole \
    --num_workers 4

# Train Phase 1 (counterbored holes) — output_dir auto-increments to next run_NNN
python scripts/train.py --config configs/counterbored_hole.yaml --seed 42 \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs

# Train Phase 2 (add through-holes, fine-tune from checkpoint)
python scripts/train.py --config configs/extensibility_v2.yaml \
    --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
    --checkpoint results/runs/run_001/checkpoints/best.pt

# Inference: reference + query → JSON
python scripts/inference_demo.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --reference_step path/to/ref.step \
    --reference_face_ids 3 4 5 \
    --query_dir path/to/queries/ \
    --output results/demo_output.json

# Evaluate GNN + baselines, generate comparison table
python scripts/evaluate.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --test_dir data/processed/test \
    --methods gnn rule_based llm \
    --output_dir results/runs/run_001

# Evaluate baselines only (no checkpoint needed)
python scripts/run_baselines.py

# Ablation experiments
python scripts/train.py --config configs/ablations/A_full.yaml --output_dir results/ablations/A_full
# configs/ablations/[A_full|B_no_contrastive|C_no_edge_features|D_1hop|E_3hop].yaml

# Tests
pytest tests/ -v
pytest tests/test_h5_dataset.py -v          # single file
pytest --cov=src --cov-report=term-missing  # with coverage

# Lint
ruff check src/ scripts/ tests/
```

## Architecture

```
STEP file → pythonocc-core → B-Rep AAG (faces=nodes, shared-edges=edges)
    ↓ 9-dim node features (STEP path) / 8-dim (H5 path), 3-dim edge features
GINEConv encoder (3 layers, hidden=128, out=64)        ← BRepEncoder
    ↓ per-face embeddings [N, 64]
    ├── SegmentationHead  → [N, 25] logits → cross-entropy loss
    └── SubgraphPooling   → [dim] embedding → pairwise contrastive loss
```

**Training loss** (`src/losses/hybrid.py`): `L = 1.0·L_CE + 0.5·L_contrastive` (temperature τ=0.07).

**Inference** (`src/inference/seed_expand.py`) — 3 stages, no brute-force:
1. Heuristic seed filtering by surface type (CPU, no GNN)
2. k-hop BFS expansion around each seed (default k=2)
3. Single GNN forward pass → cosine similarity against reference embedding → NMS at IoU 0.5

**Baselines**:
- `src/baselines/rule_based.py` — topological pattern matching on face type + concave-edge sequences
- `src/baselines/llm_baseline.py` — serialize 2-hop subgraph → JSON → Claude/GPT; never serialize full model; chunk if >50 seeds

## Feature Schema

**Node features** differ by data path:
- **H5 path** (8-dim): `[area, cx, cy, cz, surface_type/11, degree, n_convex_nbrs, n_concave_nbrs]`
- **STEP path** (9-dim): `[surface_type, area, normal_xyz(3), cylinder_radius, cylinder_axis_z, num_adjacent_faces, num_boundary_edges]`

`base.yaml` sets `node_in_dim: 8` (H5 path). **A model trained on H5 cannot be used directly on STEP-parsed graphs** — the feature schemas are different. To use the STEP path end-to-end: preprocess STEP files with `scripts/preprocess.py`, then train with `node_in_dim: 9` in the config.

`step_to_graph(filepath)` in `src/parsing/step_parser.py` is the convenience entry point for inference on raw STEP files — returns a PyG Data object with real `occ_face_ids` (e.g. `#12`).

**Edge features** (3-dim): `[convexity ∈ {-1,0,1}, is_convex, is_concave]`

**Surface type encoding**: In H5 node features, `surface_type` is stored pre-divided by 11. Recover the integer type with `int(x[i, 4] * 11)`. This pattern appears in `seed_expand.py`, `rule_based.py`, and anywhere surface types are compared.

## Datasets

Datasets live **outside** the project directory so they are excluded when zipping the repo for Kaggle submission. Configs reference them via absolute paths.

### MFCAD++ (`/Users/anmolsen/Developer/MFCAD++_dataset/`)
- **Format**: H5 files in `hierarchical_graphs/` — `{training,val,test}_MFCAD++.h5`
- **Splits**: 41766 / 8950 / 8949 models
- **Labels**: 25 classes (0=Chamfer … 24=Stock). Full map in `src/data/h5_dataset.py::LABEL_NAMES`.
- **Counterbored hole**: Not a native class. Proxied by training jointly on `through_hole` (1) + `blind_hole` (12); metric learning separates the adjacency subgraph pattern.
- **H5 `idx` field is unreliable** — it stores absolute global node offsets that drift across batches. `MFCADPlusPlusDataset._parse_batch_group` ignores it and uses connected-component analysis on `A_1` to locate model boundaries. Do not use `idx` directly.

### Fusion 360 Gallery (`/Users/anmolsen/Developer/s1.0.0/`)
- **Format**: `.smt` + `.seg` files; 8 segment types in `segment_names.json`
- **Use**: Generalization test only — MFCAD++ is the primary training set

## Critical Design Decisions

- **GraphSAGE not GCN**: inductive — generalizes to unseen STEP files at inference without retraining
- **Hybrid loss**: contrastive component means adding new feature types requires new data, not architecture changes
- **Seed-and-expand**: O(N·K) not exponential — enumerate seeds by surface type, BFS-expand k hops, single GNN forward, NMS
- **LLM baseline**: 2-hop neighborhoods only; serialize to JSON; chunk if >50 seeds

## Inference Output Format

```json
{
  "query_model": "query1.step",
  "instances": [
    {"face_ids": [12, 13, 14], "occ_face_ids": ["#456", "#457", "#458"], "confidence": 0.921}
  ]
}
```

## Config System

All experiments extend `configs/base.yaml`. Key overrides in derived configs:
- `feature_types` — which MFCAD++ labels to train on
- `encoder.node_in_dim` — must match data path (8 for H5, 9 for STEP)
- `inference.reference_surface_types` — surface type ints per feature class
- Ablations in `configs/ablations/` — see README there for expected F1 deltas

All scripts accept `--seed` (default 42) via `src/utils/seed.py::set_seed()`.
