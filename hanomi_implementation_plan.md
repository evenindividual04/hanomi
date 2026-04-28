# Hanomi ML Engineer Take-Home — Implementation Plan
## Feature Recognition Across CAD Models

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Architecture Decision](#2-architecture-decision)
3. [Environment Setup](#3-environment-setup)
4. [Repository Structure](#4-repository-structure)
5. [Dataset Pipeline](#5-dataset-pipeline)
6. [B-Rep Graph Construction](#6-b-rep-graph-construction)
7. [Model Architecture](#7-model-architecture)
8. [Training Pipeline](#8-training-pipeline)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Baselines](#10-baselines)
11. [Extensibility Demo](#11-extensibility-demo)
12. [Evaluation & Benchmarking](#12-evaluation--benchmarking)
13. [Programmatic Results Logging](#13-programmatic-results-logging)
14. [Bonus Sections](#14-bonus-sections)
15. [Day-by-Day Schedule](#15-day-by-day-schedule)
16. [Write-Up Guide](#16-write-up-guide)
17. [45-Min Walkthrough Prep](#17-45-min-walkthrough-prep)

---

## 1. Problem Framing

### 1.1 What the task actually is

This is **not** a standard classification problem. Most B-Rep papers (BRepNet, AAGNet) solve
segmentation — label every face in a model. Our task is different:

```
Given:  reference STEP model + a labeled feature subgraph (face IDs + OCC face IDs)
Find:   all subgraphs in query STEP models that are semantically equivalent to that feature
Output: per-instance {face_ids, occ_face_ids, confidence_score}
```

The correct formulation is **metric learning on B-Rep subgraphs**:
- Embed the reference feature subgraph into a vector space
- Enumerate candidate subgraphs in the query model
- Retrieve candidates whose embedding is closest to the reference
- Threshold by confidence

"Same feature" is semantic — a counterbored hole matches regardless of diameter, depth,
orientation, or surrounding body geometry. Size and pose are irrelevant; topological
structure and surface type composition are what matter.

### 1.2 Why this framing matters for every design choice

| Design Choice | Classification framing | Metric learning framing (ours) |
|---|---|---|
| Loss function | Cross-entropy | Contrastive / triplet |
| Query time | Forward pass → argmax | Embed reference → nearest neighbor search |
| New feature type | Retrain classifier head | Add labeled examples → new cluster in embedding space |
| Generalizes to unseen feature? | No | Potentially yes |

This distinction is the answer to the extensibility question: adding a new feature type is
a **data problem** (provide labeled examples) not an architecture problem (no retraining
of shared encoder needed for inference).

### 1.3 Key invariances the model must respect

- Size (diameter, depth) — same feature type regardless of dimensions
- Orientation — a hole pointing down = hole pointing sideways
- Position — anywhere in the body
- Surrounding geometry — hole in a flat plate = hole in a curved boss

---

## 2. Architecture Decision

### 2.1 Representation: B-Rep Attributed Adjacency Graph

**Why B-Rep over alternatives:**

| Representation | Geometry | Topology | Invariant to tessellation | Notes |
|---|---|---|---|---|
| Point cloud | ✅ | ❌ | ❌ | Loses topology, needs sampling |
| Mesh | ✅ | Partial | ❌ | Tessellation-dependent |
| Multi-view images | Partial | ❌ | ✅ | Loses exact geometry |
| B-Rep graph | ✅ | ✅ | ✅ | Native CAD format |

B-Rep is the only representation that natively captures both continuous geometry and discrete
topology without information loss. This is non-negotiable for a production CAD system.

**Why face-level graph over coedge-level (BRepNet):**

BRepNet defines kernels over coedges (oriented half-edges). More expressive, but:
- Complex implementation, custom kernels
- High risk on a 5-day timeline
- Face-level adjacency graph captures ~80% of the value with 20% of the complexity

We use face-level graph with edge attributes encoding the geometric relationship between
adjacent faces (dihedral angle, convexity). This is the AAGNet formulation and directly
handles intersection cases.

### 2.2 Model: GraphSAGE with Hybrid Loss

```
STEP file
   ↓  pythonocc-core
B-Rep AAG (faces=nodes, shared-edges=edges)
   ↓  per-face feature vector
GraphSAGE encoder (2-3 layers)
   ↓
Per-face embeddings
   ↓           ↓
Segmentation   Subgraph pooling
head           (mean/attention)
(cross-entropy) ↓
               Feature-level embedding
               ↓
               Contrastive loss
```

**Why hybrid loss (segmentation + contrastive):**

- Segmentation cross-entropy: strong supervision from per-face labels in MFCAD++,
  forces encoder to learn discriminative per-face geometry
- Contrastive on subgraph embeddings: forces same-type features to cluster regardless
  of context, directly optimizes the matching objective.

### Hardware requirements
- **Training**: Single NVIDIA T4 or P100 (Kaggle notebooks are perfectly suited, 16GB VRAM target).
- **Inference**: CPU-only feasible for single parts (<0.5s per forward pass).


Using only one or the other is strictly worse.

### 2.3 Inference: Seed-and-Expand

Brute-force subgraph enumeration is NP-complete. Instead:

```
1. Run GNN on full query graph → per-face embeddings
2. For each face in query: cosine similarity against reference feature faces (seeds)
3. Select top-K seed faces above threshold τ_seed
4. Expand each seed along adjacency edges greedily (BFS)
   - Add neighbor if: similarity to reference subgraph improves or stays stable
   - Stop when: adding next neighbor decreases cluster similarity
5. Pool candidate cluster → single embedding
6. Cosine similarity against reference subgraph embedding → confidence score
7. NMS: remove clusters with IoU > 0.5 with higher-confidence cluster
```

This is O(N·K) at inference where N = number of faces, K = expansion steps. Linear,
not exponential.

### 2.4 Baselines

Three baselines in increasing sophistication:

1. **Rule-based heuristic** (Baseline 0): topological pattern matching on face type sequences
2. **LLM over structured serialization** (Baseline 1): local subgraph → JSON → Claude/GPT
3. **GNN segmentation only, no contrastive** (Ablation): removes metric learning component

### 2.5 Self-supervised path (described, not implemented)

Cite BRepMAE: pretrain GraphSAGE encoder on ABC dataset (1M unlabeled STEP files) by
masking random face attributes (surface type, area, radius) and training the network to
reconstruct them from topological context. Fine-tune on MFCAD++. Expected benefit: better
generalization to unseen geometry, smaller labeled dataset requirement.

---

## 3. Environment Setup

### 3.1 Conda environment

```bash
conda create -n hanomi python=3.10
conda activate hanomi
```

### 3.2 Core dependencies

```bash
# pythonocc-core — STEP parsing (conda-forge only, not pip)
conda install -c conda-forge pythonocc-core=7.7.2

# Deep learning
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Graph utilities
pip install networkx==3.2

# Data & experiment tracking
pip install numpy pandas scikit-learn tqdm
pip install wandb                    # experiment tracking
pip install pytest                   # testing

# Visualization & results
pip install matplotlib seaborn
pip install trimesh                  # optional mesh viz

# LLM baseline
pip install anthropic                # or openai

# Notebook (Colab fallback)
pip install ipykernel jupyter
```

### 3.3 requirements.txt (pinned)

```
pythonocc-core==7.7.2        # conda-forge
torch==2.1.0
torch-geometric==2.4.0
networkx==3.2.1
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2
tqdm==4.66.1
wandb==0.16.1
matplotlib==3.8.2
seaborn==0.13.0
anthropic==0.8.1
pytest==7.4.3
```

### 3.4 Colab setup cell

```python
# Cell 1: Mount drive + install deps
from google.colab import drive
drive.mount('/content/drive')

!pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html -q

# pythonocc must be conda, use this workaround in Colab:
!pip install cadquery -q  # fallback if pythonocc conda fails in Colab
# Preferred: upload pre-built pythonocc wheel or use conda in Colab via condacolab
!pip install condacolab -q
import condacolab
condacolab.install()
!conda install -c conda-forge pythonocc-core -y -q
```

### 3.5 Reproducibility

```python
# src/utils/seed.py
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

All train/eval scripts accept `--seed` flag, default 42.

---

## 4. Repository Structure

```
hanomi-feature-recognition/
│
├── README.md                          # run instructions, architecture overview
├── requirements.txt
├── environment.yml                    # conda env export
├── setup.py                           # makes src/ importable
├── .gitignore
│
├── configs/
│   ├── base.yaml                      # default hyperparameters
│   ├── counterbored_hole.yaml         # feature-type-specific overrides
│   ├── through_hole.yaml
│   └── extensibility_v2.yaml         # two-type training config
│
├── data/
│   ├── raw/                           # untouched downloads
│   │   ├── mfcad++/
│   │   └── fusion360/
│   ├── processed/                     # .pt graph files per model
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── splits/
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── step_parser.py             # pythonocc STEP → face/edge data
│   │   ├── graph_builder.py           # face data → PyG Data object
│   │   ├── feature_extractor.py       # per-face geometric attributes
│   │   └── utils.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mfcad_dataset.py           # PyG Dataset class for MFCAD++
│   │   ├── transforms.py              # augmentation, normalization
│   │   └── dataloader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                 # GraphSAGE encoder
│   │   ├── seg_head.py                # per-face segmentation head
│   │   ├── pooling.py                 # subgraph → embedding (mean, attention)
│   │   └── feature_recognizer.py     # full model combining all components
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── contrastive.py             # NT-Xent / triplet margin loss
│   │   └── hybrid.py                  # weighted seg + contrastive
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── seed_expand.py             # seed-and-expand subgraph search
│   │   ├── nms.py                     # non-max suppression on overlapping instances
│   │   └── confidence.py              # cosine similarity → confidence score
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── rule_based.py              # topological pattern matching (Baseline 0)
│   │   └── llm_baseline.py            # LLM over serialized subgraph (Baseline 1)
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                 # precision, recall, F1, IoU at face level
│       └── results_logger.py          # programmatic results → JSON + CSV
│
├── scripts/
│   ├── download_data.sh               # download MFCAD++ + Fusion360
│   ├── preprocess.py                  # raw STEP → processed .pt graphs
│   ├── train.py                       # main training entry point
│   ├── evaluate.py                    # run eval on test set, dump results
│   ├── inference_demo.py              # single reference + query → output
│   └── run_baselines.py               # run all baselines, dump comparison
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # visualize MFCAD++ graphs
│   ├── 02_training_curves.ipynb       # loss/metric plots from wandb
│   ├── 03_qualitative_results.ipynb   # visualize matched features on models
│   └── 04_extensibility_demo.ipynb    # one-type vs two-type comparison
│
├── results/
│   ├── runs/                          # one subdir per experiment run
│   │   └── {run_id}/
│   │       ├── config.yaml
│   │       ├── metrics.json
│   │       ├── per_model_results.csv
│   │       └── checkpoints/
│   └── summary_table.csv             # auto-generated comparison table
│
└── tests/
    ├── test_parser.py
    ├── test_graph_builder.py
    ├── test_model_forward.py
    └── test_inference.py
```

---

## 5. Dataset Pipeline

### 5.1 MFCAD++ — primary training dataset

**Download:**
```bash
# scripts/download_data.sh
wget -O data/raw/mfcad++.zip \
    https://github.com/ZhangYikaii/MFCAD_plus_plus/releases/...
unzip data/raw/mfcad++.zip -d data/raw/mfcad++/
```

**Structure after download:**
```
data/raw/mfcad++/
├── step/          # .step files
├── labels/        # per-face label JSONs {face_id: feature_type}
└── metadata.csv   # model_id, num_faces, feature_types_present
```

**Feature types we care about (from MFCAD++ 24 types):**

| Priority | Type | MFCAD++ label | Day used |
|---|---|---|---|
| 1 (primary) | Counterbored hole | `counterbored_hole` | Day 2 |
| 2 (extensibility) | Through hole | `through_hole` | Day 4 |
| Reference only | Blind hole | `blind_hole` | Failure analysis |
| Reference only | Rectangular pocket | `rectangular_pocket` | Failure analysis |

### 5.2 Data splits

```python
# scripts/preprocess.py  — deterministic split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
```

Stratified by feature type presence to ensure all types appear in all splits.

### 5.3 Per-face feature vector schema

This is the exact dictionary structure pythonocc generates per face:

```python
# src/parsing/feature_extractor.py

FaceFeatures = {
    # Surface geometry (5 dims)
    "surface_type": int,        # enum: PLANE=0, CYLINDER=1, CONE=2, SPHERE=3, TORUS=4, OTHER=5
    "area": float,              # normalized by model bounding box diagonal squared
    "normal_x": float,          # face normal (for planar faces), 0 for others
    "normal_y": float,
    "normal_z": float,

    # Cylinder-specific (2 dims, 0 if not cylinder)
    "cylinder_radius": float,   # normalized by bounding box diagonal
    "cylinder_axis_z": float,   # axis direction z-component (orientation hint)

    # Topological context (2 dims)
    "num_adjacent_faces": int,  # degree in adjacency graph
    "num_boundary_edges": int,  # number of edges bounding this face

    # Total: 9-dimensional node feature vector
}

EdgeFeatures = {
    # Geometric relationship between two adjacent faces
    "dihedral_angle": float,    # angle between face normals at shared edge (radians)
    "convexity": int,           # CONVEX=1, CONCAVE=-1, SMOOTH=0
    "edge_length": float,       # normalized

    # Total: 3-dimensional edge feature vector
}
```

### 5.4 PyG Data object schema

```python
# src/parsing/graph_builder.py
from torch_geometric.data import Data

Data(
    x           = torch.FloatTensor,   # [num_faces, 9]  node features
    edge_index  = torch.LongTensor,    # [2, num_edges]  adjacency (undirected)
    edge_attr   = torch.FloatTensor,   # [num_edges, 3]  edge features
    y           = torch.LongTensor,    # [num_faces]     segmentation labels
    face_ids    = list[int],           # original face IDs (for output mapping)
    occ_face_ids= list[str],           # OCC face IDs from pythonocc
    model_id    = str,                 # MFCAD++ model identifier
)
```

### 5.5 Preprocessing script

```bash
python scripts/preprocess.py \
    --input_dir data/raw/mfcad++/step \
    --label_dir data/raw/mfcad++/labels \
    --output_dir data/processed \
    --feature_types counterbored_hole through_hole \
    --num_workers 4
```

Outputs one `.pt` file per STEP model. Failed parses (malformed STEP) are logged to
`data/processed/failed_parses.txt` and skipped gracefully.

---

## 6. B-Rep Graph Construction

### 6.1 pythonocc STEP parsing

```python
# src/parsing/step_parser.py

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface

class STEPParser:
    def parse(self, step_file: str) -> dict:
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_file)
        if status != 1:
            raise ValueError(f"Failed to read {step_file}")
        reader.TransferRoots()
        shape = reader.OneShape()
        return self._extract_brep(shape)

    def _extract_brep(self, shape) -> dict:
        faces = self._enumerate_faces(shape)
        adjacency = self._build_adjacency(shape, faces)
        return {"faces": faces, "adjacency": adjacency}

    def _enumerate_faces(self, shape) -> list[dict]:
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = []
        face_idx = 0
        while explorer.More():
            face = explorer.Current()
            adaptor = BRepAdaptor_Surface(face)
            surface_type = adaptor.GetType()
            face_data = {
                "face_id": face_idx,
                "occ_face_id": face.HashCode(10**9),  # OCC hash as ID
                "surface_type": int(surface_type),
                "area": self._compute_area(face),
                "normal": self._compute_normal(face, adaptor),
                "cylinder_radius": self._get_radius(adaptor) if surface_type == GeomAbs_Cylinder else 0.0,
            }
            faces.append(face_data)
            face_idx += 1
            explorer.Next()
        return faces

    def _build_adjacency(self, shape, faces) -> list[tuple]:
        # Two faces are adjacent if they share an edge
        # Returns list of (face_i, face_j, dihedral_angle, convexity)
        ...
```

### 6.2 Handling parse failures

```python
# Always wrap in try-except, log failures, never crash the pipeline
try:
    graph = parser.parse(step_file)
except Exception as e:
    logger.warning(f"SKIP {step_file}: {e}")
    failed_files.append(step_file)
    continue
```

---

## 7. Model Architecture

### 7.1 GraphSAGE Encoder

```python
# src/models/encoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm

class BRepEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_proj(x).relu()
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = x.relu()
            x = self.dropout(x)
        return self.out_proj(x)   # [num_faces, out_dim]
```

**Why GraphSAGE over GCN:**
- Inductive: can generalize to new graph structures at inference (new STEP files)
- Neighborhood sampling: memory-efficient on Colab
- Proven on B-Rep tasks (cited in AAGNet comparisons)

### 7.2 Segmentation Head

```python
# src/models/seg_head.py

class SegmentationHead(nn.Module):
    def __init__(self, in_dim: int = 64, num_classes: int = 25):
        # 24 MFCAD++ feature types + 1 background class
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.mlp(x)   # [num_faces, num_classes] logits
```

### 7.3 Subgraph Pooling

```python
# src/models/pooling.py

class SubgraphPooling(nn.Module):
    """
    Given per-face embeddings and a mask of faces in a subgraph,
    return a single embedding for the subgraph.
    Mode: 'mean' (fast), 'attention' (better, slightly more compute)
    """
    def __init__(self, dim: int = 64, mode: str = "attention"):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.attn = nn.Linear(dim, 1)

    def forward(self, face_embeddings, subgraph_mask):
        # face_embeddings: [num_faces, dim]
        # subgraph_mask:   [num_faces] bool
        sub = face_embeddings[subgraph_mask]   # [k, dim]
        if self.mode == "mean":
            return sub.mean(dim=0)
        else:
            weights = self.attn(sub).softmax(dim=0)   # [k, 1]
            return (weights * sub).sum(dim=0)          # [dim]
```

### 7.4 Full Model

```python
# src/models/feature_recognizer.py

class FeatureRecognizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder   = BRepEncoder(**config.encoder)
        self.seg_head  = SegmentationHead(**config.seg_head)
        self.pooling   = SubgraphPooling(**config.pooling)

    def forward(self, data):
        face_emb = self.encoder(data.x, data.edge_index, data.edge_attr)
        seg_logits = self.seg_head(face_emb)
        return face_emb, seg_logits

    def embed_subgraph(self, data, subgraph_mask):
        face_emb, _ = self.forward(data)
        return self.pooling(face_emb, subgraph_mask)
```

---

## 8. Training Pipeline

### 8.1 Hybrid Loss

```python
# src/losses/hybrid.py

import torch
import torch.nn.functional as F
from .contrastive import NTXentLoss

class HybridLoss(nn.Module):
    def __init__(self, seg_weight=1.0, contrastive_weight=0.5, temperature=0.07):
        super().__init__()
        self.seg_weight = seg_weight
        self.contrastive_weight = contrastive_weight
        self.ntxent = NTXentLoss(temperature)

    def forward(self, seg_logits, seg_labels, anchor_emb, positive_emb, negative_emb):
        # Segmentation loss: per-face cross-entropy
        seg_loss = F.cross_entropy(seg_logits, seg_labels, ignore_index=-1)

        # Contrastive loss: push same-type subgraph embeddings together
        contrastive_loss = self.ntxent(anchor_emb, positive_emb, negative_emb)

        total = self.seg_weight * seg_loss + self.contrastive_weight * contrastive_loss
        return total, {"seg": seg_loss.item(), "contrastive": contrastive_loss.item()}
```

```python
# src/losses/contrastive.py

class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy)"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: [batch, dim] — subgraph embeddings
        anchor   = F.normalize(anchor,   dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_sim = (anchor * positive).sum(-1) / self.temp   # [batch]
        neg_sim = (anchor * negative).sum(-1) / self.temp   # [batch]

        logits  = torch.stack([pos_sim, neg_sim], dim=1)    # [batch, 2]
        labels  = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)
```

### 8.2 Triplet construction

For each training step, construct triplets:
- **Anchor**: subgraph of a counterbored hole in model A
- **Positive**: subgraph of a counterbored hole in model B (different model, different size)
- **Negative**: subgraph of a through-hole or pocket in any model

Hard negative mining: after epoch 5, sample negatives that are closest in embedding space
(not random). This prevents the contrastive loss from saturating early.

### 8.3 Training script

```bash
python scripts/train.py \
    --config configs/counterbored_hole.yaml \
    --seed 42 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_dim 128 \
    --out_dim 64 \
    --num_layers 3 \
    --seg_weight 1.0 \
    --contrastive_weight 0.5 \
    --wandb_project hanomi-feature-recognition \
    --output_dir results/runs/run_001
```

### 8.4 Training config YAML

```yaml
# configs/counterbored_hole.yaml
feature_types: [counterbored_hole]
epochs: 50
batch_size: 32
lr: 1.0e-3
weight_decay: 1.0e-5
scheduler: cosine
warmup_epochs: 5

encoder:
  node_in_dim: 9
  edge_in_dim: 3
  hidden_dim: 128
  out_dim: 64
  num_layers: 3
  dropout: 0.1

seg_head:
  in_dim: 64
  num_classes: 25          # 24 types + background

pooling:
  dim: 64
  mode: attention

loss:
  seg_weight: 1.0
  contrastive_weight: 0.5
  temperature: 0.07

data:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

---

## 9. Inference Pipeline

### 9.1 Input format

```python
# The assignment provides:
reference = {
    "step_file": "path/to/reference.step",
    "face_ids": [3, 4, 5],               # integer face IDs
    "occ_face_ids": ["#123", "#124", "#125"]  # OCC hash IDs
}
query_files = ["query1.step", "query2.step", ...]
```

### 9.2 Seed-and-expand algorithm

```python
# src/inference/seed_expand.py

def find_feature_instances(
    model: FeatureRecognizer,
    reference_graph: Data,
    reference_mask: BoolTensor,     # which faces = labeled feature
    query_graph: Data,
    tau_seed: float = 0.6,          # similarity threshold to become a seed
    tau_expand: float = 0.4,        # threshold to include a neighbor
    tau_confidence: float = 0.5,    # final cluster threshold
) -> list[dict]:

    # 1. Embed reference feature
    ref_emb = model.embed_subgraph(reference_graph, reference_mask)  # [dim]

    # 2. Embed all faces in query model
    query_face_emb, _ = model(query_graph)  # [num_faces, dim]

    # 3. Per-face similarity to reference embedding
    ref_emb_norm = F.normalize(ref_emb.unsqueeze(0), dim=-1)
    query_emb_norm = F.normalize(query_face_emb, dim=-1)
    face_sims = (query_emb_norm @ ref_emb_norm.T).squeeze()  # [num_faces]

    # 4. Find seed faces
    seed_faces = (face_sims > tau_seed).nonzero().squeeze()

    # 5. BFS expand from each seed
    adj = build_adjacency_dict(query_graph.edge_index)
    clusters = []
    for seed in seed_faces:
        cluster = bfs_expand(seed, adj, query_face_emb, ref_emb, tau_expand)
        clusters.append(cluster)

    # 6. Pool each cluster, compute confidence
    instances = []
    for cluster_faces in clusters:
        cluster_mask = torch.zeros(query_graph.num_nodes, dtype=torch.bool)
        cluster_mask[cluster_faces] = True
        cluster_emb = model.pooling(query_face_emb, cluster_mask)
        confidence = F.cosine_similarity(
            cluster_emb.unsqueeze(0), ref_emb.unsqueeze(0)
        ).item()
        if confidence > tau_confidence:
            instances.append({
                "face_ids": [query_graph.face_ids[i] for i in cluster_faces],
                "occ_face_ids": [query_graph.occ_face_ids[i] for i in cluster_faces],
                "confidence": round(confidence, 4),
            })

    # 7. NMS: remove duplicate clusters
    instances = non_max_suppression(instances, iou_threshold=0.5)
    return sorted(instances, key=lambda x: -x["confidence"])
```

### 9.3 Output format

```json
{
  "query_model": "query1.step",
  "reference_feature": "counterbored_hole",
  "instances": [
    {
      "face_ids": [12, 13, 14],
      "occ_face_ids": ["#456", "#457", "#458"],
      "confidence": 0.921
    },
    {
      "face_ids": [27, 28, 29],
      "occ_face_ids": ["#512", "#513", "#514"],
      "confidence": 0.876
    }
  ]
}
```

### 9.4 Demo script

```bash
python scripts/inference_demo.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --reference_step data/raw/test/ref_model.step \
    --reference_face_ids 3 4 5 \
    --query_dir data/raw/test/query_models/ \
    --output results/inference_demo_output.json
```

---

## 10. Baselines

### 10.1 Baseline 0: Rule-Based Heuristic

A counterbored hole has a specific topological signature:
```
[cylinder, large radius]
    → connected via concave edge to →
[flat annular face]
    → connected via concave edge to →
[cylinder, small radius]
```

```python
# src/baselines/rule_based.py

COUNTERBORED_HOLE_PATTERN = [
    {"surface_type": CYLINDER, "radius": "large"},
    {"edge": "concave"},
    {"surface_type": PLANE, "shape": "annular"},
    {"edge": "concave"},
    {"surface_type": CYLINDER, "radius": "small"},
]

def match_counterbored_hole(graph: Data) -> list[dict]:
    instances = []
    for face_idx in range(graph.num_nodes):
        if graph.x[face_idx, SURFACE_TYPE] == CYLINDER:
            # Try to grow pattern from this seed
            match = try_grow_pattern(face_idx, graph, COUNTERBORED_HOLE_PATTERN)
            if match:
                instances.append(match)
    return instances
```

**When it works:** clean models, no feature intersections, canonical orientation.
**When it breaks:** features intersecting a pocket wall, non-standard topology, blended edges.
This failure mode analysis is valuable — it motivates the GNN approach.

### 10.2 Baseline 1: LLM over Structured Serialization

Serialize the local neighborhood (2-hop subgraph around seed faces) into a JSON string.
Pass to Claude / GPT with a prompt describing the reference feature. Ask it to classify
whether this subgraph matches the reference feature type.

```python
# src/baselines/llm_baseline.py

SYSTEM_PROMPT = """
You are a CAD feature recognition system. Given a JSON description of a set of
connected faces from a CAD model, determine if they represent a {feature_type}.

A counterbored hole consists of:
- A large cylinder (outer bore) connected via concave edges to a flat annular plane
- The annular plane connected via concave edges to a smaller cylinder (inner bore)
- Size, orientation, and surrounding geometry are irrelevant to the classification.

Respond with JSON: {"is_match": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
"""

def serialize_subgraph(graph: Data, face_indices: list[int]) -> str:
    faces = []
    for idx in face_indices:
        faces.append({
            "face_id": graph.face_ids[idx],
            "surface_type": SURFACE_TYPE_NAMES[graph.x[idx, 0].int().item()],
            "area": round(graph.x[idx, 1].item(), 4),
            "adjacent_faces": get_neighbors(graph, idx),
            "dihedral_angles": get_dihedral_angles(graph, idx),
        })
    return json.dumps({"faces": faces}, indent=2)
```

**Token cost analysis (required by assignment):**

| Model | Tokens per subgraph | Cost per query | Cost for 1000 queries |
|---|---|---|---|
| GPT-4o | ~800 tokens | ~$0.003 | ~$3.00 |
| Claude Sonnet | ~800 tokens | ~$0.002 | ~$2.00 |
| GPT-3.5-turbo | ~800 tokens | ~$0.0003 | ~$0.30 |

**Context overflow strategy:** For models with many candidate subgraphs, we enumerate
candidate seed faces first (per-face similarity > τ), serialize only 2-hop neighborhoods
of top-K seeds, never the full model. A model with 500 faces and 5 candidate seeds
→ ~5 × 800 = 4000 tokens, well within any context window. For pathologically large
models (>2000 faces, >50 seeds), chunk into batches of 10 seeds, aggregate results.

---

## 11. Extensibility Demo

### 11.1 Phase 1: Train on counterbored holes only

```bash
python scripts/train.py \
    --config configs/counterbored_hole.yaml \
    --output_dir results/runs/phase1_counterbored_hole
```

Record: val F1, precision, recall, inference time per model.

### 11.2 Phase 2: Add through-holes

```yaml
# configs/extensibility_v2.yaml
feature_types: [counterbored_hole, through_hole]
# Everything else identical to base config
```

```bash
python scripts/train.py \
    --config configs/extensibility_v2.yaml \
    --checkpoint results/runs/phase1_counterbored_hole/checkpoints/best.pt  # fine-tune
    --output_dir results/runs/phase2_two_types
```

**What to measure and report:**

| Metric | Phase 1 (1 type) | Phase 2 (2 types) | Delta |
|---|---|---|---|
| Counterbored hole F1 | ? | ? | ? |
| Through hole F1 | N/A | ? | — |
| Training time (epochs to convergence) | ? | ? | ? |
| Inference time per model (ms) | ? | ? | ? |
| Embedding space separation (mean inter-class distance) | ? | ? | ? |

**Expected findings:**
- Counterbored hole F1 drops slightly (shared embedding space, harder negatives)
- Through hole F1 is high if similar topology, lower if dissimilar
- Inference time unchanged (architecture identical)
- Training time increases ~20% (more triplets to mine)

### 11.3 What breaks at the 10th type

This is a key walkthrough question. Prepare this answer explicitly:

1. **Embedding space crowding:** With 10+ types sharing a 64-dim space, inter-class
   separation decreases. Fix: increase embedding dimension, add projection heads per class.
2. **Hard negative mining degrades:** Types that are geometrically similar (through-hole
   vs. blind-hole) create very hard negatives that destabilize training early.
   Fix: curriculum learning — easy negatives first, hard negatives after epoch 10.
3. **Class imbalance:** MFCAD++ has ~3× more through-holes than keyways. Fix: weighted
   sampling, per-class loss weighting.
4. **Annotation bottleneck:** Each new type needs hundreds of labeled instances.
   Fix: this is where the self-supervised pretrained encoder (BRepMAE on ABC) becomes
   critical — fewer labeled examples needed per new type.

### 11.4 What breaks at the 50th type

1. **Architecture is still fine** — the GNN encoder is shared and learns geometry-agnostic
   face embeddings. No architectural change needed.
2. **Inference NMS becomes expensive** — 50 feature types × N candidate clusters ×
   IoU computation = O(50N²) in the worst case. Fix: parallel cluster evaluation,
   early pruning.
3. **Confusion between subtypes** — at 50 types you start having very similar features
   (step hole vs. counterbored hole vs. deep counterbored hole). Fix: hierarchical
   classification (coarse type → fine subtype), or prototype networks.
4. **Rare types with few examples** — for 50 types in MFCAD++ we only have 24, so
   synthetic generation or data augmentation (mirroring, scaling) becomes necessary.

---

## 12. Evaluation & Benchmarking

### 12.1 Metrics

**At face level:**
```
Precision = |predicted_faces ∩ gt_faces| / |predicted_faces|
Recall    = |predicted_faces ∩ gt_faces| / |gt_faces|
F1        = 2 × Precision × Recall / (Precision + Recall)
```

**At instance level:**
```
IoU per instance = |predicted_faces ∩ gt_faces| / |predicted_faces ∪ gt_faces|
Instance match: IoU > 0.5 (standard detection threshold)
AP (Average Precision) across confidence thresholds
```

### 12.2 Ablation matrix

Run every combination, report face-level F1 on counterbored hole:

| Experiment | Node features | Edge features | Contrastive loss | Expected F1 |
|---|---|---|---|---|
| A (full model) | ✅ | ✅ | ✅ | Best |
| B | ✅ | ✅ | ❌ | Lower |
| C | ✅ | ❌ | ✅ | Lower |
| D (geometry only) | ✅ | ❌ | ❌ | Baseline |
| E (topology only) | surface_type only | ✅ | ✅ | Test |

### 12.3 Benchmark comparison table

| Method | CB Hole F1 | Through Hole F1 | Inference (ms/model) | Notes |
|---|---|---|---|---|
| Rule-based (Baseline 0) | ? | ? | < 10 | Fails on intersections |
| LLM (Baseline 1) | ? | ? | 500–2000 | Expensive, interpretable |
| GNN seg only (Ablation B) | ? | ? | ? | No metric learning |
| **GNN + contrastive (Ours)** | ? | ? | ? | **Primary** |

### 12.4 Compute budget

**Training (Colab T4):**
- MFCAD++ preprocessing: ~30 min (one-time)
- 50 epochs on counterbored holes: ~45 min
- Phase 2 fine-tuning: ~30 min

**Inference:**
- GNN forward pass: O(N) in faces, ~5–15ms per model (Colab T4)
- Seed-and-expand: ~10–30ms per model depending on num seeds
- LLM baseline: 500–2000ms per model (API latency)

**Production cost estimate (bonus):**

| Scale | GPU-seconds/query | $/query (A10G @ $0.002/s) |
|---|---|---|
| Single STEP, 200 faces | 0.025s | $0.00005 |
| 1M queries/day | 25,000 GPU-seconds | $50/day |
| LLM baseline | N/A | $2–3/query |

**Conclusion:** GNN approach is 40,000× cheaper than LLM at production scale.

---

## 13. Programmatic Results Logging

The assignment explicitly requires a "programmatic way of documenting results." This is
a first-class deliverable, not an afterthought.

### 13.1 Results logger

```python
# src/evaluation/results_logger.py

import json, csv, datetime
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class ModelResult:
    run_id: str
    model_file: str
    feature_type: str
    method: str                   # "gnn", "rule_based", "llm"
    predicted_instances: list
    gt_instances: list
    precision: float
    recall: float
    f1: float
    inference_ms: float
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.datetime.now().isoformat()

class ResultsLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def log(self, result: ModelResult):
        self.results.append(result)

    def save(self):
        # JSON — full detail
        with open(self.run_dir / "per_model_results.json", "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # CSV — summary table
        with open(self.run_dir / "per_model_results.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

        # Aggregate metrics
        agg = self._aggregate()
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(agg, f, indent=2)

        print(f"Results saved to {self.run_dir}")
        return agg

    def _aggregate(self) -> dict:
        import numpy as np
        return {
            "mean_f1": np.mean([r.f1 for r in self.results]),
            "mean_precision": np.mean([r.precision for r in self.results]),
            "mean_recall": np.mean([r.recall for r in self.results]),
            "mean_inference_ms": np.mean([r.inference_ms for r in self.results]),
            "n_models": len(self.results),
        }
```

### 13.2 Evaluation script

```bash
# Runs inference on the entire test set, logs all results
python scripts/evaluate.py \
    --checkpoint results/runs/run_001/checkpoints/best.pt \
    --test_dir data/processed/test \
    --methods gnn rule_based llm \
    --output_dir results/runs/run_001 \
    --feature_types counterbored_hole through_hole
```

### 13.3 Summary table auto-generation

```bash
# Compares all runs in results/runs/ and writes results/summary_table.csv
python scripts/compare_runs.py --results_dir results/runs/
```

Output:

```
run_id             | feature_type      | method      | F1    | P     | R     | ms/model
-------------------|-------------------|-------------|-------|-------|-------|----------
run_001            | counterbored_hole | gnn         | 0.89  | 0.91  | 0.87  | 18.3
run_001            | counterbored_hole | rule_based  | 0.71  | 0.88  | 0.59  | 4.1
run_001            | counterbored_hole | llm         | 0.78  | 0.82  | 0.74  | 1240.0
run_002_two_types  | counterbored_hole | gnn         | 0.86  | 0.89  | 0.83  | 18.9
run_002_two_types  | through_hole      | gnn         | 0.84  | 0.87  | 0.81  | 18.9
```

---

## 14. Bonus Sections

### 14.1 Domain retrofitting: Molecular Similarity

**The analogy:**

| Molecular similarity | B-Rep feature matching |
|---|---|
| Atoms | Faces |
| Bonds | Shared edges |
| Functional groups | Machining features |
| Molecular fingerprint | Feature subgraph embedding |
| Tanimoto similarity | Cosine similarity of embeddings |

Molecular GNNs (e.g., MPNN from "Neural Message Passing for Quantum Chemistry") solve
exactly the same problem: given a labeled functional group in one molecule, find it in
others regardless of surrounding molecular context. The architecture we propose is
literally a molecular GNN applied to B-Rep topology.

**Trade-offs (honest):**
- ✅ Transfer: a pretrained molecular GNN encoder could be fine-tuned on B-Rep data with
  minimal architectural change. Node features map cleanly (atom type → surface type,
  bond type → edge convexity).
- ❌ Geometry: molecular GNNs use 3D Euclidean coordinates as node features. B-Rep faces
  don't have a single coordinate — they have a surface with extent. UV-Net's approach
  (sample a UV grid on the surface) is closer to what molecular GNNs do, but is heavier.
- ❌ Scale: protein contact graphs have O(100) nodes. CAD models can have O(1000) faces.
  Molecular GNN architectures don't scale as gracefully.

**Verdict:** Molecular GNN is the closest analogue and would be the first cross-domain
transfer to try. We use the same loss (NT-Xent) and architecture (MPNN/GraphSAGE).
The difference is in the node feature design, where B-Rep requires geometry-aware features
that molecular GNNs don't need.

### 14.2 Self-supervised path (described)

**BRepMAE approach on ABC dataset:**

1. Download ABC dataset (1M STEP files, no feature labels)
2. Parse each model → B-Rep graph
3. Randomly mask 30% of face attributes (surface type, area, radius)
4. Train GraphSAGE encoder to reconstruct masked attributes from topological context
5. Fine-tune on MFCAD++ with supervised segmentation + contrastive loss

**Expected benefit:** Encoder learns geometry-agnostic, topology-aware representations
before seeing any labels. Fine-tuning on MFCAD++ should reach same F1 with 5–10× fewer
labeled examples. This is the answer to "what does self-supervised look like" in the
walkthrough.

### 14.3 Failure mode analysis

| Scenario | Failure mode | Severity | Fix |
|---|---|---|---|
| Feature intersects pocket | Adjacency graph breaks — faces are shared | High | AAGNet-style continuous attributes over topology |
| Highly curved body (turbine blade) | Normal and radius features lose meaning | Medium | Add Gaussian curvature as node feature |
| Very small feature (M2 hole) | Faces below area threshold, clipped | Medium | Adaptive normalization per model bounding box |
| Symmetric model (4 identical holes) | Model finds all 4 → correct, but NMS may merge | Low | Tune IoU threshold |
| Blended/filleted edges | Dihedral angle → π, convexity ambiguous | Medium | Add a "smooth" edge class, don't rely on convexity alone |
| Counterbored hole with added boss | Extra adjacent face breaks pattern matching | High | GNN handles this; rule-based fails entirely |
| STEP file with degenerate faces | Area = 0, parser exception | Low | Skip degenerate faces in preprocessing |

---

## 15. Day-by-Day Schedule

### Day 1 — Parsing, Graph Construction, Dataset (6–7 hrs)

**Morning (3 hrs):**
- [ ] Set up conda environment, verify pythonocc-core installs
- [ ] Download MFCAD++ dataset
- [ ] Study BRepNet open-source repo — extract their STEP→face parsing boilerplate
- [ ] Implement `STEPParser` in `src/parsing/step_parser.py`
- [ ] Unit test: parse 5 STEP files, verify face counts match expected

**Afternoon (4 hrs):**
- [ ] Implement `FeatureExtractor` — extract 9-dim node features per face
- [ ] Implement `GraphBuilder` — build PyG Data object with edge_index, edge_attr
- [ ] Implement `MFCADDataset` — wraps processed .pt files
- [ ] Run `scripts/preprocess.py` on full MFCAD++ — log failed parses
- [ ] Notebook 01: visualize a few B-Rep graphs, sanity-check node features
- [ ] Write tests: `test_parser.py`, `test_graph_builder.py`

**Day 1 checkpoint:** Can parse any STEP file into a PyG Data object with correct face
features and adjacency. Dataset of processed .pt files ready for training.

---

### Day 2 — Model, Training, Initial Results (7–8 hrs)

**Morning (4 hrs):**
- [ ] Implement `BRepEncoder` (GraphSAGE), `SegmentationHead`, `SubgraphPooling`
- [ ] Implement `HybridLoss` — segmentation cross-entropy + NT-Xent contrastive
- [ ] Implement triplet construction and hard negative mining
- [ ] `test_model_forward.py`: verify forward pass shapes, gradients flow

**Afternoon (4 hrs):**
- [ ] Implement `scripts/train.py` with wandb logging
- [ ] Train on counterbored holes only — 50 epochs, Colab T4
- [ ] Watch training curves — diagnose if seg loss or contrastive loss dominates
- [ ] Tune `seg_weight` and `contrastive_weight` if needed
- [ ] Save best checkpoint by val F1

**Day 2 checkpoint:** Trained model on counterbored holes. Val F1 > 0.75 (rough target).
Training curves logged to wandb.

---

### Day 3 — Inference Pipeline + Baselines (6–7 hrs)

**Morning (3 hrs):**
- [ ] Implement `seed_expand.py` — full seed-and-expand algorithm
- [ ] Implement `nms.py` — non-max suppression on overlapping face clusters
- [ ] Implement `inference_demo.py` — end-to-end: reference STEP + face IDs → JSON output
- [ ] `test_inference.py` — verify output format, confidence scores in [0,1]

**Afternoon (4 hrs):**
- [ ] Implement `rule_based.py` — topological pattern matcher (Baseline 0)
- [ ] Implement `llm_baseline.py` — subgraph serialization + API call (Baseline 1)
- [ ] Document LLM: token count per query, cost table, overflow handling
- [ ] Run all three methods on test set
- [ ] Implement `ResultsLogger` — save per_model_results.json + .csv
- [ ] Run `scripts/evaluate.py` — generate benchmark comparison table

**Day 3 checkpoint:** Full inference pipeline working end-to-end. Benchmark table with
3 methods (rule-based, LLM, GNN) on counterbored holes.

---

### Day 4 — Extensibility Demo + Ablations (6–7 hrs)

**Morning (3 hrs):**
- [ ] Train Phase 2: add through-holes, fine-tune from Phase 1 checkpoint
- [ ] Record: what changed in data, training loss, val F1 for both types
- [ ] Measure embedding space separation (t-SNE visualization in Notebook 04)
- [ ] Document: Phase 1 vs Phase 2 comparison table

**Afternoon (4 hrs):**
- [ ] Run full ablation matrix (5 experiments — see §12.2)
- [ ] Record F1 per ablation, generate ablation table
- [ ] Write "What breaks at 10th/50th type" section — concrete numbered analysis
- [ ] Run on Fusion 360 Gallery subset (~20 models) — generalization test
- [ ] Document failure modes (table from §14.3)
- [ ] Cost analysis: GPU-seconds/query, $/query at production scale

**Day 4 checkpoint:** Extensibility demo complete. Ablation table complete. Failure mode
analysis complete.

---

### Day 5 — Write-up, Repo Polish, README (5–6 hrs)

**Morning (3 hrs):**
- [ ] Write README.md — setup, run instructions, architecture overview, results table
- [ ] Write domain retrofitting section (molecular similarity analogy)
- [ ] Write self-supervised path section (BRepMAE / ABC)
- [ ] Repo cleanup: remove debug prints, ensure all scripts have `--help`
- [ ] Verify `pytest` passes all tests
- [ ] Verify `requirements.txt` and `environment.yml` are correct

**Afternoon (3 hrs):**
- [ ] Final end-to-end run from scratch (pretend you're the reviewer)
- [ ] Notebook 03: qualitative visualizations of matched features
- [ ] Notebook 04: extensibility demo visualization
- [ ] Push to private GitHub repo
- [ ] Prepare 45-min walkthrough talking points (see §17)

---

## 16. Write-Up Guide

Structure the write-up to match the evaluation criteria order exactly:

### Section 1: Problem Framing (1 page)
- Why this is metric learning on B-Rep subgraphs, not classification
- The invariances (size, orientation, surrounding geometry) and how each is handled

### Section 2: Representation Choice (1 page)
- B-Rep vs alternatives table
- Face-level graph vs coedge-level: why face-level for this scope
- Exact node/edge feature schema with justification for each feature

### Section 3: Architecture (1 page)
- GraphSAGE encoder: why inductive, why not GCN
- Hybrid loss: why both segmentation and contrastive
- Seed-and-expand inference: why not brute-force

### Section 4: Baselines & Benchmarking (1 page)
- Benchmark table (3 methods × 2 feature types × key metrics)
- Ablation table (5 experiments)
- Interpretation: what each result tells us

### Section 5: Extensibility Demo (1 page)
- Phase 1 → Phase 2 delta table
- What breaks at 10th type (numbered analysis)
- What breaks at 50th type (numbered analysis)

### Section 6: Bonus (0.5 pages)
- Molecular similarity analogy + honest trade-offs
- BRepMAE self-supervised path
- Cost analysis table
- Failure mode table

---

## 17. 45-Min Walkthrough Prep

Hanomi will push on your choices. Prepare direct answers to these:

**"Why GraphSAGE and not GCN?"**
> GCN is transductive — it can't generalize to graph structures not seen during training.
> GraphSAGE is inductive, generating embeddings for new nodes via learned aggregation.
> New STEP files at inference are always new graphs. GCN would require retraining.

**"Why not UV-Net?"**
> UV-Net samples dense UV grids on each surface — more expressive but O(N×UV²) memory.
> On Colab T4 with 16GB, models with 500+ faces would OOM. Face-level AAG gets us ~80%
> of the expressiveness at 10% of the memory. UV-Net is the right next step for V2.

**"Why not point clouds?"**
> Point clouds lose topology. Two faces sharing an edge is a topological fact that tells
> you they're adjacent features — you can't recover this from a point cloud without
> expensive graph reconstruction. B-Rep gives this for free.

**"What happens if someone queries a feature type you've never trained on?"**
> The encoder still generates face embeddings. The reference subgraph still gets pooled
> into an embedding. Cosine similarity still runs. The model will find the
> most topologically similar subgraphs in the query — whether that's the right answer
> depends on whether similar geometry = same type. This is the zero-shot case and it
> partially works. The self-supervised pretrained encoder (BRepMAE) improves this.

**"Why contrastive over a simple classification head?"**
> A classification head requires knowing all feature types at training time. Contrastive
> learning produces a geometry embedding space where new types naturally find their own
> cluster. This is exactly what "adding a new type is a data problem not an architecture
> problem" means. You don't retrain the encoder — you add labeled examples and they
> populate a new region in embedding space.

**"What breaks at 50 feature types?"**
> Four things: embedding space crowding (fix: increase dim), hard negative instability
> between similar types (fix: curriculum learning), class imbalance (fix: weighted
> sampling), and annotation bottleneck (fix: BRepMAE self-supervised pretraining to
> reduce labeled examples needed per type).

**"Your LLM baseline — what happens when the model has 2000 faces?"**
> We never serialize the full model. We seed-first: per-face similarity scores identify
> the top-K candidate subgraphs (2-hop neighborhoods), serialize only those. A 2000-face
> model with 10 candidate seeds → ~8000 tokens per query. Still within context. If we
> get >50 seeds (pathological case), we batch them and run 5 LLM calls instead of 1.

**"Why MFCAD++ over synthetic generation?"**
> MFCAD++ has 24 labeled feature types, thousands of models, already parsed by the
> community, compatible with pythonocc. Synthetic generation would require designing
> a parametric CAD generator, validating the STEP output, and ensuring diversity.
> MFCAD++ is the de facto benchmark — using it also lets us compare our numbers
> to published results from BRepNet/AAGNet.

---

*Plan version: 1.0 | Author: Claude | For: Hanomi ML Engineer Take-Home*
