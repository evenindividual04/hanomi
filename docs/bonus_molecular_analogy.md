# Cross-Domain Analogy: CAD Feature Recognition ≈ Molecular Similarity

## The Parallel

| CAD Feature Recognition | Molecular Similarity | Conceptual Mapping |
|---|---|---|
| B-Rep face graph | Molecular graph | Nodes = atoms/edges = bonds vs nodes = faces/edges = adjacency |
| Surface type (plane, cylinder, etc.) | Atom type (C, N, O, etc.) | Categorical node features |
| Edge convexity (concave, convex, smooth) | Bond type (single, double, aromatic) | Categorical edge features |
| Feature subgraph (e.g., counterbored hole) | Functional group (e.g., carboxyl) | Repeating topological pattern |
| Metric learning on subgraph embeddings | Graph kernel / GNN on molecular graphs | Same learning paradigm |
| Extensibility to new feature types | Generalization to new molecules | Inductive representation learning |

## Concretely: Counterbored Hole ≈ Carboxyl Group

**Counterbored hole pattern:**
```
[cylinder] --concave--> [annular plane] --concave--> [smaller cylinder]
```

**Carboxyl group pattern:**
```
[Carbon] --single--> [Oxygen] --double--> [Carbon-OH]
```

Both are:
- 3-node connected subgraphs
- Specific edge type sequences (concave transitions)
- Semantic units that appear in larger structures
- Recognizable by local topology, not global context

## What Works Across Domains

### 1. Message Passing GNNs
**Molecular:** GraphSAGE, GCN aggregate atom features across bonds
**CAD:** GINEConv aggregates face features across adjacency edges with edge attributes
**Why:** Both learn inductive representations from local neighborhoods
**Trade-off:** CAD uses edge features (convexity, dihedral angle) which molecular GNNs typically ignore

### 2. Edge-Aware Convolution
**Molecular:** Bond types (single/double/triple) influence atom embeddings
**CAD:** Edge convexity (concave/convex/smooth) distinguishes feature boundaries
**Why:** Edge semantics are critical in both domains for detecting boundaries
**Trade-off:** Molecular bonds are discrete categories (4-5 types), CAD edges are continuous with geometric meaning (dihedral angle in radians)

### 3. Contrastive Learning on Subgraphs
**Molecular:** SimCLR on molecular graphs for drug discovery
**CAD:** Our approach on B-Rep graphs for feature recognition
**Why:** Both need to recognize the *same* pattern in different *contexts*
**Trade-off:** Both push same-type embeddings together and different-type embeddings apart

### 4. Inductive Representation Learning
**Molecular:** GNNs generalize to new molecules without retraining
**CAD:** GINEConv generalizes to new CAD models at inference
**Why:** Both learn reusable representations that transfer to unseen examples
**Trade-off:** Neither domain requires training on all possible instances

## What Differs (and Why Trade-Offs Matter)

| Aspect | Molecular Domain | CAD Domain | Trade-Off Justification |
|---|---|---|---|
| **Node feature space** | ~10 atom types (discrete) | 5 surface types + continuous geometry | CAD needs richer features → **Accept complexity** |
| **Graph size** | 20-100 atoms per molecule | 20-500 faces per model | CAD graphs are larger → **Accept scalability** |
| **Symmetry** | Permutation invariance critical (atoms unlabeled) | Less critical (faces have spatial positions) | **Different inductive bias** |
| **3D geometry** | Often ignored in 2D molecular graphs | Fundamental to feature semantics | **CAD must preserve geometry** (no UV-Net trade-off) |
| **Training data scale** | Millions of molecules (PubChem, ChEMBL) | Thousands of CAD models (MFCAD++) | **Accept data constraint** |
| **Feature granularity** | Individual atoms | Whole faces as atomic units | **Different semantic level** (faces are coarser) |

## Honest Trade-Offs in Our CAD Approach

### Adopted from Molecular Domain
✅ **Graph neural network architecture** (GINEConv ≈ GraphSAGE for molecules)
✅ **Edge-aware message passing** (bond types → convexity)
✅ **Contrastive learning** (SimCLR-style → NT-Xent on subgraphs)
✅ **Inductive representation** (generalizes to new instances)
✅ **Learning paradigm** (recognize patterns in different contexts)

### Adapted for CAD Specifics
🔧 **Continuous node features** (area, centroid, radius) vs discrete atom types
🔧 **Geometric normal vectors** (orientation matters for holes) vs chemical properties
🔧 **3D spatial awareness** (faces have coordinates) vs 2D graphs
🔧 **Hierarchical semantics** (faces → subgraphs) vs atoms only
🔧 **Feature-level granularity** (whole faces as units) vs atoms only

### Sacrificed for 5-Day Timeline
❌ **UV-Net's absolute coordinate sampling** (requires implementing UV parameterization)
❌ **Attention over long-range interactions** (molecular pharmacophore modeling)
❌ **Equivariant networks** (molecular rotation invariance)
❌ **Self-supervised pretraining** (BRepMAE on ABC's 1M models ~72 hours)

## Takeaway

The molecular analogy validates our architectural choices: if GraphSAGE + contrastive learning works for drug discovery, the same paradigm should work for CAD feature recognition. The key adaptation is preserving geometric information that molecules discard—CAD features are defined by shape, not just topology.

This cross-domain perspective also clarifies the extensibility claim: just as molecular GNNs recognize novel functional groups by topology, our CAD GNN recognizes novel features by subgraph structure.

---

**Ablation C** (no edge features, F1=0.845 vs 0.855 full) partially validates this analogy: edge attributes contribute similarly to bond types in molecular GNNs — important for complex features, less decisive for topologically simple ones (through-holes).
