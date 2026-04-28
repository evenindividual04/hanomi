# Walkthrough Q&A Preparation

Crisp answers (2–4 sentences each) for anticipated walkthrough questions.

---

## Architecture

**Q: Why GINEConv over SAGEConv?**
SAGEConv aggregates only node features and drops `edge_attr` silently. GINEConv injects edge attributes into every message-passing step — this is the only way concavity flags (which mark feature boundaries) reach the model. Without GINEConv, the encoder is completely blind to the convex/concave edge signals that distinguish through-holes from background faces.

**Q: Why GINEConv over GCN?**
GCN is transductive — it computes fixed spectral Laplacians over the training graph. New STEP files at inference have entirely new graph structures, breaking the Laplacian. GINEConv/GIN is inductive: it generalizes to unseen topologies via learned neighbourhood aggregation, with no retraining needed per model.

**Q: Why not UV-Net?**
UV-Net uses 10×10×7 UV-grid samples per face — richer but requires OCC surface parameterization and 5-day implementation risk. Our 8-dim scalar features sidestep OCC at training time while being sufficient for through/blind-hole topological discrimination where curvature matters less. The known failure we accept: seam-split cylinders (a 360° cylinder becomes two 180° nodes in some kernels). V2 adds seam merge preprocessing.

**Q: Why PairwiseContrastiveLoss instead of NT-Xent?**
NT-Xent uses all in-batch samples as negatives — with ~8 triplets per batch, there are too few negatives to be stable and the batch × batch similarity matrix is expensive. Our PairwiseContrastiveLoss uses explicit (anchor, positive, negative) triplets where negatives are hard-mined from the closest embeddings after epoch 5. This is controllable and doesn't require large batch sizes.

**Q: What is τ=0.07 and why?**
Temperature in the contrastive softmax. Lower τ sharpens the hard negative emphasis — the positive pair must score significantly above all negatives. τ=0.07 is the SimCLR/MoCo empirical standard, validated across graph contrastive learning papers (SGNCL, SOLA-GCL). We adopt it directly; no ablation needed.

**Q: What is the SimCLR projection head and why use_proj=False at inference?**
The projection head (2-layer MLP) absorbs the contrastive uniformity cost during training. Representations *before* the head transfer better to downstream retrieval (this is the core SimCLR finding). At inference, `use_proj=False` — the head is bypassed, using the pre-projection representation directly for cosine similarity. Existing V1 checkpoints load without changes.

---

## Loss and Training

**Q: Why hybrid loss (CE + contrastive)?**
Segmentation CE alone produces a closed-set classifier — adding a new feature type requires retraining the head. It also doesn't directly optimize subgraph-level matching. Contrastive loss alone has weak supervision — face labels are plentiful but contrastive triplets are sparse. Together: CE gives dense per-face gradients, contrastive ensures same-type features cluster in embedding space regardless of model context.

**Q: Why not just cross-entropy on all 25 MFCAD++ classes?**
CE trains a fixed-output classifier. The model can never recognize a 26th feature type without retraining. Metric learning with contrastive loss creates a metric space where a new feature type can be matched by embedding its reference subgraph — no retrain, only new labeled examples.

---

## Inference

**Q: Why seed-and-expand instead of brute-force subgraph search?**
Brute-force subgraph enumeration is NP-complete. Seed-and-expand is O(N·K) — enumerate seeds by surface type (CPU), BFS-expand k hops (pruned by cosine similarity), pool once per candidate, NMS. A 2000-face model with 10 seeds evaluates ~100 candidates in ~12ms.

**Q: What happens when reference_surface_types is wrong or missing?**
If `reference_surface_types` is not provided, it's inferred from the reference mask: `{int(x[i,4]*11) for i in reference_faces}`. If that set is empty (empty mask), `get_heuristic_seeds()` raises a clear `ValueError`. The system fails fast rather than returning spurious results.

**Q: How does NMS work here?**
Non-max suppression: sort candidates by confidence descending, greedily keep each candidate if its face-IoU with all already-kept instances is < 0.5. This removes overlapping detections (e.g. two seeds that both expand to the same hole). Face-IoU not spatial-IoU — this works in graph space without 3D bounding boxes.

---

## Representation

**Q: Why B-Rep over mesh / point cloud?**
B-Rep gives exact 1:1 node-to-face mapping — output face IDs directly without post-hoc face assignment. Surface type + topology exactly encodes feature semantics: a through-hole is unambiguously CYLINDER→PLANE→CYLINDER topology (convex-concave-convex adjacency pattern), invisible in point clouds. Mesh/point cloud require post-processing to recover face IDs and lose the clean topological signal.

**Q: What does 'same feature' mean semantically?**
Same surface type sequence + same topological adjacency pattern. A 5mm through-hole and a 20mm through-hole both have CYLINDER faces with concave edges to PLANE caps — same pattern, different scale. The model is invariant to size (area normalized), orientation (no global frame in node features), and surrounding geometry (receptive field is 2-hop, not global).

**Q: Why face-level graph over coedge-level?**
Coedge-level (BRepNet) is more expressive but requires custom kernels and complex implementation. Face-level AAG captures ~80% of the expressiveness at ~20% of the implementation risk — the right tradeoff for 5-day scope. The main loss: coedge direction isn't captured, so 180° symmetric holes look identical (acceptable for recognition).

---

## Extensibility

**Q: What happens on an unseen feature type at inference?**
The encoder still runs — GINEConv produces per-face embeddings regardless of feature type. The embedding space proximity is still meaningful (similar topology → similar embeddings). Self-supervised pretraining (BRepMAE on ABC dataset) would improve this further by learning topology-aware representations before seeing any labels.

**Q: What breaks at the 10th feature type?**
Embedding space crowding (64-dim for 10 types), hard negative instability (geometrically similar types create very hard negatives that destabilize training), and class imbalance. Fixes: increase embedding dim to 128/256, curriculum learning for negatives, weighted sampling.

**Q: What breaks at the 50th feature type?**
Architecture still works. NMS becomes O(50N²) — fixable with early pruning. Rare types with <100 labeled examples become a data problem. Self-supervised pretraining on ABC dataset (1M unlabeled STEP files) reduces labeled examples needed per type by ~5-10×.

---

## Results

**Q: Your F1 is 0.829. SOTA is 99%+. Explain.**
SOTA (BRepGAT 99.1%, AAGNet 99.94%) solves face *classification* — assign a label to every face given the full model. Our task is *subgraph retrieval*: find all instances of a reference feature without knowing what feature types exist in the query. This is harder (metric learning problem, not classification). No apples-to-apples comparison exists in the literature.

**Q: Why does Phase 2 F1 drop from 0.829 to 0.810?**
Adding blind_hole introduces harder negatives into the contrastive batch — a through-hole and blind-hole have similar cylinder topology, so negative pairs are harder and can pull the through-hole cluster slightly. −2.3% is within expected noise for adding a geometrically similar class. The inference pipeline is unchanged.

**Q: Can you generate results on demand?**
`python scripts/generate_report.py` prints a consolidated table of all metrics.json files across all experiments in one command.

---

## Engineering

**Q: How is the LLM baseline scoped?**
Never serialize the full model — 2-hop neighborhoods only (~800 tokens per seed). For models with >50 seeds, chunk by seed batch. At context overflow: truncate to 2-hop neighborhood. Cost: ~$0.002/query (Claude Haiku) vs ~$0.00001/query (GNN on GPU). GNN is ~40,000× cheaper at production scale.

**Q: Why not seam merge if you know about the 180° cylinder problem?**
`merge_seam_faces()` is implemented in `src/parsing/graph_builder.py` but not yet applied in the H5 preprocessing pipeline (H5 files are pre-built). For STEP-path inference (`src/parsing/step_parser.py`), seam merge can be applied at parse time. Noted as V2 preprocessing improvement.
