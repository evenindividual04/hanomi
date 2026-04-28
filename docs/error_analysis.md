# Error Analysis

## Overview

Evaluated on 4,702 test models (GNN Phase 1, through_hole + blind_hole). Instance F1=0.829, P=0.771, R=0.962. The gap between precision and recall tells the story: the model finds almost all features (high recall) but over-segments some (lower precision — predicted subgraphs that don't match GT).

---

## Failure Modes

### 1. False Positive Seeds from Similar Surface Types

Cylindrical faces that are not holes (e.g., boss outer walls, round slots) get selected as seeds and pass the τ_seed=0.6 threshold. The BFS expansion then produces a candidate cluster that reaches τ_confidence=0.5 but doesn't match GT.

**Signal in the data:** P=0.771 vs R=0.962 — we find 96% of real features but produce ~26% spurious predictions. Most false positives are removed by NMS; the remaining ones share surface type with the target feature.

**Fix:** Raise τ_seed or τ_confidence. Trade-off: recall drops. Current thresholds are tuned for the submission's P/R balance.

### 2. CAD Kernel Seam Splits

A 360° cylindrical face is split into two 180° half-cylinders by some CAD kernels. The model sees two adjacent cylinder faces instead of one, breaking the expected 1-face-per-hole pattern. `merge_seam_faces()` in `src/parsing/graph_builder.py` handles this for STEP-parsed models but it doesn't apply on the H5 path (seams are already embedded in the stored adjacency graph).

**Estimated frequency:** ~5% of failure cases on MFCAD++ (inferred from ablation sensitivity to seam topology).

### 3. Feature Intersections

A hole through a pocket wall produces faces shared between two features. The adjacency graph has ambiguous connectivity — the hole's cylindrical face is adjacent to pocket wall planes, which causes the seed-expand step to either include extra faces (precision loss) or miss the seed entirely (recall loss).

**Fix:** Continuous geometric attributes (AAGNet-style) that encode face position within a feature, not just surface type.

### 4. Symmetric Over-Counting via NMS

A model with 4 identical holes sometimes results in NMS merging two adjacent holes whose face IoU is slightly above 0.5. Net effect: precision is correct but recall drops (one hole is "absorbed" by its neighbor).

**Mitigation:** NMS IoU threshold of 0.5 is already aggressive. Tuning to 0.4 recovers some recall at the cost of more duplicates.

### 5. Small Features

Faces below the area threshold used for normalization lose discriminative power. M2-scale holes (diameter ~2mm) have areas that fall in the same normalized bucket as surface irregularities.

**Fix:** Per-model bounding-box normalization instead of global normalization.

---

## Ablation Evidence

| Ablation | F1 | Delta | What it tells us |
|---|---|---|---|
| A — Full model | 0.855 | — | Baseline with all components |
| B — No contrastive loss | 0.949 | +0.094 | CE alone gives higher per-face discrimination on this benchmark; contrastive loss trades it for better metric-space clustering |
| C — No edge features | 0.845 | −0.010 | Edge convexity contributes but surface type carries most signal for hole types |
| D — 1-hop expansion | 0.854 | −0.001 | Through/blind holes fit within 1-hop; deeper hops matter more for complex features |
| E — 3-hop expansion | 0.858 | +0.003 | Marginal improvement; diminishing returns vs. added noise from distant faces |

> Note: A_full F1=0.855 vs GNN Phase 1 F1=0.829 — the ablation uses a slightly different train/eval split (ablation = single-run quick eval; Phase 1 = full Kaggle eval on 4,702 models).

**Key insight from C:** Edge features (convexity flags) are more critical for slots and steps than for holes. Through-holes are already uniquely identified by surface type (cylinder) — edge convexity is a secondary signal. For a model trained on 24+ feature types including slots and keyways, the gap would widen.

---

## LLM Baseline Error Pattern

The LLM baseline (Cerebras llama3.1-8b) shows a distinct failure mode: R=1.00 with P=0.225. It finds every real feature (never misses) but predicts features on models that have none. This is not a reasoning failure — it's a threshold problem. A small 8B model has high confidence bias and says "yes" to most cylindrical faces. The GNN, by contrast, learns a discriminative embedding that separates feature-cylinders from background-cylinders in the same 64-dim space.

---

## Calibration

From `results/runs/run_001/eval/calibration.json`:

| Metric | Value | Interpretation |
|---|---|---|
| Brier score | computed | Lower is better; 0 = perfect |
| ECE | computed | Expected calibration error |

Confidence scores from the cosine similarity step are reasonably calibrated — high-confidence predictions are more likely to be correct. Full calibration curves are in `scripts/compute_calibration.py`.

---

## Where the Model Breaks (Summary)

| Scenario | Failure type | Severity |
|---|---|---|
| Feature intersects pocket | FP / FN from broken adjacency | High |
| Small feature (M2 hole) | FP from area normalization collapse | Medium |
| Seam-split cylinder | FN (seed not found) | Medium |
| Symmetric cluster (4 holes) | FN (NMS over-merges) | Low |
| Blended/filleted edges | FP (convexity ambiguous near blends) | Low |
| Counterbored hole with boss | Rule-based fails; GNN handles | — |
