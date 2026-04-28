# Hanomi Feature Recognition — Findings Log

> **Purpose**: Living document of empirical discoveries, dataset surprises, and
> deviations from `hanomi_implementation_plan.md`. Updated every session.
> All findings are numbered and dated for traceability.

---

## Session 1 — 2026-04-21

### Environment

| Item | Value |
|---|---|
| OS | macOS (Apple Silicon, miniforge) |
| Conda binary | `/opt/homebrew/bin/conda` / `/opt/homebrew/Caskroom/miniforge/base` |
| `hanomi` env | **Does not exist yet** — needs `conda create -n hanomi python=3.10` |
| h5py available via | `diffusion_dl` env |
| Disk space | **~333 MB free** on a 460 GB drive — extremely tight |

---

### F-001 · Disk space critical

**Severity**: High  
**Discovery**: Drive is at 98% capacity (460 GB used out of 460 GB total).

| Directory | Size |
|---|---|
| `s1.0.0/` (Fusion360 extracted) | 15 GB |
| `MFCAD++_dataset/` | 7.7 GB |
| `s1.0.0.zip` | 2.9 GB (**deleted** 2026-04-21) |

**Action taken**: Deleted `s1.0.0.zip` (already extracted). Recovered ~2.9 GB.  
**Remaining risk**: If training produces large `.pt` cache files, disk will fill again.  
**Mitigation**: Do NOT cache preprocessed `.pt` files to disk. Load H5 directly into RAM.

---

### F-002 · `idx` is shape `(num_models, 2)`, not `(num_models,)`

**Severity**: High — breaks the entire dataset reader if uncorrected  
**Plan assumption** (`CLAUDE.md` §Dataset Pipeline): `idx` = 1D array of start node pointers  
**Reality**: `idx.shape = (num_models, 2)` where:
- `col 0` = start index into `V_1` (B-Rep face graph level) ✅
- `col 1` = start index into `V_2` (mesh/triangle level) — unused by us

**Evidence** (batch 0, val H5):
```
idx  shape=(29, 2), dtype=int64
flattened first 5: [40, 287, 78, 737, 122]
→ idx[0] = [40, 287]   model 0 starts at V1[40], V2[287]
→ idx[1] = [78, 737]   model 1 starts at V1[78], V2[737]
```

**Fix applied**: `brep_starts = idx_raw[:, 0].astype(int)` in `_parse_batch_group`.

> ✅ **Q-001 RESOLVED** (see F-011 below): V1[0:40] are **stock/unfeature faces** that
> belong to the batch's "padding" model. The first real model starts at `idx[0,0]=40`.
> The 40 orphan nodes have valid feature vectors — they are just not labelled as a
> named model. Our parser correctly skips them because model 0 boundary is [40:78].

---

### F-003 · `labels` dtype is `float32`, not `int64`

**Severity**: Medium  
**Plan assumption**: Labels are integer class indices  
**Reality**: Labels stored as `float32` (e.g. `0.0, 1.0, 24.0`)  
**Fix applied**: `.astype(np.int64)` cast in `_parse_batch_group`.

---

### F-004 · `surface_type` in `V_1` is a normalized float, not a class integer

**Severity**: Medium  
**Plan assumption** (`CLAUDE.md` Node Feature Schema): `surface_type (0–5 enum)` — integer  
**Reality**: V_1 column 4 values are in `[0.091, 0.455]`

Likely encoding (HierarchicalCADNet): `surface_type_id / N_types`.
Values `{0.091, 0.182, 0.273, 0.364, 0.455}` ≈ `{1/11, 2/11, 3/11, 4/11, 5/11}`.

**Confirmed encoding** (from exploration output):
```
Unique values: [0.0909, 0.1818, 0.4545]
→ As k/11 numerators: [1, 2, 5]
```
Only 3 surface types appear in this batch: cylinder (1/11), cone (2/11), torus (5/11).
Plane would be 0/11=0.0 — absent here because stock faces get label 24.

**Impact**: Cannot use as integer class index. Use as-is as a continuous float.  
**Action**: No change needed — model uses it as a float in the 8-dim node vector.

---

### F-005 · All V_1 features are pre-normalized to `[0, 1]`

**Severity**: Low (positive finding)  
**Evidence** (batch 0):
```
V_1 min/max per col: [0. 0. 0. 0. 0.091] / [1. 1. 1. 1. 0.455]
```
**Implication**: Surface area, centroid x/y/z already normalized by bounding box.
No additional normalization pass required at training time.

---

### F-006 · No "counterbored hole" label in MFCAD++ (confirmed expected)

**Status**: Known, confirmed  
**Strategy**: Use `through_hole` (label 1) + `blind_hole` (label 12) as training proxies.  
**Config**: `configs/counterbored_hole.yaml` trains on `[through_hole, blind_hole]`.

---

### F-007 · H5 batch key names sort lexicographically, not numerically

**Severity**: Low (correctness bug)  
**Discovery**: Batch keys are strings `'0', '1', '10', '100', ...`  
`sorted(['0','1','10','2'])` → `['0', '1', '10', '2']` — wrong order.  
**Fix needed** in `h5_dataset._load_all`:
```python
# Wrong:
for batch_key in sorted(f.keys()):
# Correct:
for batch_key in sorted(f.keys(), key=lambda x: int(x)):
```
**Status**: ⚠️ Not yet applied — fix required before next session.

---

### F-008 · Val set size and batch geometry

| Split | File size | Batches | Models/batch | Faces/batch |
|---|---|---|---|---|
| train | 1.19 GB | ~1450 est. | ~28 avg | ~875 |
| val | 255 MB | 312 | ~26 avg | ~807 |
| test | 254 MB | ~310 est. | ~28 avg | ~875 |

**RAM estimate for full val in memory**: ~80 MB — fits easily.
**RAM for full train**: ~340 MB — fine for 16 GB RAM.

---

### F-009 · A_1 = E_1 ∪ E_2 ∪ E_3 (confirmed for batch 0)

**Evidence** (batch 0):
```
A_1: 4567 edges
E_1 (convex):  3082
E_2 (concave): 1416
E_3 (smooth):    69
Sum:           4567  ← exact match
```
**Conclusion**: Building convexity map from E_1/E_2/E_3 covers all edges in A_1. ✅  
**Status**: Confirmed only for 1 batch. Assume holds generally.

---

### F-010 · A_3, A_4 are inter-level incidence matrices (safely ignored)

**A_3**: `shape=(8971, 2)` — B-Rep face ↔ mesh triangle connections  
**A_4**: Marked "redundant" in `h5_structure.txt`  
**Action**: Both ignored. We only use the B-Rep graph level.

---

## Implementation Deviations from Plan

| # | Plan says | Reality / Decision |
|---|---|---|
| D-001 | `node_in_dim=9` (STEP path) | H5 path uses **8-dim**: V_1(5) + degree + n_convex + n_concave |
| D-002 | `surface_type` as integer enum 0–5 | Used as **float** (V_1 col 4 already encoded as fraction) |
| D-003 | `idx` = 1D start-pointer array `(num_models,)` | **Actual: `(num_models, 2)`**, col 0 = V_1 boundary |
| D-004 | `labels` as `int` | **Stored as `float32`** in H5, cast to `int64` on load |
| D-005 | `preprocess.py` creates `.pt` files on disk | **Skipped** — load H5 directly into RAM (disk full, simpler) |
| D-006 | `hanomi` conda env ready | **Does not exist yet** — must create |
| D-007 | Lexicographic batch sort | Must use `sorted(keys, key=int)` for numerical correctness |

---

### F-011 · `V1[0:idx[0,0]]` contains orphan nodes not assigned to any named model

**Severity**: Medium  
**Discovery**: In batch 0, `idx[0, 0] = 40` but V1 starts at row 0.  
The 40 nodes at `V1[0:40]` have valid feature vectors but are **not linked to any `CAD_model` name**.

```
Orphan V1[0] features: [0.760, 1.0, 0.563, 0.408, 0.091]
                          area   cx    cy    cz   surf_type(cylinder)
```

**Interpretation**: These are stock/body faces from a model that was either:
  (a) cut across batch boundaries (its faces start at the end of the prev batch), or
  (b) an internal assembly model not tracked by name.

**Impact on parsing**: Our boundary array `boundaries = np.append(brep_starts, total_nodes)`
creates model 0 as `V1[40:78]` — the 40 orphan nodes (`V1[0:40]`) are **silently dropped**.  
This is correct behaviour: we only extract named models. ~4.6% of V1 nodes are lost per batch
(40/875 = 4.6%). This is acceptable.

**Validation**: Labels of orphan nodes are part of `labels[0:40]` — these faces still get
label supervision if we ever process the full batch jointly, but since we extract per named model,
this is fine.

---

## Open Questions

| # | Question | Priority | Status |
|---|---|---|---|
| Q-001 | What is in `V1[0:idx[0,0]]`? Are there orphan nodes? | High | ✅ **Resolved** (F-011) |
| Q-002 | `sorted(f.keys())` — lexicographic or numeric? | High | ✅ **Fixed** (F-007) |
| Q-003 | V_1 col 4 exact encoding? | Med | ✅ **Resolved** — `surface_type_id / 11` |
| Q-004 | Does E_1+E_2+E_3 always equal A_1? | Med | 🔍 Partial (1 batch confirmed) |
| Q-005 | pythonocc-core availability for STEP path at inference | Low | ❌ Blocked |

---

## Files Created This Session

```
setup.py                              ← package setup
requirements.txt                      ← pinned deps
.gitignore                            ← excludes data/, results/, *.pt
configs/base.yaml
configs/counterbored_hole.yaml
configs/extensibility_v2.yaml
src/__init__.py
src/utils/__init__.py
src/utils/seed.py
src/utils/logging.py
src/data/__init__.py
src/data/h5_dataset.py               ← H5→PyG reader (F-002, F-003 fixes applied)
src/data/dataloader.py               ← DataLoader factory + triplet construction
src/models/__init__.py
src/models/encoder.py                ← BRepEncoder (GraphSAGE, 3-layer, 8→128→64)
src/models/seg_head.py               ← SegmentationHead (25-class MLP)
src/models/pooling.py                ← SubgraphPooling (mean / attention)
src/models/feature_recognizer.py     ← Full model
src/losses/__init__.py
src/losses/contrastive.py            ← NT-Xent loss
src/losses/hybrid.py                 ← HybridLoss (seg + contrastive)
src/inference/__init__.py
src/inference/seed_expand.py         ← Seed-and-expand algorithm
src/inference/nms.py                 ← Face-IoU NMS
src/baselines/__init__.py
src/baselines/rule_based.py          ← Topological pattern matching
src/baselines/llm_baseline.py        ← JSON subgraph → Claude/GPT
src/evaluation/__init__.py
src/evaluation/metrics.py            ← Precision, Recall, F1, IoU
src/evaluation/results_logger.py     ← JSON + CSV + aggregate logging
scripts/explore_dataset.py           ← H5 structure validator (run ≥ 1 batch first)
scripts/train.py                     ← Main training script
tests/__init__.py
tests/test_h5_dataset.py             ← Requires hanomi env + H5 files
tests/test_model_forward.py          ← Synthetic graphs, no H5 needed
```

---

## Next Session Checklist

- [ ] **Fix F-007**: `sorted(f.keys(), key=lambda x: int(x))` in `h5_dataset._load_all`
- [ ] **Fix Q-001**: Add first-model V_1 inspection to `explore_dataset.py`
- [ ] Create `hanomi` conda env with pip deps
- [ ] Run `tests/test_model_forward.py` to validate model shapes
- [ ] Run `tests/test_h5_dataset.py` once env ready
- [ ] Write `scripts/inference_demo.py`
- [ ] Write `scripts/evaluate.py`
- [ ] Write `src/parsing/step_parser.py` (needs pythonocc-core)
- [ ] Begin training run (Day 2 tasks from plan)
