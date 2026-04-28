# LLM Baseline F1=0 Bug — Root Cause Analysis

**Symptom:** LLM baseline consistently reported F1=0.000 across all evaluation runs, regardless of model quality or sample size.

---

## The Root Bug: `round()` vs `int()` for Surface Type Decoding

### Background

MFCAD++ stores the surface type of each B-Rep face as a normalized float:

```
stored_value = surface_type / 11
```

Because the value is stored as **float32**, it cannot represent fractions exactly. The actual stored values are slightly below the true rational:

| True type | True value | float32 stored | `× 11` result |
|-----------|-----------|----------------|---------------|
| Plane (0) | 0/11 = 0.0 | 0.00000 | 0.00000 |
| Cylinder (1) | 1/11 ≈ 0.0909 | **0.09091** | **0.99999...** |
| Cone (2) | 2/11 ≈ 0.1818 | **0.18182** | **1.99999...** |

### The Wrong Decoder

`llm_baseline.py` decoded surface types using `round()`:

```python
def _decode_surface_type(val: float) -> int:
    return round(float(val) * 11)
```

Applied to the float32 stored values:

```python
round(0.09091 * 11) = round(0.99999) = 1   # "cylinder" — WRONG, this is plane
round(0.18182 * 11) = round(1.99999) = 2   # "cone"     — WRONG, this is cylinder
```

`round()` rounds **up** when the true result is just below an integer — the opposite of what the float32 truncation requires.

### The Correct Decoder

`rule_based.py` had already fixed this exact bug (noted in session history as *"round()→int() stype decode"*):

```python
def _get_surface_type(data, face_idx):
    return int(float(data.x[face_idx, 4].item()) * 11)
```

With `int()` (truncation):

```python
int(0.09091 * 11) = int(0.99999) = 0   # plane    ✓
int(0.18182 * 11) = int(1.99999) = 1   # cylinder ✓
```

`llm_baseline.py` was never updated with the same fix.

---

## How the Bug Cascaded into F1=0

The wrong decoder caused a chain of failures:

### Step 1 — Wrong seed selection

The LLM baseline seeds from cylindrical faces, filtering `surface_type == 1`:

```python
candidate_seeds = [
    i for i in range(data.num_nodes)
    if _decode_surface_type(data.x[i, 4].item()) == 1   # should match cylinders
]
```

With `round()`, `== 1` matched **plane faces** (stored as 0.09091 → round → 1), not cylinder faces (stored as 0.18182 → round → 2). Every seed was a flat planar face: stock material, pocket walls, steps.

**Verified:** for model 0 with GT through_hole faces at [7, 20], the "cylinder" seed list was [0, 1, 2, 3, 4] — all plane faces. Face 7 (the actual cylinder) was never in the candidate set.

### Step 2 — LLM sees wrong geometry

The LLM was sent subgraph JSON for planar faces and asked "is this a through_hole?". Planar faces don't look like holes, but a small 8B model (llama3.1-8b) has high confidence bias and said "yes" frequently anyway — producing many false positives on every model.

### Step 3 — Face IoU always zero

Even when the LLM said "yes" for a seed, the returned prediction was `face_ids = [seed]` — a plane face index. The GT instance contained the actual cylinder face index. These are different faces, so:

```
IoU({plane_face}, {cylinder_face}) = 0 < 0.5 threshold → counted as FP, not TP
```

No prediction ever matched any GT instance.

### Step 4 — F1=0

With TP=0, FP=many, FN=all GT instances:

```
Precision = 0 / (0 + FP) = 0
Recall    = 0 / (0 + FN) = 0
F1        = 0
```

---

## Other Bugs Found During Investigation

These were also fixed but are secondary to the root cause:

| Bug | Location | Effect |
|-----|----------|--------|
| Predicted `face_ids` was 2-hop subgraph (~25 faces) | `llm_baseline.py` | IoU(pred, GT) ≤ 0.13 even for correct seeds — all predictions false positives |
| Confidence threshold `> 0.5` too permissive | `llm_baseline.py` | 8B model over-predicted; raised to `> 0.7` |
| Cerebras/Mistral API keys not detected | `evaluate_baselines.py` | Script bailed out with "No API key found" even when `CEREBRAS_API_KEY` was set |

---

## Fix Applied

```python
# Before (wrong)
def _decode_surface_type(val: float) -> int:
    return round(float(val) * 11)

# After (correct — matches rule_based.py)
def _decode_surface_type(val: float) -> int:
    return int(float(val) * 11)
```

After this fix, for model 0:
- Cylinder seed list: `[7, 20, 26, 28]` — GT faces 7 and 20 are the **first two seeds**
- LLM is now shown actual cylindrical hole geometry
- Predictions are `[seed]` = the cylinder face itself
- IoU with GT = 1.0 when the LLM correctly identifies the hole
