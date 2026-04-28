# Ablation Experiments

This directory contains configuration files for ablation experiments that test the contribution of different architectural choices.

## Ablation Matrix

| Ablation | Description | Key Change | Expected F1 | Notes |
|----------|-------------|------------|------------|-------|
| **A: Full** | Baseline model | All features enabled | **0.84** | Reference point |
| **B: No Contrastive** | Remove contrastive loss | `contrastive_weight: 0.0` | 0.76 (-0.08) | Tests metric learning contribution |
| **C: No Edge Features** | Remove edge attributes | Zero out edge_attr | 0.78 (-0.06) | Tests edge semantics importance |
| **D: 1-Hop** | Smaller receptive field | `k_hop: 1`, 2 layers | 0.71 (-0.13) | Tests receptive field size |
| **E: 3-Hop** | Larger receptive field | `k_hop: 3`, 4 layers | 0.82 (-0.02) | Tests over-expansion effect |

## Running Ablations

```bash
# Run each ablation
python scripts/train.py --config configs/ablations/A_full.yaml --output_dir results/ablations/A_full
python scripts/train.py --config configs/ablations/B_no_contrastive.yaml --output_dir results/ablations/B_no_contrastive
python scripts/train.py --config configs/ablations/C_no_edge_features.yaml --output_dir results/ablations/C_no_edge_features
python scripts/train.py --config configs/ablations/D_1hop.yaml --output_dir results/ablations/D_1hop
python scripts/train.py --config configs/ablations/E_3hop.yaml --output_dir results/ablations/E_3hop
```

## Multi-Seed Ablations (Recommended)

For robust results, run each ablation with multiple seeds:

```bash
python scripts/train.py --config configs/ablations/A_full.yaml --multi_seed --seeds 42 123 456 --output_dir results/ablations/A_full
python scripts/train.py --config configs/ablations/B_no_contrastive.yaml --multi_seed --seeds 42 123 456 --output_dir results/ablations/B_no_contrastive
# ... repeat for all ablations
```

## Implementation Notes

### Ablation C (No Edge Features)
The config uses `zero_edge_features: true` which requires modifying the dataloader to zero out edge attributes. This is a cleaner approach than changing the model architecture.

**Modification needed in `src/data/dataloader.py`:**
```python
if config.get('data', {}).get('zero_edge_features', False):
    for batch in loader:
        if hasattr(batch, 'edge_attr'):
            batch.edge_attr.zero_()
```

### Ablations D & E (k-hop)
The `k_hop` parameter affects inference expansion. The encoder receptive field is determined by `num_layers`:
- 1-hop: 2 layers (each provides 1-hop context)
- 2-hop (baseline): 3 layers
- 3-hop: 4 layers

## Analyzing Results

After running all ablations, compare the aggregated results:

```python
import json
import pandas as pd

results = {}
for ablation in ['A_full', 'B_no_contrastive', 'C_no_edge_features', 'D_1hop', 'E_3hop']:
    with open(f'results/ablations/{ablation}/aggregated_results.json') as f:
        results[ablation] = json.load(f)

df = pd.DataFrame({
    ablation: {
        'F1 Mean': r['test_f1']['mean'],
        'F1 Std': r['test_f1']['std'],
        'Val Loss': r['best_val_loss']['mean'],
    }
    for ablation, r in results.items()
}).T

print(df)
```

## Expected Insights

1. **Contrastive Loss (A vs B)**: Should show significant drop in F1, confirming that metric learning helps cluster same-type features.

2. **Edge Features (A vs C)**: Should show moderate drop, confirming that edge convexity and dihedral angles are critical for feature boundary detection.

3. **Receptive Field (A, D, E)**: Should show optimal performance at 2-hop, with 1-hop too small and 3-hop adding noise.

4. **Combined Effect**: If we ran ablations in combination (e.g., no contrastive + no edges), we'd expect even larger drops, showing these components work synergistically.

## Reporting

In the write-up, present:

1. **Ablation Table**: Show F1, Precision, Recall for each ablation
2. **Component Contribution**: Calculate ΔF1 for each change
3. **Discussion**: Explain why each component matters for CAD feature recognition
4. **Trade-offs**: Discuss complexity vs performance for different k-hop sizes
