# Kaggle Goal Coverage Checklist

This checklist cross-verifies heavy-compute goals from project plans/docs against current Kaggle notebook execution paths.

## Reference Sources
- [README.md](../README.md)
- [PROGRESS.md](../PROGRESS.md)
- [IMPLEMENTATION_COMPLETE.md](../IMPLEMENTATION_COMPLETE.md)
- External plan used by the team: /Users/anmolsen/.local/share/kilo/plans/1776958545753-cosmic-engine.md

## Coverage Matrix

| Goal | Source of Goal | Expected Heavy Compute | Current Kaggle Execution Path | Coverage |
|---|---|---|---|---|
| Phase 1 core model training | README + plan + progress | 50-epoch training on baseline config | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 3 -> [scripts/train.py](../scripts/train.py) with [configs/counterbored_hole.yaml](../configs/counterbored_hole.yaml) | Yes |
| Phase 1 evaluation (instance + face metrics) | README + plan + progress | Evaluation on test H5 with metrics artifacts | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 4 -> [scripts/evaluate.py](../scripts/evaluate.py) | Yes |
| Phase 2 extensibility fine-tuning | README + plan + progress | Fine-tune on extensibility config | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 5 -> [scripts/train.py](../scripts/train.py) with [configs/extensibility_v2.yaml](../configs/extensibility_v2.yaml) | Yes |
| Phase 2 evaluation | README + plan + progress | Evaluate fine-tuned checkpoint | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 6 -> [scripts/evaluate.py](../scripts/evaluate.py) | Yes |
| Ablation matrix A/B/C/D/E | plan + progress + [configs/ablations/README.md](../configs/ablations/README.md) | Multiple expensive training runs for ablations | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 8 -> [scripts/train.py](../scripts/train.py) on [configs/ablations](../configs/ablations) | Yes |
| Rule-based baseline | plan + progress + implementation complete | Baseline detection comparison | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 9 -> [scripts/run_baselines.py](../scripts/run_baselines.py) | Yes |
| LLM baseline | plan + progress + implementation complete | Optional LLM baseline with sampled workload | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 9 -> [scripts/evaluate_baselines.py](../scripts/evaluate_baselines.py) with h5 override (if API key available) | Yes (key-gated) |
| Inference benchmarking + cost | plan + progress + implementation complete | Time/model and GPU-seconds style estimates | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 10 -> [scripts/benchmark_inference.py](../scripts/benchmark_inference.py) with notebook fallback benchmark | Yes |
| Data quality validation | plan + progress + implementation complete | Validate H5 splits and generate reports | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 11 -> [scripts/validate_data.py](../scripts/validate_data.py) | Yes |
| Result artifact packaging | plan + implementation complete | Single downloadable tarball for post-Kaggle docs | [notebooks/kaggle_training.ipynb](../notebooks/kaggle_training.ipynb) Step 12 packaging cell | Yes |

## Artifact Expectations by Step

| Step | Expected Artifact(s) |
|---|---|
| Step 3 | results/runs/run_001/checkpoints/best.pt |
| Step 4 | results/eval/metrics.json, results/eval/per_model_results.json |
| Step 5 | results/runs/run_002_extensibility/checkpoints/best.pt |
| Step 6 | results/eval_phase2/metrics.json |
| Step 8 | results/ablations/* |
| Step 9 | results/baselines/*, results/baselines_eval/* |
| Step 10 | benchmark_inference.json, cost_estimate.json (under run output) |
| Step 11 | results/data_quality/*.json |
| Step 12 | results/packages/hanomi_results.tar.gz |

## Practical Notes
- LLM baseline requires at least one API key in environment (ANTHROPIC_API_KEY or GEMINI_API_KEY).
- Benchmark step now has both a script path and an in-notebook fallback timing path using explicit state dict loading.
- Parsing-side features in [src/parsing](../src/parsing) are not part of the H5-based Kaggle heavy-compute pipeline.
