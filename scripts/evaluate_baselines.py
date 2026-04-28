"""Evaluate rule-based and LLM baselines on test set.

This script provides a fair comparison between our GNN approach and baselines.
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.baselines.rule_based import run_rule_based
from src.baselines.llm_baseline import run_llm_baseline
from src.data.dataloader import make_dataloaders
from src.data.h5_dataset import MFCADPlusPlusDataset
from src.evaluation.metrics import instance_f1


def _build_gt_instances(data, target_label: int) -> List[Dict]:
    """Build connected-component GT instances for a target label."""
    target = set((data.y == target_label).nonzero(as_tuple=True)[0].tolist())
    if not target:
        return []

    adj = {i: [] for i in range(data.num_nodes)}
    for c in range(data.edge_index.size(1)):
        src = int(data.edge_index[0, c])
        dst = int(data.edge_index[1, c])
        adj[src].append(dst)

    visited = set()
    instances = []
    for start in target:
        if start in visited:
            continue
        comp = []
        q = deque([start])
        visited.add(start)
        while q:
            node = q.popleft()
            comp.append(node)
            for nb in adj.get(node, []):
                if nb not in visited and nb in target:
                    visited.add(nb)
                    q.append(nb)
        instances.append({"face_ids": sorted(comp)})

    return instances


def evaluate_rule_based(config_path: str, output_dir: str) -> Dict:
    """Run rule-based baseline on test set.

    Args:
        config_path: Path to config file
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loaders = make_dataloaders(
        h5_dir=config['data']['h5_dir'],
        feature_types=config['feature_types'],
        batch_size=1,  # Process one at a time for baselines
    )
    target_label = MFCADPlusPlusDataset.label_names_for([config['feature_types'][0]])[0]

    results = []
    total_time = 0

    for data in loaders['test']:
        # Run rule-based detection
        import time
        start = time.time()
        instances = run_rule_based(data, config['feature_types'][0])
        elapsed_ms = (time.time() - start) * 1000
        total_time += elapsed_ms

        model_id = getattr(data, 'model_id', 'unknown')
        if isinstance(model_id, list):
            model_id = model_id[0]
        if hasattr(model_id, 'item'):
            model_id = model_id.item()

        results.append({
            'model_id': model_id,
            'predicted': instances,
            'ground_truth': _build_gt_instances(data, target_label),
            'inference_ms': elapsed_ms,
        })

    # Compute metrics
    all_preds = [inst for r in results for inst in r['predicted']]
    all_gts = [inst for r in results for inst in r['ground_truth']]

    if all_gts:
        precision, recall, f1 = instance_f1(all_preds, all_gts)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    metrics = {
        'method': 'rule_based',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_models': len(results),
        'avg_inference_ms': total_time / max(1, len(results)),
        'num_predicted': len(all_preds),
        'num_ground_truth': len(all_gts),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/rule_based_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRule-based baseline results:")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Avg inference: {metrics['avg_inference_ms']:.2f} ms/model")

    return metrics


def evaluate_llm_baseline(
    config_path: str,
    output_dir: str,
    api_key: str = None,
    sample_size: int = 50,
    h5_dir: str = None,
    llm_model: str = "gemini-2.0-flash",
    max_seeds: int = 5,
) -> Dict:
    """Run LLM baseline on test set (subset due to cost).

    Args:
        config_path: Path to config file
        output_dir: Directory to save results
        api_key: API key for LLM service
        sample_size: Number of models to evaluate (limited due to cost)

    Returns:
        Dictionary with metrics
    """
    _KEY_ENV_MAP = {
        ("gemini-", "gemma-"): "GEMINI_API_KEY",
        ("groq-", "llama-", "mixtral-", "deepseek-"): "GROQ_API_KEY",
        ("claude-",): "ANTHROPIC_API_KEY",
        ("glm-",): "ZHIPUAI_API_KEY",
        ("cerebras:",): "CEREBRAS_API_KEY",
        ("mistral:",): "MISTRAL_API_KEY",
    }

    def _detect_key_env(model: str) -> str:
        for prefixes, env_var in _KEY_ENV_MAP.items():
            if model.startswith(prefixes):
                return env_var
        return ""

    if api_key is None:
        env_var = _detect_key_env(llm_model)
        api_key = os.environ.get(env_var) if env_var else None
        if api_key is None:
            # Fall back to any available key
            for env_var in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY",
                            "GROQ_API_KEY", "ANTHROPIC_API_KEY", "ZHIPUAI_API_KEY"):
                api_key = os.environ.get(env_var)
                if api_key:
                    break

    if not api_key and not llm_model.startswith(("ollama:", "hf:")):
        print("Warning: No API key found, skipping LLM baseline")
        return {
            'method': 'llm',
            'error': 'No API key provided',
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }

    env_var = _detect_key_env(llm_model)
    if env_var and api_key:
        os.environ[env_var] = api_key

    with open(config_path) as f:
        config = yaml.safe_load(f)

    resolved_h5_dir = h5_dir or config['data']['h5_dir']

    from pathlib import Path as _Path
    h5_path = next(
        p for p in [
            _Path(resolved_h5_dir) / "test_MFCAD++.h5",
            _Path(resolved_h5_dir) / "test_MFCAD.h5",
        ] if p.exists()
    )
    print(f"Opening {h5_path.name}...", flush=True)
    dataset = MFCADPlusPlusDataset(h5_path)
    target_label = MFCADPlusPlusDataset.label_names_for([config['feature_types'][0]])[0]
    print(f"Dataset ready — {len(dataset)} models. Running on first {sample_size}.", flush=True)

    results = []
    overflow_count = 0
    total_time = 0

    print(f"\nLLM baseline — model: {llm_model} | models: {sample_size} | max_seeds: {max_seeds}")
    print("-" * 60)

    for i, data in enumerate(dataset):
        if i >= sample_size:
            break

        # Run LLM baseline on a bounded number of seeds per model.
        import time
        start = time.time()
        try:
            model_results = run_llm_baseline(
                data,
                config['feature_types'][0],
                max_seeds=max_seeds,
                llm_model=llm_model,
            )
        except Exception as e:
            print(f"  [{i+1}/{sample_size}] ERROR: {e}")
            model_results = []
            overflow_count += 1

        elapsed_ms = (time.time() - start) * 1000
        total_time += elapsed_ms

        model_id = getattr(data, 'model_id', 'unknown')
        if isinstance(model_id, list):
            model_id = model_id[0]
        if hasattr(model_id, 'item'):
            model_id = model_id.item()

        n_pred = len(model_results)
        avg_so_far = total_time / (i + 1)
        eta_s = avg_so_far * (sample_size - i - 1) / 1000
        print(f"  [{i+1:>3}/{sample_size}] model={model_id} | "
              f"found={n_pred} instances | {elapsed_ms:.0f}ms | ETA {eta_s:.0f}s",
              flush=True)

        if (i + 1) % 10 == 0:
            done = i + 1
            running_results = results + [{
                'predicted': model_results,
                'ground_truth': _build_gt_instances(data, target_label),
            }]
            all_p = [x for r in running_results for x in r['predicted']]
            all_g = [x for r in running_results for x in r['ground_truth']]
            if all_g:
                from src.evaluation.metrics import instance_f1
                p, r, f = instance_f1(all_p, all_g)
                print(f"\n  --- Milestone {done}/{sample_size}: F1={f:.3f} P={p:.3f} R={r:.3f} ---\n",
                      flush=True)

        results.append({
            'model_id': model_id,
            'predicted': model_results,
            'ground_truth': _build_gt_instances(data, target_label),
            'inference_ms': elapsed_ms,
        })

    # Compute metrics
    all_preds = [inst for r in results for inst in r['predicted']]
    all_gts = [inst for r in results for inst in r['ground_truth']]

    if all_gts:
        precision, recall, f1 = instance_f1(all_preds, all_gts)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    overflow_rate = overflow_count / max(1, len(results))

    metrics = {
        'method': 'llm',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_models': len(results),
        'avg_inference_ms': total_time / max(1, len(results)),
        'num_predicted': len(all_preds),
        'num_ground_truth': len(all_gts),
        'overflow_rate': overflow_rate,
        'overflow_count': overflow_count,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/llm_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nLLM baseline results (sampled {len(results)} models):")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Avg inference: {metrics['avg_inference_ms']:.2f} ms/model")
    print(f"  Overflow rate: {overflow_rate:.2%}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    parser.add_argument('--methods', nargs='+', default=['rule_based'],
                       help='Baseline methods to evaluate')
    parser.add_argument('--llm_api_key', default=None, help='API key for LLM service')
    parser.add_argument('--llm_sample_size', type=int, default=50,
                       help='Number of models to evaluate with LLM')
    parser.add_argument('--h5_dir', default=None, help='Optional override for H5 dataset directory')
    parser.add_argument('--llm_model', default=None, help='LLM model name to use (e.g. gemini-2.0-flash, llama-3.3-70b-versatile, claude-3-5-sonnet-latest)')
    parser.add_argument('--llm_max_seeds', type=int, default=5, help='Max seeds per model for LLM baseline')
    args = parser.parse_args()

    print(f"Running baseline evaluation: {args.methods}", flush=True)

    all_metrics = {}

    if args.h5_dir:
        with open(args.config) as f:
            cfg_override = yaml.safe_load(f)
        cfg_override['data']['h5_dir'] = args.h5_dir
        tmp_cfg = Path(args.output_dir) / '_tmp_baseline_config.yaml'
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(tmp_cfg, 'w') as f:
            yaml.safe_dump(cfg_override, f)
        config_path = str(tmp_cfg)
    else:
        config_path = args.config

    if 'rule_based' in args.methods:
        metrics = evaluate_rule_based(config_path, args.output_dir)
        all_metrics['rule_based'] = metrics

    if 'llm' in args.methods:
        metrics = evaluate_llm_baseline(
            config_path,
            args.output_dir,
            args.llm_api_key,
            args.llm_sample_size,
            args.h5_dir,
            args.llm_model or "gemini-2.0-flash",
            args.llm_max_seeds,
        )
        all_metrics['llm'] = metrics

    # Save combined results
    with open(f"{args.output_dir}/baseline_comparison.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nBaseline comparison saved to {args.output_dir}/baseline_comparison.json")
