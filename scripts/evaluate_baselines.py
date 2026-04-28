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
    if api_key is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('GEMINI_API_KEY')

    if not api_key:
        print("Warning: No API key found, skipping LLM baseline")
        return {
            'method': 'llm',
            'error': 'No API key provided',
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }

    selected_model = llm_model.lower()
    if selected_model.startswith(("gemini-", "gemma-")):
        os.environ['GEMINI_API_KEY'] = api_key
    elif selected_model.startswith(("groq-", "llama-", "mixtral-", "deepseek-")):
        os.environ['GROQ_API_KEY'] = api_key
    elif selected_model.startswith("claude-"):
        os.environ['ANTHROPIC_API_KEY'] = api_key

    with open(config_path) as f:
        config = yaml.safe_load(f)

    resolved_h5_dir = h5_dir or config['data']['h5_dir']

    loaders = make_dataloaders(
        h5_dir=resolved_h5_dir,
        feature_types=config['feature_types'],
        batch_size=1,
    )
    target_label = MFCADPlusPlusDataset.label_names_for([config['feature_types'][0]])[0]

    results = []
    overflow_count = 0
    total_time = 0

    for i, data in enumerate(loaders['test']):
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
            print(f"Error calling LLM: {e}")
            model_results = []
            overflow_count += 1

        elapsed_ms = (time.time() - start) * 1000
        total_time += elapsed_ms

        model_id = getattr(data, 'model_id', 'unknown')
        if isinstance(model_id, list):
            model_id = model_id[0]
        if hasattr(model_id, 'item'):
            model_id = model_id.item()

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

    print(f"Running baseline evaluation: {args.methods}")

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
