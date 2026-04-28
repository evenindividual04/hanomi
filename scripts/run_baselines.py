"""Executes baseline models (LLM and Rule-based) against the test dataset.

Evaluates metric performance and formats the results out to the same
ResultsLogger format that `evaluate.py` uses, enabling direct comparison tables.

Usage:
  python scripts/run_baselines.py --h5_file data/test_MFCAD++.h5
"""

import argparse
import time
from collections import deque
from tqdm import tqdm
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.h5_dataset import MFCADPlusPlusDataset, LABEL_NAMES
from src.baselines.rule_based import run_rule_based
from src.baselines.llm_baseline import run_llm_baseline
from src.evaluation.metrics import instance_f1
from src.evaluation.results_logger import ResultsLogger, ModelResult


def _build_gt_instances(data, target_label: int) -> list:
    target = set((data.y == target_label).nonzero(as_tuple=True)[0].tolist())
    if not target:
        return []
    adj = {i: [] for i in range(data.num_nodes)}
    for c in range(data.edge_index.size(1)):
        adj[int(data.edge_index[0, c])].append(int(data.edge_index[1, c]))
    visited, instances = set(), []
    for s in target:
        if s in visited:
            continue
        comp, q = [], deque([s])
        visited.add(s)
        while q:
            n = q.popleft()
            comp.append(n)
            for nb in adj.get(n, []):
                if nb not in visited and nb in target:
                    visited.add(nb)
                    q.append(nb)
        instances.append({"face_ids": sorted(comp)})
    return instances

def evaluate_baseline(dataset, method_name, baseline_func, results_dir, feature_type, limit=None):
    logger = ResultsLogger(results_dir)
    print(f"\nEvaluating Baseline: {method_name}")

    label_id = MFCADPlusPlusDataset.label_names_for([feature_type])[0]
    count = 0

    for i, data in enumerate(tqdm(dataset, desc=method_name)):
        if limit and count >= limit:
            break

        gt_instances = _build_gt_instances(data, label_id)
        if not gt_instances:
            continue

        t0 = time.time()

        try:
            predicted_instances = baseline_func(data, feature_type)
        except Exception as e:
            print(f"Error in {method_name} for model {i}: {e}")
            predicted_instances = []

        elapsed_ms = (time.time() - t0) * 1000

        prec, rec, f1 = instance_f1(
            predicted=predicted_instances,
            ground_truth=gt_instances,
            iou_threshold=0.5
        )

        model_id_str = getattr(data, "model_id", [f"model_{i}"])[0]

        res = ModelResult(
            run_id="baselines",
            model_file=model_id_str,
            feature_type=feature_type,
            method=method_name,
            predicted_instances=predicted_instances,
            gt_instances=gt_instances,
            precision=prec,
            recall=rec,
            f1=f1,
            inference_ms=elapsed_ms
        )
        logger.log(res)
        count += 1

    print("\n--- Summary Table ---")
    logger.print_summary_table()
    logger.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", required=True, help="Path to test H5 file")
    parser.add_argument("--results_dir", default="results/baselines", help="Output directory")
    parser.add_argument("--limit", type=int, default=10, help="Max models to evaluate (LLM can be very slow)")
    parser.add_argument("--skip_llm", action="store_true", help="Skip the LLM evaluation to save API costs")
    parser.add_argument("--feature_type", default="through_hole", help="Feature type to evaluate")
    args = parser.parse_args()

    dataset = MFCADPlusPlusDataset(args.h5_file)
    print(f"Loaded {len(dataset)} graphs from test set.")
    
    evaluate_baseline(dataset, "rule_based", run_rule_based, args.results_dir, args.feature_type, limit=args.limit)

    if not args.skip_llm:
        print("\nWARNING: Calling external LLM API...")
        evaluate_baseline(dataset, "llm_baseline", run_llm_baseline, args.results_dir, args.feature_type, limit=args.limit)

if __name__ == "__main__":
    main()
