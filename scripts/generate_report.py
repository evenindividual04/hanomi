"""Programmatic results report across all experiments.

Reads all results/**/metrics.json files and prints a single consolidated
comparison table: method × F1 / Precision / Recall / ms/model.

Usage:
  python scripts/generate_report.py
  python scripts/generate_report.py --results_dir results
"""

import argparse
import json
import sys
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_rows(metrics: dict, label: str) -> list[dict]:
    """Extract one row per method from a metrics dict."""
    rows = []
    per_method = metrics.get("per_method", {})
    if per_method:
        for method, m in per_method.items():
            rows.append({
                "experiment": label,
                "method": method,
                "f1": m.get("mean_f1", float("nan")),
                "precision": m.get("mean_precision", float("nan")),
                "recall": m.get("mean_recall", float("nan")),
                "ms": m.get("mean_ms", float("nan")),
                "n_models": m.get("n_models", 0),
            })
    else:
        rows.append({
            "experiment": label,
            "method": "unknown",
            "f1": metrics.get("mean_f1", float("nan")),
            "precision": metrics.get("mean_precision", float("nan")),
            "recall": metrics.get("mean_recall", float("nan")),
            "ms": metrics.get("mean_inference_ms", float("nan")),
            "n_models": metrics.get("n_models", 0),
        })
    return rows


def collect_results(results_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        rel = metrics_path.relative_to(results_dir)
        label = str(rel.parent).replace("/eval", "").replace("\\eval", "")
        try:
            metrics = _load_metrics(metrics_path)
            rows.extend(_extract_rows(metrics, label))
        except Exception as e:
            print(f"[WARN] Could not read {metrics_path}: {e}", file=sys.stderr)
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No metrics.json files found.")
        return

    header = f"{'Experiment':<35} {'Method':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'ms':>7} {'N':>6}"
    print(header)
    print("-" * len(header))

    for r in rows:
        f1  = f"{r['f1']:.4f}"  if r['f1'] == r['f1']  else "  n/a"
        pr  = f"{r['precision']:.4f}" if r['precision'] == r['precision'] else "  n/a"
        rec = f"{r['recall']:.4f}" if r['recall'] == r['recall'] else "  n/a"
        ms  = f"{r['ms']:.1f}" if r['ms'] == r['ms'] else "  n/a"
        print(f"{r['experiment']:<35} {r['method']:<22} {f1:>6} {pr:>6} {rec:>6} {ms:>7} {r['n_models']:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print consolidated results table")
    parser.add_argument("--results_dir", default="results", help="Root results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect_results(results_dir)
    print_table(rows)


if __name__ == "__main__":
    main()
