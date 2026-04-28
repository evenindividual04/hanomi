"""Programmatic results logging.

Saves per-model results as JSON (full detail) and CSV (summary table),
then computes aggregate metrics in metrics.json.
"""

from __future__ import annotations

import csv
import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelResult:
    run_id:               str
    model_file:           str
    feature_type:         str
    method:               str        # "gnn" | "rule_based" | "llm"
    predicted_instances:  List[Dict]
    gt_instances:         List[Dict]
    precision:            float
    recall:               float
    f1:                   float
    inference_ms:         float
    timestamp:            str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    notes:                str = ""


class ResultsLogger:
    """Log, aggregate, and persist per-model evaluation results.

    Usage::

        logger = ResultsLogger("results/runs/run_001")
        logger.log(ModelResult(...))
        agg = logger.save()
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ModelResult] = []
        self.calibration_results: List[Dict] = []
        self.per_model_stats: List[Dict] = []

    def log(self, result: ModelResult) -> None:
        self.results.append(result)

    def log_instance(
        self,
        confidence: float,
        is_true_positive: bool,
    ) -> None:
        """Log a single instance for calibration tracking.

        Args:
            confidence: Predicted confidence score
            is_true_positive: Whether the instance is a true positive
        """
        self.calibration_results.append({
            "confidence": confidence,
            "is_true_positive": is_true_positive,
        })

    def log_model_results(
        self,
        model_id: str,
        predicted_instances: List[Dict],
        gt_instances: List[Dict],
    ) -> None:
        """Log statistics for a single model.

        Args:
            model_id: Model identifier
            predicted_instances: List of predicted instances
            gt_instances: List of ground truth instances
        """
        n_predicted = len(predicted_instances)
        n_ground_truth = len(gt_instances)

        # Compute per-model metrics
        precision, recall, f1 = instance_f1(predicted_instances, gt_instances)

        self.per_model_stats.append({
            "model_id": model_id,
            "n_predicted": n_predicted,
            "n_ground_truth": n_ground_truth,
            "n_overpredict": max(0, n_predicted - n_ground_truth),
            "n_underpredict": max(0, n_ground_truth - n_predicted),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    def save(self) -> Dict[str, Any]:
        """Write JSON, CSV, and aggregate metrics. Returns aggregate dict."""
        if not self.results:
            return {}

        # Full detail JSON
        json_path = self.run_dir / "per_model_results.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Summary CSV
        csv_path = self.run_dir / "per_model_results.csv"
        fieldnames = list(asdict(self.results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                row = asdict(r)
                # Flatten list fields to JSON strings for CSV
                row["predicted_instances"] = json.dumps(row["predicted_instances"])
                row["gt_instances"]        = json.dumps(row["gt_instances"])
                writer.writerow(row)

        # Aggregate metrics
        agg = self._aggregate()

        # Add calibration metrics if available
        if self.calibration_results:
            from .metrics import compute_calibration_metrics
            calibration = compute_calibration_metrics(self.calibration_results)
            agg["calibration"] = calibration

        # Add per-model statistics
        if self.per_model_stats:
            import pandas as pd
            df_per_model = pd.DataFrame(self.per_model_stats)
            csv_path = self.run_dir / "per_model_stats.csv"
            df_per_model.to_csv(csv_path, index=False)

            # Compute aggregate statistics
            agg["avg_overpredict"] = float(df_per_model["n_overpredict"].mean())
            agg["avg_underpredict"] = float(df_per_model["n_underpredict"].mean())
            agg["n_models_with_overpredict"] = int((df_per_model["n_overpredict"] > 0).sum())
            agg["n_models_with_underpredict"] = int((df_per_model["n_underpredict"] > 0).sum())
            agg["per_model_f1_mean"] = float(df_per_model["f1"].mean())
            agg["per_model_f1_std"] = float(df_per_model["f1"].std())

        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(agg, f, indent=2)

        print(f"Results saved to {self.run_dir}")
        return agg

    def _aggregate(self) -> Dict[str, Any]:
        f1s   = [r.f1   for r in self.results]
        precs = [r.precision for r in self.results]
        recs  = [r.recall    for r in self.results]
        mss   = [r.inference_ms for r in self.results]

        # Per-method breakdown
        methods = sorted({r.method for r in self.results})
        per_method = {}
        for m in methods:
            rs = [r for r in self.results if r.method == m]
            per_method[m] = {
                "mean_f1":       float(np.mean([r.f1 for r in rs])),
                "mean_precision":float(np.mean([r.precision for r in rs])),
                "mean_recall":   float(np.mean([r.recall for r in rs])),
                "mean_ms":       float(np.mean([r.inference_ms for r in rs])),
                "n_models":      len(rs),
            }

        return {
            "mean_f1":        float(np.mean(f1s)),
            "mean_precision": float(np.mean(precs)),
            "mean_recall":    float(np.mean(recs)),
            "mean_inference_ms": float(np.mean(mss)),
            "n_models":       len(self.results),
            "per_method":     per_method,
        }

    def print_summary_table(self) -> None:
        """Print a human-readable comparison table to stdout."""
        header = f"{'model':<30} {'method':<12} {'F1':>6} {'P':>6} {'R':>6} {'ms':>8}"
        print("\n" + header)
        print("─" * len(header))
        for r in sorted(self.results, key=lambda x: (x.method, -x.f1)):
            print(
                f"{r.model_file:<30} {r.method:<12} "
                f"{r.f1:6.3f} {r.precision:6.3f} {r.recall:6.3f} "
                f"{r.inference_ms:8.1f}"
            )
