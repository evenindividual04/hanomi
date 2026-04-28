"""Benchmark inference speed and cost on different hardware."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import psutil
import torch
import yaml

from src.data.dataloader import make_dataloaders
from src.models.feature_recognizer import FeatureRecognizer


def benchmark_inference(
    config_path: str,
    checkpoint_path: str,
    n_samples: int = 100,
    h5_dir: str | None = None,
) -> Dict:
    """Benchmark inference speed on test set.

    Measures:
    - Time per model (ms)
    - Memory usage
    - Batch throughput

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        n_samples: Number of models to benchmark

    Returns:
        Dictionary with benchmark results
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = FeatureRecognizer(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # Get test data
    resolved_h5_dir = h5_dir or config['data']['h5_dir']
    loaders = make_dataloaders(
        h5_dir=resolved_h5_dir,
        feature_types=config['feature_types'],
        batch_size=1,
    )

    # Benchmark
    times = []
    memory_usage = []

    for i, data in enumerate(loaders['test']):
        if i >= n_samples:
            break

        data = data.to(device)

        # Measure memory before
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Measure inference time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000

        # Measure memory after
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        times.append(elapsed_ms)
        memory_usage.append(mem_after - mem_before)

    # Compute statistics
    times_array = __import__('numpy').array(times)
    memory_array = __import__('numpy').array(memory_usage)

    results = {
        "hardware": torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU",
        "n_samples": n_samples,
        "time_per_model_ms": {
            "mean": float(times_array.mean()),
            "std": float(times_array.std()),
            "min": float(times_array.min()),
            "max": float(times_array.max()),
            "median": float(__import__('numpy').median(times_array)),
        },
        "memory_per_model_mb": {
            "mean": float(memory_array.mean()),
            "std": float(memory_array.std()),
        },
        "throughput_models_per_sec": 1000.0 / float(times_array.mean()),
    }

    # Add GPU-seconds metric
    if device.type == 'cuda':
        # Approximate GPU-seconds per query
        gpu_seconds_per_query = float(times_array.mean()) / 1000.0
        results["gpu_seconds_per_query"] = gpu_seconds_per_query
    else:
        results["gpu_seconds_per_query"] = 0.0

    # Save results
    output_dir = Path(checkpoint_path).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_inference.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Inference Benchmark Results")
    print(f"{'='*60}")
    print(f"Hardware: {results['hardware']}")
    print(f"Time per model: {results['time_per_model_ms']['mean']:.2f} ± {results['time_per_model_ms']['std']:.2f} ms")
    print(f"Throughput: {results['throughput_models_per_sec']:.1f} models/sec")
    if device.type == 'cuda':
        print(f"GPU-seconds/query: {results['gpu_seconds_per_query']:.4f}")
    print(f"{'='*60}\n")

    return results


def estimate_daily_cost(
    queries_per_day: int,
    gpu_type: str = "T4",
    results: Dict = None,
) -> Dict:
    """Estimate daily cost for production inference.

    Args:
        queries_per_day: Number of queries per day
        gpu_type: Type of GPU (T4, g4dn.xlarge, etc.)
        results: Benchmark results (optional)

    Returns:
        Dictionary with cost estimates
    """
    # GPU hourly costs (approximate 2024 prices)
    gpu_costs = {
        "T4": 0.35,
        "g4dn.xlarge": 0.526,
        "local": 0.0,  # Local GPU, no direct cost
    }

    hourly_cost = gpu_costs.get(gpu_type, 0.35)

    if results:
        gpu_seconds_per_query = results.get("gpu_seconds_per_query", 0.015)
    else:
        gpu_seconds_per_query = 0.015  # Default estimate

    total_gpu_seconds = gpu_seconds_per_query * queries_per_day
    total_gpu_hours = total_gpu_seconds / 3600
    daily_cost = total_gpu_hours * hourly_cost

    cost_estimate = {
        "gpu_type": gpu_type,
        "queries_per_day": queries_per_day,
        "gpu_seconds_per_query": gpu_seconds_per_query,
        "total_gpu_seconds_per_day": total_gpu_seconds,
        "total_gpu_hours_per_day": total_gpu_hours,
        "hourly_gpu_cost": hourly_cost,
        "daily_cost": daily_cost,
    }

    return cost_estimate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--h5_dir', default=None, help='Optional override for H5 dataset directory')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of models to benchmark')
    parser.add_argument('--estimate_cost', type=int, default=None,
                       help='Estimate daily cost for N queries')
    args = parser.parse_args()

    # Run benchmark
    results = benchmark_inference(args.config, args.checkpoint, args.n_samples, args.h5_dir)

    # Estimate cost if requested
    if args.estimate_cost:
        cost = estimate_daily_cost(args.estimate_cost, results=results)

        output_dir = Path(args.checkpoint).parent.parent
        cost_path = output_dir / "cost_estimate.json"

        with open(cost_path, 'w') as f:
            json.dump(cost, f, indent=2)

        print(f"\nCost Estimate for {args.estimate_cost:,} queries/day:")
        print(f"  Daily cost: ${cost['daily_cost']:.2f}")
        print(f"  GPU-seconds: {cost['total_gpu_seconds_per_day']:.0f}")
        print(f"  Cost saved to {cost_path}")
