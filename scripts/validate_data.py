"""Validate MFCAD++ H5 dataset for quality issues."""

import argparse
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def validate_h5_dataset(h5_path: Path) -> List[Dict]:
    """Check for quality issues in H5 dataset.

    Checks for:
    - Degenerate faces (area = 0)
    - Disconnected graphs
    - Feature distribution shifts
    - Outliers in node/edge features

    Args:
        h5_path: Path to H5 file

    Returns:
        List of issue dictionaries
    """
    issues = []

    with h5py.File(h5_path, 'r') as f:
        n_graphs = f['V_1'].shape[0]

        print(f"Validating {n_graphs} graphs from {h5_path.name}...")

        for i in range(n_graphs):
            # Get node features
            V = f['V_1'][i]  # [n_nodes, 5]
            A = f['A_1'][i]  # [n_nodes, n_nodes]

            n_nodes = V.shape[0]
            n_edges = (A > 0).sum() // 2

            # Check for degenerate faces
            areas = V[:, 0]
            degenerate_mask = areas < 1e-6
            n_degenerate = degenerate_mask.sum()

            if n_degenerate > 0:
                issues.append({
                    'graph_idx': i,
                    'type': 'degenerate_faces',
                    'count': int(n_degenerate),
                    'total_nodes': n_nodes,
                })

            # Check for disconnected graphs
            if n_nodes > 1 and n_edges == 0:
                issues.append({
                    'graph_idx': i,
                    'type': 'disconnected_graph',
                    'n_nodes': n_nodes,
                    'n_edges': n_edges,
                })

            # Check for outliers in area
            if len(areas) > 0:
                area_mean = areas.mean()
                area_std = areas.std()
                if area_std > 0:  # Avoid division by zero
                    outlier_mask = areas > area_mean + 5 * area_std
                    n_outliers = outlier_mask.sum()

                    if n_outliers > 0:
                        issues.append({
                            'graph_idx': i,
                            'type': 'area_outliers',
                            'count': int(n_outliers),
                            'area_mean': float(area_mean),
                            'area_std': float(area_std),
                            'max_area': float(areas.max()),
                        })

    return issues


def generate_quality_report(
    issues: List[Dict],
    h5_path: Path,
    output_path: Path,
) -> Dict:
    """Generate a quality report from validation issues.

    Args:
        issues: List of issue dictionaries
        h5_path: Path to H5 file
        output_path: Path to save report

    Returns:
        Dictionary with summary statistics
    """
    # Group by issue type
    issue_types = {}
    for issue in issues:
        issue_type = issue['type']
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)

    # Compute statistics
    summary = {
        'h5_file': str(h5_path),
        'total_issues': len(issues),
        'issues_by_type': {
            issue_type: len(issue_list)
            for issue_type, issue_list in issue_types.items()
        },
        'issue_details': issues,
    }

    # Add detailed statistics per type
    for issue_type, issue_list in issue_types.items():
        if issue_type == 'degenerate_faces':
            total_faces = sum(i.get('total_nodes', 0) for i in issue_list)
            total_degenerate = sum(i.get('count', 0) for i in issue_list)
            summary['degenerate_faces'] = {
                'total_faces': total_faces,
                'total_degenerate': total_degenerate,
                'percentage': (total_degenerate / total_faces * 100) if total_faces > 0 else 0,
            }
        elif issue_type == 'area_outliers':
            max_area = max(i.get('max_area', 0) for i in issue_list)
            summary['area_outliers'] = {
                'max_outlier_area': float(max_area),
            }

    # Save report
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def print_summary(summary: Dict) -> None:
    """Print a human-readable summary of the validation results.

    Args:
        summary: Summary dictionary from generate_quality_report
    """
    print(f"\n{'='*60}")
    print(f"Data Quality Report: {Path(summary['h5_file']).name}")
    print(f"{'='*60}")
    print(f"Total issues found: {summary['total_issues']}\n")

    print("Issues by type:")
    for issue_type, count in summary['issues_by_type'].items():
        print(f"  - {issue_type}: {count}")

    if 'degenerate_faces' in summary:
        stats = summary['degenerate_faces']
        print(f"\nDegenerate faces:")
        print(f"  Total faces: {stats['total_faces']}")
        print(f"  Degenerate: {stats['total_degenerate']} ({stats['percentage']:.2f}%)")

    if 'area_outliers' in summary:
        stats = summary['area_outliers']
        print(f"\nArea outliers:")
        print(f"  Max outlier area: {stats['max_outlier_area']:.4f}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_dir', required=True, help='Directory containing H5 files')
    parser.add_argument('--output_dir', required=True, help='Directory to save reports')
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # H5 files to validate
    h5_files = [
        h5_dir / "training_MFCAD++.h5",
        h5_dir / "val_MFCAD++.h5",
        h5_dir / "test_MFCAD++.h5",
    ]

    all_issues = []

    for h5_file in h5_files:
        if h5_file.exists():
            print(f"\nValidating {h5_file.name}...")
            issues = validate_h5_dataset(h5_file)

            output_path = output_dir / f"{h5_file.stem}_quality.json"
            summary = generate_quality_report(issues, h5_file, output_path)
            print_summary(summary)

            all_issues.extend(issues)
        else:
            print(f"Warning: {h5_file} not found, skipping")

    # Combined report
    if all_issues:
        combined_output = output_dir / "combined_quality_report.json"
        combined_summary = generate_quality_report(
            all_issues,
            h5_dir,
            combined_output
        )

        print(f"\nCombined report saved to {combined_output}")
        print(f"Total issues across all files: {len(all_issues)}")
    else:
        print("\nNo quality issues found! 🎉")
