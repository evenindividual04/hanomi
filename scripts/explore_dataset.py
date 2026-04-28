"""Dataset exploration script.

Prints the internal structure of an MFCAD++ H5 file to validate
assumptions made in h5_dataset.py before running full training.

Usage:
    python scripts/explore_dataset.py \\
        --h5_file MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5 \\
        --n_batches 2
"""

import argparse
import numpy as np
import h5py
import sys
from pathlib import Path


def explore(h5_path: str, n_batches: int = 2) -> None:
    print(f"\n{'='*60}")
    print(f"H5 FILE: {h5_path}")
    print(f"{'='*60}")

    with h5py.File(h5_path, "r") as f:
        batch_keys = sorted(f.keys())
        print(f"\nTotal batches : {len(batch_keys)}")
        print(f"First keys    : {batch_keys[:5]}")

        for bi, bk in enumerate(batch_keys[:n_batches]):
            grp = f[bk]
            print(f"\n{'─'*50}")
            print(f"BATCH: {bk}")
            print(f"{'─'*50}")

            for dataset_key in sorted(grp.keys()):
                ds = grp[dataset_key]
                try:
                    arr = np.array(ds)
                    print(f"  {dataset_key:20s}  shape={arr.shape}  dtype={arr.dtype}")
                    if dataset_key == "CAD_model":
                        names = [n.decode() if isinstance(n, bytes) else str(n)
                                 for n in arr[:3]]
                        print(f"    sample names: {names}")
                    elif dataset_key == "idx":
                        print(f"    first 5 values : {arr.flatten()[:5].tolist()}")
                        print(f"    last  5 values : {arr.flatten()[-5:].tolist()}")
                    elif dataset_key == "V_1":
                        print(f"    min/max per col: {arr.min(axis=0).round(3)} / {arr.max(axis=0).round(3)}")
                    elif dataset_key == "labels":
                        unique, counts = np.unique(arr, return_counts=True)
                        print(f"    unique labels  : {dict(zip(unique.tolist(), counts.tolist()))}")
                except Exception as e:
                    print(f"  {dataset_key:20s}  (could not read: {e})")

            # Validate idx → model boundary assumption
            if "idx" in grp and "V_1" in grp and "CAD_model" in grp:
                idx = np.array(grp["idx"])
                V1 = np.array(grp["V_1"])
                n_models_by_name = len(np.array(grp["CAD_model"]))
                total_nodes = len(V1)

                print(f"\n  [VALIDATION]")
                print(f"  CAD_model count   : {n_models_by_name}")
                print(f"  idx shape         : {idx.shape}  dtype={idx.dtype}")
                print(f"  V_1 total nodes   : {total_nodes}")

                if idx.ndim == 2:
                    brep_starts = idx[:, 0].astype(int)
                    mesh_starts = idx[:, 1].astype(int)
                    print(f"  idx col0 (V1)     : first5={brep_starts[:5].tolist()} last5={brep_starts[-5:].tolist()}")
                    print(f"  idx col1 (V2)     : first5={mesh_starts[:5].tolist()}")

                    # Q-001: What's in V1[0 : brep_starts[0]]?
                    first_model_start = int(brep_starts[0])
                    if first_model_start > 0:
                        print(f"\n  [Q-001] V1[0:{first_model_start}] exists before first model — {first_model_start} orphan nodes")
                        print(f"  Orphan V1[0] features: {V1[0].tolist()}")
                    else:
                        print(f"\n  [Q-001] Model 0 starts at V1[0] — no orphan nodes ✅")

                    # Model sizes
                    boundaries = np.append(brep_starts, total_nodes)
                    sizes = np.diff(boundaries)
                    print(f"\n  Model sizes (first 5): {sizes[:5].tolist()}")
                    print(f"  Model sizes (last  5): {sizes[-5:].tolist()}")
                    print(f"  Min/max model size    : {sizes.min()} / {sizes.max()}")

                    # Q-003: Surface type encoding in V1 col 4
                    surf_col = V1[:, 4]
                    unique_surf = np.unique(surf_col).round(4)
                    print(f"\n  [Q-003] Unique surface_type values in V1 col4: {unique_surf.tolist()}")
                    # Check if they match k/11 pattern
                    fracs = np.round(unique_surf * 11).astype(int)
                    print(f"  → As k/11 numerators: {fracs.tolist()}")
                else:
                    idx_flat = idx.flatten().astype(int)
                    print(f"  idx (1D) range    : [{idx_flat.min()}, {idx_flat.max()}]")
                    boundaries = np.append(idx_flat, total_nodes)
                    sizes = np.diff(boundaries)
                    print(f"  Model sizes (first 5): {sizes[:5].tolist()}")


            # Spot-check edge matrices
            print(f"\n  [EDGE MATRIX CHECK]")
            for prefix in ["A_1", "E_1", "E_2", "E_3"]:
                idx_key = f"{prefix}_idx"
                if idx_key in grp:
                    eidx = np.array(grp[idx_key])
                    print(f"  {prefix}_idx shape: {eidx.shape}  (n_edges = {len(eidx)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore MFCAD++ H5 file structure")
    parser.add_argument("--h5_file", default="MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5")
    parser.add_argument("--n_batches", type=int, default=2,
                        help="Number of batch groups to inspect")
    args = parser.parse_args()

    if not Path(args.h5_file).exists():
        print(f"ERROR: file not found: {args.h5_file}", file=sys.stderr)
        sys.exit(1)

    explore(args.h5_file, args.n_batches)


if __name__ == "__main__":
    main()
