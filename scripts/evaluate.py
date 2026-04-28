"""Evaluation script for testing the FeatureRecognizer.

Runs a test loop over the test_MFCAD++.h5 file and calculates:
1. Face-Level F1 (per class and macro) via sklearn classification_report
2. Instance-Level Precision, Recall, F1 (IoU >= 0.5) using real seed-and-expand inference

Usage:
  python scripts/evaluate.py --weights checkpoints/best.pt --h5_file test_MFCAD++.h5
"""

import argparse
import time
from collections import deque
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.models.feature_recognizer import FeatureRecognizer
from src.data.h5_dataset import MFCADPlusPlusDataset, LABEL_NAMES, LABEL_IDS
from src.evaluation.metrics import face_level_f1, instance_f1
from src.evaluation.results_logger import ResultsLogger, ModelResult
from src.inference.seed_expand import find_feature_instances


def _build_adj_dict_local(edge_index: torch.Tensor, n_nodes: int) -> dict:
    adj = {i: [] for i in range(n_nodes)}
    for col in range(edge_index.size(1)):
        s = int(edge_index[0, col])
        d = int(edge_index[1, col])
        adj[s].append(d)
    return adj


def _build_gt_instances(data, target_label: int) -> list:
    """Group contiguous faces with *target_label* into GT instance clusters.

    Finds connected components among faces whose label equals *target_label*
    using the graph's adjacency (edge_index).  Returns a list of dicts
    ``[{"face_ids": [...]}]`` compatible with :func:`instance_f1`.
    """
    label_mask = data.y == target_label
    target_faces = set(label_mask.nonzero(as_tuple=True)[0].tolist())

    if not target_faces:
        return []

    adj = _build_adj_dict_local(data.edge_index, data.num_nodes)

    visited = set()
    instances = []

    for start in target_faces:
        if start in visited:
            continue
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr in adj.get(node, []):
                if nbr not in visited and nbr in target_faces:
                    visited.add(nbr)
                    queue.append(nbr)
        instances.append({"face_ids": sorted(component)})

    return instances


def evaluate_segmentation(model: FeatureRecognizer, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Segmentation Eval"):
            batch = batch.to(device)
            _, seg_logits = model(batch)
            preds = seg_logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    from sklearn.metrics import classification_report
    print("\n--- Face-Level Segmentation Report ---")
    print(classification_report(all_labels, all_preds, zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", required=True, help="Path to test H5 file")
    parser.add_argument("--weights", required=True, help="Path to best model weights")
    parser.add_argument("--config", default="configs/counterbored_hole.yaml", help="Model config")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--results_dir", default="results/eval", help="Output dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = FeatureRecognizer(cfg).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    feature_types = cfg.get("feature_types", ["through_hole"])
    inf_cfg = cfg.get("inference", {})
    tau_seed          = inf_cfg.get("tau_seed", 0.6)
    tau_expand        = inf_cfg.get("tau_expand", 0.4)
    tau_confidence    = inf_cfg.get("tau_confidence", 0.5)
    nms_iou_threshold = inf_cfg.get("nms_iou_threshold", 0.5)

    feature_label_ids = MFCADPlusPlusDataset.label_names_for(feature_types)

    full_dataset = MFCADPlusPlusDataset(args.h5_file)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(full_dataset)} graphs for evaluation.")

    print("\n--- Face-Level Segmentation Report ---")
    evaluate_segmentation(model, full_loader, device)

    logger = ResultsLogger(args.results_dir)

    for ft_name, label_id in zip(feature_types, feature_label_ids):
        print(f"\n{'='*60}")
        print(f"Instance evaluation: {ft_name} (label {label_id})")
        print(f"{'='*60}")

        ft_dataset = MFCADPlusPlusDataset(args.h5_file, feature_labels=[label_id])
        if len(ft_dataset) == 0:
            print(f"No models contain label {label_id} ({ft_name}); skipping.")
            continue

        ft_loader = DataLoader(ft_dataset, batch_size=1, shuffle=False)

        ref_graph = None
        ref_mask = None
        for data in ft_loader:
            ref_graph = data
            ref_mask = (data.y == label_id)
            break

        if ref_mask.sum().item() == 0:
            print(f"Reference graph has no faces with label {label_id}; skipping.")
            continue

        print(f"Reference model: {getattr(ref_graph, 'model_id', ['unknown'])[0] if hasattr(ref_graph, 'model_id') and isinstance(ref_graph.model_id, list) else getattr(ref_graph, 'model_id', 'unknown')}")
        print(f"Query models to evaluate: {len(ft_dataset)}")

        for i, data in enumerate(tqdm(ft_loader, desc=f"SeedExpand [{ft_name}]")):
            model_name = getattr(data, "model_id", [f"model_{i}"])
            if isinstance(model_name, list):
                model_name = model_name[0]

            gt_instances = _build_gt_instances(data, label_id)
            if not gt_instances:
                continue

            t0 = time.time()
            pred_instances = find_feature_instances(
                model,
                ref_graph,
                ref_mask,
                data,
                tau_seed=tau_seed,
                tau_expand=tau_expand,
                tau_confidence=tau_confidence,
                nms_iou_threshold=nms_iou_threshold,
                device=device,
            )
            elapsed_ms = (time.time() - t0) * 1000

            prec, rec, f1 = instance_f1(pred_instances, gt_instances, iou_threshold=0.5)

            res = ModelResult(
                run_id="eval",
                model_file=model_name,
                feature_type=ft_name,
                method="gnn_seed_expand",
                predicted_instances=pred_instances,
                gt_instances=gt_instances,
                precision=prec,
                recall=rec,
                f1=f1,
                inference_ms=elapsed_ms,
            )
            logger.log(res)

        all_pred_faces = []
        all_gt_faces = []
        with torch.no_grad():
            for data in ft_loader:
                data = data.to(device)
                mask = (data.y == label_id)
                all_gt_faces.extend(mask.cpu().nonzero(as_tuple=True)[0].tolist())
                _, seg_logits = model(data)
                pred_mask = (seg_logits.argmax(dim=-1) == label_id)
                all_pred_faces.extend(pred_mask.cpu().nonzero(as_tuple=True)[0].tolist())

        if all_gt_faces:
            _, _, face_f1_val = face_level_f1(all_pred_faces, all_gt_faces)
            print(f"Face-level F1 for {ft_name}: {face_f1_val:.4f}")

    print("\n--- Instance-Level Summary ---")
    logger.print_summary_table()
    agg = logger.save()
    if agg:
        print(f"\nAggregate: F1={agg['mean_f1']:.4f}  P={agg['mean_precision']:.4f}  R={agg['mean_recall']:.4f}  "
              f"over {agg['n_models']} models  ({agg.get('mean_inference_ms', 0):.1f} ms avg)")


if __name__ == "__main__":
    main()
