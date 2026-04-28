"""Inference Demo Script

Takes a trained FeatureRecognizer and runs the seed-and-expand
subgraph extraction algorithm over a given graph.

Usage:
  python scripts/inference_demo.py --weights checkpoints/best.pt
"""

import argparse
import sys
import torch
from torch_geometric.data import Data
from pathlib import Path
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from src.models.feature_recognizer import FeatureRecognizer
from src.data.h5_dataset import MFCADPlusPlusDataset
from src.inference.seed_expand import find_feature_instances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--config", type=str, default="configs/counterbored_hole.yaml")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to validation/test H5")
    parser.add_argument("--query_idx", type=int, default=1, help="Index of query model in H5")
    parser.add_argument("--ref_idx", type=int, default=0, help="Index of reference model in H5")
    parser.add_argument("--feature_class", type=int, default=2, help="Label index of feature to expand")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = FeatureRecognizer(cfg).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    print(f"Loading {args.h5_file}...")
    dataset = MFCADPlusPlusDataset(args.h5_file)
    if args.ref_idx >= len(dataset) or args.query_idx >= len(dataset):
        print("Error: query_idx or ref_idx out of bounds.")
        sys.exit(1)

    ref_graph = dataset[args.ref_idx]
    query_graph = dataset[args.query_idx]

    # Create boolean mask for the target feature class in the reference graph
    # If the reference graph doesn't contain the feature class, the code will exit.
    ref_mask = (ref_graph.y == args.feature_class)
    if not ref_mask.any():
        print(f"Error: Reference model {getattr(ref_graph, 'model_id', args.ref_idx)} "
              f"contains NO faces of class {args.feature_class}.")
        sys.exit(1)

    print(f"Reference: {getattr(ref_graph, 'model_id', args.ref_idx)} (Faces: {ref_graph.num_nodes})")
    print(f"Query:     {getattr(query_graph, 'model_id', args.query_idx)} (Faces: {query_graph.num_nodes})")
    print("Initiating seed-and-expand extraction...")

    inf_cfg = cfg.get("inference", {})
    instances = find_feature_instances(
        model=model,
        reference_graph=ref_graph,
        reference_mask=ref_mask,
        query_graph=query_graph,
        tau_seed=inf_cfg.get("tau_seed", 0.6),
        tau_expand=inf_cfg.get("tau_expand", 0.4),
        tau_confidence=inf_cfg.get("tau_confidence", 0.5),
        nms_iou_threshold=inf_cfg.get("nms_iou_threshold", 0.5),
        device=device,
    )

    print(f"\nExtracted {len(instances)} clusters:")
    print(json.dumps(instances, indent=2))

if __name__ == "__main__":
    main()
