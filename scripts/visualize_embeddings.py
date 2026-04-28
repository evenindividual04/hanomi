"""t-SNE visualization of subgraph embeddings.

Loads a trained checkpoint, runs GNN inference on test models,
aggregates to subgraph-level embeddings, and plots t-SNE.

Phase 1 (through_hole only) → one tight cluster vs background.
Phase 2 (through_hole + blind_hole) → two clusters vs background.

Usage:
  python scripts/visualize_embeddings.py \
      --checkpoint results/runs/run_001/checkpoints/best.pt \
      --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
      --output_dir results/figures \
      --n_models 500 \
      --label phase1

  python scripts/visualize_embeddings.py \
      --checkpoint results/runs/run_002/checkpoints/best.pt \
      --h5_dir /Users/anmolsen/Developer/MFCAD++_dataset/hierarchical_graphs \
      --output_dir results/figures \
      --n_models 500 \
      --label phase2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FEATURE_LABEL_MAP = {
    1: "through_hole",
    12: "blind_hole",
    24: "stock",
}

COLORS = {
    "through_hole": "#2196F3",
    "blind_hole": "#4CAF50",
    "stock": "#BDBDBD",
    "other": "#FF9800",
}


def collect_embeddings(model, dataset, n_models: int, device: torch.device):
    """Run GNN on dataset models, return (embeddings, labels) lists."""
    model.eval()
    embeddings = []
    labels = []

    seen = 0
    for data in tqdm(dataset, desc="Embedding models", total=min(n_models, len(dataset))):
        if seen >= n_models:
            break
        data = data.to(device)

        with torch.no_grad():
            face_emb, _ = model(data)

        # Group faces by their label, compute mean embedding per group
        face_labels = data.y.cpu().numpy() if hasattr(data, "y") else None
        if face_labels is None:
            seen += 1
            continue

        for label_id in np.unique(face_labels):
            mask = face_labels == label_id
            group_emb = F.normalize(face_emb[mask].mean(0), dim=0)
            embeddings.append(group_emb.cpu().numpy())

            label_name = FEATURE_LABEL_MAP.get(int(label_id), "other")
            labels.append(label_name)

        seen += 1

    return np.array(embeddings), labels


def plot_tsne(embeddings: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    label_set = sorted(set(labels))

    for lbl in label_set:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        color = COLORS.get(lbl, "#9C27B0")
        alpha = 0.4 if lbl in ("stock", "other") else 0.8
        size = 15 if lbl in ("stock", "other") else 25
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=lbl, alpha=alpha, s=size, linewidths=0)

    ax.set_title(title, fontsize=13)
    ax.legend(markerscale=1.5, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5_dir", required=True)
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--n_models", type=int, default=500)
    parser.add_argument("--label", default="phase1",
                        help="Output filename prefix (e.g. phase1, phase2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    # Search for config.yaml relative to checkpoint, then fall back to base config
    candidate_paths = [
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent / "config.yaml",
        Path("configs/counterbored_hole.yaml"),
        Path("configs/base.yaml"),
    ]
    config_path = next((p for p in candidate_paths if p.exists()), None)
    if config_path is None:
        print("No config.yaml found — using default encoder config", file=sys.stderr)
        config = {"encoder": {"node_in_dim": 8, "edge_in_dim": 3, "hidden_dim": 128, "out_dim": 64, "num_layers": 3, "dropout": 0.1},
                  "seg_head": {"in_dim": 64, "num_classes": 25},
                  "pooling": {"dim": 64, "mode": "attention"}}
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    from src.models.feature_recognizer import FeatureRecognizer
    from src.data.h5_dataset import MFCADPlusPlusDataset

    model = FeatureRecognizer(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    h5_dir = Path(args.h5_dir)
    h5_candidates = list(h5_dir.glob("test_*.h5"))
    if not h5_candidates:
        print(f"No test H5 file found in {h5_dir}", file=sys.stderr)
        sys.exit(1)
    h5_file = h5_candidates[0]
    print(f"Using: {h5_file}")

    dataset = MFCADPlusPlusDataset(h5_file)

    embeddings, labels = collect_embeddings(model, dataset, args.n_models, device)

    if len(embeddings) == 0:
        print("No embeddings collected — check dataset and model compatibility.")
        sys.exit(1)

    label_counts = {l: labels.count(l) for l in set(labels)}
    print(f"Label distribution: {label_counts}")

    output_path = Path(args.output_dir) / f"tsne_{args.label}.png"
    phase_num = args.label.replace("phase", "Phase ")
    title = f"Subgraph Embeddings — {phase_num} ({len(embeddings)} groups, {args.n_models} models)"
    plot_tsne(embeddings, labels, title, output_path)


if __name__ == "__main__":
    main()
