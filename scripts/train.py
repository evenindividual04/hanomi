"""Main training script.

Usage:
    python scripts/train.py --config configs/counterbored_hole.yaml --seed 42

Supports:
  - WandB experiment tracking (--wandb_project)
  - Fine-tuning from checkpoint (--checkpoint)
  - Cosine LR with warmup
  - Hard negative mining after epoch 5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Make hanomi-repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.training import EarlyStopping
from src.data.h5_dataset import MFCADPlusPlusDataset
from src.data.dataloader import make_dataloaders, build_triplets
from src.models.feature_recognizer import FeatureRecognizer
from src.losses.hybrid import HybridLoss

log = get_logger("train")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Hanomi feature recognizer")
    p.add_argument("--config",     required=True, help="Path to YAML config")
    p.add_argument("--checkpoint", default=None,  help="Resume/fine-tune from .pt checkpoint")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--seeds",      type=int, nargs='+', default=[42, 123, 456],
                       help="Multiple seeds for reproducibility")
    p.add_argument("--multi_seed", action="store_true",
                       help="Run with multiple seeds and aggregate results")
    p.add_argument("--output_dir", default=None,
                       help="Output directory. Auto-increments run_NNN if omitted.")
    p.add_argument("--wandb_project", default=None, help="WandB project name (disabled if None)")
    p.add_argument("--epochs",     type=int, default=None, help="Override config epochs")
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--h5_dir",     type=str, default=None, help="Override H5 data directory")
    p.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def next_run_dir(base: str = "results/runs") -> str:
    """Return the next non-existent results/runs/run_NNN path."""
    base_path = Path(base)
    n = 1
    while True:
        candidate = base_path / f"run_{n:03d}"
        if not candidate.exists():
            return str(candidate)
        n += 1


def load_config(path: str) -> dict:
    """Load YAML config and resolve nested structure."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    warmup_steps = cfg.get("warmup_epochs", 5) * steps_per_epoch
    total_steps  = cfg.get("epochs", 50) * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, target_label_ids, label_names) -> dict:
    """Run validation pass, return metrics dict."""
    model.eval()
    total_loss = 0.0
    correct = wrong = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        face_emb, seg_logits = model(batch)
        loss, _ = loss_fn(seg_logits, batch.y)
        total_loss += loss.item()

        pred = seg_logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        wrong   += (pred != batch.y).sum().item()
        n_batches += 1

    total_faces = correct + wrong
    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_acc":  correct / max(1, total_faces),
    }


def compute_f1_per_class(model, loader, device, num_classes=25):
    """Compute per-class F1 on the given loader."""
    model.eval()
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, logits = model(batch)
            pred = logits.argmax(dim=1).cpu()
            gt   = batch.y.cpu()
            for c in range(num_classes):
                tp[c] += ((pred == c) & (gt == c)).sum().float()
                fp[c] += ((pred == c) & (gt != c)).sum().float()
                fn[c] += ((pred != c) & (gt == c)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return f1   # [num_classes]


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    set_seed(args.seed)
    cfg = load_config(args.config)

    # Command-line overrides
    if args.epochs:     cfg["epochs"]     = args.epochs
    if args.lr:         cfg["lr"]         = args.lr
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.h5_dir:     cfg["data"]["h5_dir"] = args.h5_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Output dirs ───────────────────────────────────────────────────────
    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── Data ──────────────────────────────────────────────────────────────
    feature_types = cfg["feature_types"]
    label_ids = MFCADPlusPlusDataset.label_names_for(feature_types)
    log.info("Feature types: %s → label IDs: %s", feature_types, label_ids)

    loaders = make_dataloaders(
        h5_dir=cfg["data"]["h5_dir"],
        feature_types=feature_types,
        batch_size=cfg["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=args.seed,
    )
    log.info(
        "Dataset sizes — train: %d  val: %d  test: %d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = FeatureRecognizer(cfg).to(device)
    log.info(
        "Model params: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        log.info("Loaded checkpoint from %s", args.checkpoint)

    # ── Loss + optimiser ──────────────────────────────────────────────────
    loss_cfg = cfg.get("loss", {})
    loss_fn  = HybridLoss(**loss_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = get_scheduler(optimizer, cfg, len(loaders["train"]))

    # ── WandB ─────────────────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=cfg, name=args.output_dir)
        except ImportError:
            log.warning("wandb not installed — running without tracking")
            use_wandb = False

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs = cfg["epochs"]
    grad_clip = cfg.get("grad_clip", 1.0)

    # Early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=cfg.get("early_stopping", {}).get("patience", 10),
            min_delta=cfg.get("early_stopping", {}).get("min_delta", 0.001),
            restore_best_weights=cfg.get("early_stopping", {}).get("restore_best_weights", True),
        )
        log.info("Early stopping enabled: patience=%d, min_delta=%.4f",
                 early_stopping.patience, early_stopping.min_delta)

    for epoch in range(1, epochs + 1):
        model.train()
        hard_neg = epoch > 5
        epoch_log: dict = {"epoch": epoch}
        running: dict = {"total": 0.0, "seg": 0.0, "contrastive": 0.0}
        t0 = time.time()

        for step, batch in enumerate(loaders["train"]):
            batch = batch.to(device)
            optimizer.zero_grad()

            face_emb, seg_logits = model(batch)

            # Build triplets for contrastive loss
            triplets = build_triplets(
                face_emb.detach(),   # detach for mining; gradients via full forward
                batch.y,
                label_ids,
                batch.ptr,
                n_triplets=8,
                hard_negatives=hard_neg,
            )
            if triplets is not None:
                # Re-compute subgraph embeddings with gradients
                a_emb = _pool_triplet_embs(model, batch, batch.y, label_ids, batch.ptr, "anchor")
                p_emb = _pool_triplet_embs(model, batch, batch.y, label_ids, batch.ptr, "positive")
                n_emb = _pool_triplet_embs(model, batch, batch.y, label_ids, batch.ptr, "negative")
            else:
                a_emb = p_emb = n_emb = None

            loss, log_dict = loss_fn(seg_logits, batch.y, a_emb, p_emb, n_emb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            for k, v in log_dict.items():
                running[k] = running.get(k, 0.0) + v

        # ── Epoch summary ─────────────────────────────────────────────────
        n = len(loaders["train"])
        for k in running:
            epoch_log[f"train_{k}"] = running[k] / n

        val_metrics = evaluate(model, loaders["val"], loss_fn, device, label_ids, feature_types)
        epoch_log.update(val_metrics)
        epoch_log["lr"] = scheduler.get_last_lr()[0]
        epoch_log["elapsed"] = time.time() - t0

        log.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | lr=%.2e | %.1fs",
            epoch, epochs,
            epoch_log["train_total"],
            epoch_log["val_loss"],
            epoch_log["val_acc"],
            epoch_log["lr"],
            epoch_log["elapsed"],
        )

        if use_wandb:
            import wandb
            wandb.log(epoch_log)

        # ── Early stopping check ──────────────────────────────────────────
        if args.early_stopping:
            early_stopping(val_metrics["val_loss"], model)
            if early_stopping.early_stop:
                log.info(f"Early stopping triggered at epoch {epoch}")
                if early_stopping.restore_best_weights:
                    early_stopping.restore_best_model(model)
                break

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt_data = {
            "epoch":  epoch,
            "model":  model.state_dict(),
            "optim":  optimizer.state_dict(),
            "config": cfg,
            "val_loss": epoch_log["val_loss"],
        }
        torch.save(ckpt_data, ckpt_dir / "last.pt")
        if epoch_log["val_loss"] < best_val_loss:
            best_val_loss = epoch_log["val_loss"]
            torch.save(ckpt_data, ckpt_dir / "best.pt")
            log.info("  ✓ Saved best checkpoint (val_loss=%.4f)", best_val_loss)

    # ── Final F1 on test set ──────────────────────────────────────────────
    log.info("Computing F1 on test set …")
    f1 = compute_f1_per_class(model, loaders["test"], device)
    for lid in label_ids:
        from src.data.h5_dataset import LABEL_NAMES
        log.info("  F1[%s] = %.4f", LABEL_NAMES.get(lid, str(lid)), f1[lid].item())

    if use_wandb:
        import wandb
        wandb.finish()

    log.info("Done. Best checkpoint: %s", ckpt_dir / "best.pt")


# ──────────────────────────────────────────────────────────────────────────────
# Triplet pooling helper (with gradients)
# ──────────────────────────────────────────────────────────────────────────────

def _pool_triplet_embs(model, batch, labels, target_labels, ptr, role: str):
    """Pool per-graph target-face embeddings for contrastive loss."""
    import torch.nn.functional as F
    from src.data.h5_dataset import LABEL_NAMES

    target_set = set(target_labels)
    face_emb, _ = model(batch)  # will share computation graph
    pooled = []

    graphs = [g for g in range(len(ptr) - 1)]
    selected = graphs if role in ("anchor", "negative") else graphs[1:] + graphs[:1]

    for g in selected:
        start = int(ptr[g])
        end   = int(ptr[g + 1])
        if start >= end:
            continue
        g_emb = face_emb[start:end]
        g_lbl = labels[start:end]

        if role == "negative":
            mask = torch.tensor([l.item() not in target_set for l in g_lbl], dtype=torch.bool)
        else:
            mask = torch.tensor([l.item() in target_set for l in g_lbl], dtype=torch.bool)

        if mask.sum() > 0:
            pooled.append(g_emb[mask].mean(0))
        else:
            pooled.append(g_emb.mean(0))

    if not pooled:
        return None
    return torch.stack(pooled[:8])   # max 8 triplets


def aggregate_multi_seed_results(all_results: list, output_dir: Path) -> dict:
    """Aggregate results across multiple random seeds.

    Args:
        all_results: List of result dicts from each seed run
        output_dir: Directory to save aggregated results

    Returns:
        Dictionary with aggregated statistics
    """
    import numpy as np

    # Extract metrics
    test_f1s = [r["test_f1"] for r in all_results if "test_f1" in r]
    val_losses = [r["best_val_loss"] for r in all_results if "best_val_loss" in r]
    epochs_run = [r.get("epochs_run", r.get("epochs", 50)) for r in all_results]

    aggregation = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "test_f1": {
            "mean": float(np.mean(test_f1s)) if test_f1s else None,
            "std": float(np.std(test_f1s)) if test_f1s else None,
            "min": float(np.min(test_f1s)) if test_f1s else None,
            "max": float(np.max(test_f1s)) if test_f1s else None,
            "values": test_f1s,
        },
        "best_val_loss": {
            "mean": float(np.mean(val_losses)) if val_losses else None,
            "std": float(np.std(val_losses)) if val_losses else None,
            "values": val_losses,
        },
        "epochs_run": {
            "mean": float(np.mean(epochs_run)),
            "std": float(np.std(epochs_run)),
        },
        "individual_results": all_results,
    }

    # Save aggregation
    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregation, f, indent=2)

    # Print summary
    log.info("\n" + "="*60)
    log.info("Multi-Seed Aggregated Results")
    log.info("="*60)
    log.info(f"Seeds: {aggregation['seeds']}")
    if test_f1s:
        log.info(f"Test F1: {aggregation['test_f1']['mean']:.4f} ± {aggregation['test_f1']['std']:.4f}")
        log.info(f"  (min: {aggregation['test_f1']['min']:.4f}, max: {aggregation['test_f1']['max']:.4f})")
    if val_losses:
        log.info(f"Best val loss: {aggregation['best_val_loss']['mean']:.4f} ± {aggregation['best_val_loss']['std']:.4f}")
    log.info(f"Epochs run: {aggregation['epochs_run']['mean']:.1f} ± {aggregation['epochs_run']['std']:.1f}")
    log.info(f"Aggregation saved to {output_dir / 'aggregated_results.json'}")
    log.info("="*60 + "\n")

    return aggregation


def train_with_seed(args: argparse.Namespace, seed: int) -> dict:
    """Train with a specific seed and return results.

    Args:
        args: Command-line arguments
        seed: Random seed to use

    Returns:
        Dictionary with training results
    """
    # Set seed-specific output directory
    if args.multi_seed:
        base_out_dir = Path(args.output_dir)
        seed_out_dir = base_out_dir / f"seed_{seed}"
        args.output_dir = str(seed_out_dir)

    # Update seed in args
    args.seed = seed

    # Run training
    train(args)

    # Load results
    results = {
        "seed": seed,
        "output_dir": args.output_dir,
    }

    # Try to load metrics
    try:
        import json
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        if (ckpt_dir / "best.pt").exists():
            ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu")
            results["best_val_loss"] = ckpt.get("val_loss", float('inf'))
            results["epochs_run"] = ckpt.get("epoch", 0)
    except Exception as e:
        log.warning(f"Could not load checkpoint results for seed {seed}: {e}")

    # Compute test F1 if possible
    try:
        f1_path = Path(args.output_dir) / "test_f1.json"
        if f1_path.exists():
            with open(f1_path) as f:
                f1_data = json.load(f)
                results["test_f1"] = f1_data.get("mean_f1", 0.0)
    except Exception:
        pass

    return results


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = next_run_dir()
    log.info("Output dir: %s", args.output_dir)

    if args.multi_seed:
        # Multi-seed run
        all_results = []
        base_out_dir = Path(args.output_dir)
        base_out_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Running multi-seed experiment with seeds: {args.seeds}")

        for seed in args.seeds:
            log.info(f"\n{'='*60}")
            log.info(f"Training with seed {seed}")
            log.info(f"{'='*60}\n")

            try:
                result = train_with_seed(args, seed)
                all_results.append(result)
            except Exception as e:
                log.error(f"Seed {seed} failed: {e}")
                continue

        # Aggregate results
        if all_results:
            aggregate_multi_seed_results(all_results, base_out_dir)
        else:
            log.error("No successful seed runs!")
    else:
        # Single seed run
        train(args)
