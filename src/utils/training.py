"""Training utilities including early stopping."""

import logging
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation loss improved.
        min_delta (float): Minimum change to qualify as an improvement.
        restore_best_weights (bool): Whether to restore model weights from the epoch with best value.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save state from
        """
        score = -val_loss  # Higher is better (we're tracking negative loss)

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model when validation loss decreases.

        Args:
            model: Model to save state from
        """
        if self.restore_best_weights:
            self.best_model_state = model.state_dict().copy()

    def restore_best_model(self, model: nn.Module) -> None:
        """Restore model weights from best epoch.

        Args:
            model: Model to restore state to
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logging.info(f'Restored best model with loss {self.best_loss:.4f}')
