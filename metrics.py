"""Defines tracker for metrics."""
import numpy as np


def _tensor_to_list(t):
    """Convert PyTorch tensor to list.

    Args:
        t: PyTorch tensor to convert.

    Returns:
        List with same content as given tensor.
    """
    return t.cpu().detach().tolist()


class MetricsTracker:
    """Collects minibatch data to compute per-epoch metrics."""

    def __init__(self):
        """Initialize lists to save minibatch data."""
        self.losses = []
        self.labels = []
        self.preds = []

    def extend(self, losses, labels, preds):
        """Save minibatch data."""
        self.losses.extend(_tensor_to_list(losses))
        self.labels.extend(_tensor_to_list(labels))
        self.preds.extend(_tensor_to_list(preds))

    def get_metrics(self, prefix=""):
        """Compute per-epoch metrics from saved minibatch data."""
        return {
            f"{prefix}loss": np.mean(self.losses),
            f"{prefix}acc": np.mean(np.array(self.preds) == np.array(self.labels)),
        }
