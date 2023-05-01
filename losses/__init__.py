"""Initialize loss."""
import logging

import torch.nn as nn


def init_loss(config):
    """Initialize loss.

    Args:
        config: Configuration for the loss.

    Returns:
        Loss specified by arguments.
    """
    if config.name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(reduction="none")

    logging.error(f"Could not create loss {config.name}.")
    raise ValueError(f"Could not create loss {config.name}.")
