"""Initialize optimizers."""
import logging

import torch.optim as optim


def init_optimizer(parameters, config):
    """Initialize optimizer based on config.

    Args:
        parameters: Parameters to optimize.
        config: Specifics defining the optimizer.
            name: Name of the optimizer.

    Returns:
        Optimizer specified by config.
    """
    if config.name == "SGD":
        return optim.SGD(
            parameters,
            lr=config.SGD.lr,
            momentum=config.SGD.momentum,
            weight_decay=config.SGD.weight_decay,
        )

    logging.error(f"Could not create optimizer {config.name}.")
    raise ValueError(f"Could not create optimizer {config.name}.")
