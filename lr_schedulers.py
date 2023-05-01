"""Initialize optimizers."""
import logging

import torch.optim as optim


def init_lr_scheduler(optimizer, config):
    """Initialize learning rate schedulers.

    Args:
        optimizer: Optimizer to update learning rate.
        config: Additional parameters for the scheduler.
            name: Name of the learning rate scheduler.

    Returns:
        LR scheduler specified by arguments.
    """
    schedulers = []
    for name in config.names:
        if name == "LinearLR":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.LinearLR.start_factor,
                end_factor=1,
                total_iters=config.LinearLR.total_iters,
            )
        elif name == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.MultiStepLR.milestones,
                gamma=config.MultiStepLR.gamma,
            )
        else:
            logging.error(f"Could not create learning rate scheduler {name}.")
            raise ValueError(f"Could not create learning rate scheduler {name}.")

        schedulers.append(scheduler)

    return optim.lr_scheduler.ChainedScheduler(schedulers)
