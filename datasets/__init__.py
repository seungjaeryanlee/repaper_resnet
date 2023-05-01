"""Initialize dataset."""
import logging

from torchvision.datasets import CIFAR10, CIFAR100


def init_dataset(config, transform, train):
    """Initialize dataset.

    Args:
        config: Configuration for dataset.
        transform: Data augmentation to apply to dataset.
        train: Flag to indicate training or test dataset.

    Returns:
        Dataset specified by arguments.
    """
    if config.name == "CIFAR10":
        return CIFAR10(root="./data", train=train, transform=transform, download=True)
    if config.name == "CIFAR100":
        return CIFAR100(root="./data", train=train, transform=transform, download=True)

    logging.error(f"Could not create dataset {config.name}.")
    raise ValueError(f"Could not create dataset {config.name}.")
