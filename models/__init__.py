"""Initialize models."""
import logging

from .resnet import ResNet


def init_model(config):
    """Initialize model based on model name.

    Args:
        config: Configuration for the model.

    Returns:
        Model specified by arguments.
    """
    if config.name == "ResNet_32":
        return ResNet(num_blocks=[5, 5, 5], num_classes=config.num_classes)

    logging.error(f"Could not create model {config.name}.")
    raise ValueError(f"Could not create model {config.name}.")
