"""Initialize transforms for data augmentation."""
import inspect
import logging

from torchvision import transforms

REPR_TO_CLASS = {k: v for k, v in inspect.getmembers(transforms) if inspect.isclass(v)}


def init_transform(config):
    """Initialize transforms.

    Args:
        config: Configuration for the transform.
            name: Name of data augmentation.
            reprs: List of transforms as a string.

    Returns:
        Transforms specified by arguments.
    """
    if config.name == "Identity":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    elif config.name == "FlipCrop":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    elif config.name == "Custom":
        transform_list = [eval(repr_, REPR_TO_CLASS) for repr_ in config.reprs]
        return transforms.Compose(transform_list)

    logging.error(f"Could not create transform with name {config.name}")
    raise ValueError(f"Could not create transform with name {config.name}")
