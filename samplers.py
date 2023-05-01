"""Initialize sampler."""
import logging

from torch.utils.data import RandomSampler, SequentialSampler


def init_sampler(config, dataset):
    """Initialize sampler with given name.

    Args:
        config: Configuration defining the sampler.
        dataset: Dataset to sample from.

    Returns:
        Sampler with name given as argument.
    """
    if config.name == "SequentialSampler":
        return SequentialSampler(dataset)
    elif config.name == "RandomSampler":
        return RandomSampler(dataset)

    logging.error(f"Could not create sampler {config.name}.")
    raise ValueError(f"Could not create sampler {config.name}.")
