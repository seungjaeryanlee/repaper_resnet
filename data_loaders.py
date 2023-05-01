"""Initialize data loader."""
from torch.utils.data import DataLoader


def init_data_loader(dataset, sampler, config):
    """Initialize data loader.

    Args:
        dataset: Dataset to load.
        sampler: Sampler defining how samples are generated.
        config: Additional parameters for the data loader.
            batch_size: Number of samples per batch.
            num_workers: Number of workers to get items in parallel.
            pin_memory: Flag to enable pinned memory.

    Returns:
        Data Loader specified by arguments.
    """
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
