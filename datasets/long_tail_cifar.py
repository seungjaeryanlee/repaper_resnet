"""Define long-tail variants of CIFAR-10/100 datasets.

TODO: CIFAR10LT and CIFAR100LT has a lot of duplicate code. Consider refactoring.
"""
import os
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100


def _read_indices(relative_filepath):
    with open(os.path.join(os.path.dirname(__file__), relative_filepath), "r") as f:
        return [int(line.strip()) for line in f.readlines()]


class CIFAR10LT(Dataset):
    """Long-tail variant of CIFAR-10 dataset.

    Attributes:
        original_dataset: Original CIFAR-10 dataset.
        ir: Imbalance ratio measuring class imbalance.
        indices: Indices of CIFAR-10 images used to create long-tail subset.
    """

    ir_to_indices = {
        10: _read_indices("cifar10ir10.indices"),
        20: _read_indices("cifar10ir20.indices"),
        50: _read_indices("cifar10ir50.indices"),
        100: _read_indices("cifar10ir100.indices"),
    }

    def __init__(
        self,
        ir: int,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """Initialize CIFAR-10-LT dataset.

        Args:
            ir: Imbalance ratio.
            root: Where CIFAR-10 dataset is saved.
            train: Flag indicating training or validation dataset.
            transform: Transforms for the images.
            target_transform: Transform for the targets.
            download: If True, download CIFAR-10 data if it does not exist.
        """
        assert train, "CIFAR10LT only works on training datsaet"

        self.original_dataset = CIFAR10(
            root, train, transform, target_transform, download
        )
        self.ir = ir

        if ir not in CIFAR10LT.ir_to_indices:
            raise ValueError(f"Imbalance ratio of {ir} is not supported in CIFAR10.")

        self.indices = CIFAR10LT.ir_to_indices[ir]
        self.dataset = Subset(self.original_dataset, self.indices)
        self.data = self.original_dataset.data[self.indices]
        self.targets = np.array(self.original_dataset.targets)[self.indices].tolist()

    def __getitem__(self, idx):
        """Get example from given index."""
        return self.dataset[idx]

    def __len__(self):
        """Get number of examples."""
        return len(self.targets)


class CIFAR100LT(Dataset):
    """Long-tail variant of CIFAR-100 dataset.

    Attributes:
        original_dataset: Original CIFAR-100 dataset.
        ir: Imbalance ratio measuring class imbalance.
        indices: Indices of CIFAR-100 images used to create long-tail subset.
    """

    ir_to_indices = {
        10: _read_indices("cifar100ir10.indices"),
        20: _read_indices("cifar100ir20.indices"),
        50: _read_indices("cifar100ir50.indices"),
        100: _read_indices("cifar100ir100.indices"),
    }

    def __init__(
        self,
        ir: int,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """Initialize CIFAR-100-LT dataset.

        Args:
            ir: Imbalance ratio.
            root: Where CIFAR-100 dataset is saved.
            train: Flag indicating training or validation dataset.
            transform: Transforms for the images.
            target_transform: Transform for the targets.
            download: If True, download CIFAR-100 data if it does not exist.
        """
        assert train, "CIFAR100LT only works on training datsaet"

        self.original_dataset = CIFAR100(
            root, train, transform, target_transform, download
        )
        self.ir = ir

        if ir not in CIFAR100LT.ir_to_indices:
            raise ValueError(f"Imbalance ratio of {ir} is not supported in CIFAR100.")

        self.indices = CIFAR100LT.ir_to_indices[ir]
        self.dataset = Subset(self.original_dataset, self.indices)
        self.data = self.original_dataset.data[self.indices]
        self.targets = np.array(self.original_dataset.targets)[self.indices].tolist()

    def __getitem__(self, idx):
        """Get example from given index."""
        return self.dataset[idx]

    def __len__(self):
        """Get number of examples."""
        return len(self.targets)
