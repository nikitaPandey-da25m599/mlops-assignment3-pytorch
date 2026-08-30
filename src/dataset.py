"""
src/dataset.py
--------------
Data loading helpers for CIFAR-10 using torchvision.
Provides ``get_transforms`` and ``get_dataloaders``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel-wise mean and std (computed on training split).
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the appropriate transform pipeline.

    Training transforms include random horizontal flip and random crop for
    data augmentation. Validation transforms apply only normalisation.

    Args:
        train: ``True`` for the training split, ``False`` for validation.

    Returns:
        A ``transforms.Compose`` object.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )


def get_single_image_transform() -> transforms.Compose:
    """Return the transform to apply to a single PIL image at inference time."""
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for CIFAR-10.

    The dataset is automatically downloaded to ``data_dir`` if it is not
    already present.

    Args:
        data_dir:    Root directory where CIFAR-10 will be stored / loaded from.
        batch_size:  Mini-batch size for both loaders.
        num_workers: Number of parallel data-loading workers.
        pin_memory:  Pin host memory for faster GPU transfers.

    Returns:
        A ``(train_loader, val_loader)`` tuple.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    # Avoid pin_memory on CPU-only machines to suppress warnings.
    _pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
    )

    return train_loader, val_loader


# CIFAR-10 class labels (index → name)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
