"""
src/model.py
------------
Defines the image-classification model backed by a pre-trained ResNet-18
(torchvision). The final fully-connected layer is replaced to match
`num_classes` so the network can be fine-tuned end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    """Return a torchvision model with a custom classification head.

    Args:
        architecture: One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``.
        num_classes:  Number of output classes (10 for CIFAR-10).
        pretrained:   Whether to load ImageNet weights.

    Returns:
        An ``nn.Module`` ready for fine-tuning or inference.
    """
    architecture = architecture.lower()

    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif architecture == "resnet34":
        from torchvision.models import ResNet34_Weights

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif architecture == "resnet50":
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            "Choose from: resnet18, resnet34, resnet50."
        )

    # Replace the final FC layer to match the target number of classes.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def load_checkpoint(
    checkpoint_path: str,
    architecture: str = "resnet18",
    num_classes: int = 10,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a saved checkpoint and return an evaluation-ready model.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        architecture:    Architecture string passed to :func:`get_model`.
        num_classes:     Number of output classes.
        device:          Target device; defaults to CPU.

    Returns:
        Model in ``eval()`` mode with weights loaded.
    """
    if device is None:
        device = torch.device("cpu")

    model = get_model(architecture=architecture, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state-dict and our richer checkpoint dict.
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
