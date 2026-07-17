"""
emotion_module/model.py
------------------------
Small PyTorch CNN for 7-class facial emotion classification.

Architecture:
    Input:  (B, 1, 48, 48)  -- grayscale 48x48 face crop
    Block 1: Conv(32) -> BN -> ReLU -> MaxPool(2)   -> (B, 32, 24, 24)
    Block 2: Conv(64) -> BN -> ReLU -> MaxPool(2)   -> (B, 64, 12, 12)
    Block 3: Conv(128)-> BN -> ReLU -> MaxPool(2)   -> (B, 128, 6, 6)
    Flatten -> 128*6*6 = 4608
    FC1: 4608 -> 512,  Dropout(0.4), ReLU
    FC2: 512  -> 256,  Dropout(0.3), ReLU
    FC3: 256  -> 7    (logits; use CrossEntropyLoss during training)

Emotion class order (must match FER2013 label encoding):
    0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    """Lightweight convolutional net for facial emotion recognition."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 7) -> EmotionCNN:
    return EmotionCNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    dummy = torch.zeros(1, 1, 48, 48)
    out = m(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Total params : {count_parameters(m):,}")
    assert out.shape == (1, 7)
    print("Model smoke-test PASSED")
