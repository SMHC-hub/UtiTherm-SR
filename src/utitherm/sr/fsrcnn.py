"""Lightweight FSRCNN x2 for 8-bit grayscale thermal super-resolution.

Input [B,1,H,W] scaled to [0,1]; output [B,1,2H,2W] (unclipped).
Training degradation is *synthetic*; real-sensor thermal fidelity is not claimed.
"""
from __future__ import annotations

import torch
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, scale: int = 2, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        if scale != 2 or d <= 0 or not (0 < s <= d) or m < 1:
            raise ValueError("Require scale=2, d>0, 0<s<=d, and m>=1")
        layers: list[nn.Module] = [
            nn.Conv2d(1, d, 5, padding=2), nn.PReLU(d),
            nn.Conv2d(d, s, 1), nn.PReLU(s),
        ]
        for _ in range(m):
            layers += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
        layers += [nn.Conv2d(s, d, 1), nn.PReLU(d)]
        self.features = nn.Sequential(*layers)
        self.upsample = nn.ConvTranspose2d(d, 1, 9, stride=scale, padding=4,
                                           output_padding=scale - 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected grayscale BCHW input with one channel")
        return self.upsample(self.features(x))
