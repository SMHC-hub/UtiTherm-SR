"""UtiTherm-SR Step 5B: small RGB-free thermal tile+context utility predictor.

Four event logits: small TP gain, FP reduction, small TP loss, FP increase.
The estimate is a *ranking surrogate*, not a calibrated expected TP/FP change.
No ground-truth/detector outputs enter inference inputs.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

HEADS = ("small_gain", "fp_reduce", "small_loss", "fp_increase")


class UtilityPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.GroupNorm(4, 48), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(48 * 4 + 4, 64), nn.SiLU(), nn.Linear(64, len(HEADS)))

    def forward(self, patch_context: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if patch_context.ndim != 4 or patch_context.shape[1] != 2:
            raise ValueError("Expected [batch,2,height,width] grayscale tile and context")
        if coords.ndim != 2 or coords.shape != (patch_context.shape[0], 4):
            raise ValueError("Expected normalized [batch,4] tile coordinates")
        return self.classifier(torch.cat([self.features(patch_context), coords], dim=1))


def encode_tile(gray: np.ndarray, rect_hr: tuple[int, int, int, int],
                input_size: tuple[int, int] = (80, 64),
                hr_size: tuple[int, int] = (640, 512)) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tile/context [2,H,W] and normalized (center_x,center_y,width,height).

    rect_hr belongs to the *intervention grid*; no GT boxes are used.
    input_size=(width,height) and hr_size=(width,height).
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("Expected nonempty uint8 thermal frame")
    iw, ih = input_size
    hw, hh = hr_size
    if min(iw, ih, hw, hh) < 1:
        raise ValueError("Invalid image dimensions")
    x0, y0, x1, y1 = rect_hr
    if not (0 <= x0 < x1 <= hw and 0 <= y0 < y1 <= hh):
        raise ValueError("Invalid HR tile rectangle")
    h, w = gray.shape
    xa = max(0, min(w - 1, round(x0 * w / hw)))
    xb = max(xa + 1, min(w, round(x1 * w / hw)))
    ya = max(0, min(h - 1, round(y0 * h / hh)))
    yb = max(ya + 1, min(h, round(y1 * h / hh)))
    roi = gray[ya:yb, xa:xb]
    patch = cv2.resize(roi, (iw, ih), interpolation=cv2.INTER_AREA)
    context = cv2.resize(gray, (iw, ih), interpolation=cv2.INTER_AREA)
    frame_tensor = torch.from_numpy(np.stack([patch, context]).astype(np.float32) / 255.0)
    coords = torch.tensor([(x0 + x1) / (2 * hw), (y0 + y1) / (2 * hh),
                           (x1 - x0) / hw, (y1 - y0) / hh], dtype=torch.float32)
    return frame_tensor, coords


def outcome_labels(row: dict) -> np.ndarray:
    """Each head is an event: labels can overlap and are independently binary."""
    return np.asarray([
        int(int(row["delta_small_tp"]) > 0),
        int(int(row["delta_fp"]) < 0),
        int(int(row["delta_small_tp"]) < 0),
        int(int(row["delta_fp"]) > 0),
    ], dtype=np.float32)


def ranking_score(probabilities: np.ndarray, fp_penalty: float = 0.25) -> np.ndarray:
    """Heuristic ranking, not predicted utility magnitude or calibrated expected benefit."""
    p = np.asarray(probabilities)
    if p.shape[-1] != 4 or fp_penalty < 0:
        raise ValueError("Expected four event probabilities and positive penalty")
    return p[..., 0] - p[..., 2] + fp_penalty * (p[..., 1] - p[..., 3])
