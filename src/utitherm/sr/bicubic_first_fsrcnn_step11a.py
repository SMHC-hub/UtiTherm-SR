"""Bicubic-background selective Large FSRCNN renderer for Step 11A.

This is intentionally NOT the Step 6C3 renderer: that one composites
FSRCNN patches over LINEAR, whereas this one uses whole-frame BICUBIC.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from utitherm.sr.roi_bicubic import _blend
from utitherm.sr.roi_fsrcnn_batched import batched_fsrcnn_tiles
from utitherm.sr.roi_fsrcnn import _validate_frame


def composite_bicubic_patches(
    bicubic: np.ndarray,
    patches: dict[int, np.ndarray],
    tiles: list[tuple[int, int, int, int]],
    selected: list[int],
    feather: int = 8,
) -> np.ndarray:
    """Blend neural patches only inside selected tiles on an immutable Bicubic base.

    ``patches`` must be *unblended neural outputs* from ``batched_fsrcnn_tiles``.
    Passing the final Step 6C3 rendered frame would incorrectly insert a
    linear background. Unselected Bicubic pixels stay byte-for-byte identical.
    """
    if bicubic.shape != (512, 640) or bicubic.dtype != np.uint8 or bicubic.ndim != 2:
        raise ValueError('Require 640x512 uint8 full-frame Bicubic baseline')
    if not isinstance(feather, int) or feather < 0:
        raise ValueError('Feather must be a nonnegative integer')
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Select at least one distinct tile')
    if any(not isinstance(i, int) or i < 0 or i >= len(tiles) for i in selected):
        raise ValueError('Invalid tile index')
    if set(patches) != set(selected):
        raise ValueError('Patch keys must exactly match selected tile indices')
    out = bicubic.copy()
    for i in selected:
        x0, y0, x1, y1 = tiles[i]
        patch = patches[i]
        if (not isinstance(patch, np.ndarray) or patch.dtype != np.uint8 or
                patch.shape != (y1 - y0, x1 - x0)):
            raise ValueError(f'Invalid neural patch for tile {i}')
        _blend(out, patch, tiles[i], feather)
    return out


def render_bicubic_first_fsrcnn(
    lr: np.ndarray,
    tiles: list[tuple[int, int, int, int]],
    selected: list[int],
    model: torch.nn.Module,
    device: torch.device,
    feather: int = 8,
    halo: int = 10,
    bicubic: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """FSRCNN ROI(s) composited onto full Bicubic; returns (HR, processed LR pixels).

    Optional ``bicubic`` avoids repeated CPU resize in a 16-tile probe.
    Underlying GPU patch extraction is identical to Step 9G.
    """
    _validate_frame(lr)
    if halo != 10:
        raise ValueError('Step 11A feasibility protocol freezes halo=10')
    if feather != 8:
        raise ValueError('Step 11A feasibility protocol freezes feather=8')
    if bicubic is None:
        bicubic = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC)
    patches, processed = batched_fsrcnn_tiles(lr, tiles, selected, model, device, halo=halo)
    output = composite_bicubic_patches(bicubic, patches, tiles, selected, feather)
    return output, processed
