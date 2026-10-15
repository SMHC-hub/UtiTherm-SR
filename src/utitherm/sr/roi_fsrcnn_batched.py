"""Step 6C3: one-forward-pass FSRCNN x2 for selected thermal ROIs.

Each 160x128 HR grid tile corresponds to an 80x64 LR tile. Extract a
112x96 LR window (with halo=16) for *every* tile, sliding the window
inside the genuine image boundary rather than padding at the edge.
Thus all crops stack to [N,1,96,112] and run through one FSRCNN pass.

A shifted window is intentionally larger than the clipped windows in
Step 6C2; input pixel count is not an actual FLOP or speed measurement.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from utitherm.sr.roi_bicubic import _blend, _check_tile, LR_SHAPE
from utitherm.sr.roi_fsrcnn import _validate_frame


def roi_window(tile: tuple[int, int, int, int], halo: int = 16
               ) -> tuple[int, int, int, int]:
    """LR crop (left,top,right,bottom), preserving true border conditions."""
    _check_tile(tile)
    if not isinstance(halo, int) or halo < 10:
        raise ValueError('FSRCNN context halo must be integer >=10 LR pixels')
    x0, y0, x1, y1 = tile
    if (x1 - x0, y1 - y0) != (160, 128):
        raise ValueError('Step 6C3 supports the fixed 4x4 tile grid only')
    width, height = 80 + 2 * halo, 64 + 2 * halo
    if width > LR_SHAPE[1] or height > LR_SHAPE[0]:
        raise ValueError('Context halo too large for the LR image')
    xa, ya = x0 // 2, y0 // 2
    # Slide the whole window inside the frame at edges; never pad the pixels.
    left = max(0, min(xa - halo, LR_SHAPE[1] - width))
    top = max(0, min(ya - halo, LR_SHAPE[0] - height))
    return left, top, left + width, top + height


def batched_fsrcnn_tiles(lr: np.ndarray,
                         tiles: list[tuple[int, int, int, int]],
                         selected: list[int], model: torch.nn.Module,
                         device: torch.device, halo: int = 16
                         ) -> tuple[dict[int, np.ndarray], int]:
    """Compute all selected tiles in exactly ONE network forward pass.

    Returns patches keyed by tile index and number of input LR pixels.
    No GPU work is done if selection is invalid; do not invoke from timing
    code with an empty selection.
    """
    _validate_frame(lr)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select at least one distinct tile')
    if any(not isinstance(i, int) or not 0 <= i < len(tiles) for i in selected):
        raise ValueError('Tile selection out of range')
    for i in selected:
        _check_tile(tiles[i])
    rects = [tiles[i] for i in selected]
    for pos, a in enumerate(rects):
        for b in rects[pos + 1:]:
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                raise ValueError('Selected tiles must not overlap')

    windows = [roi_window(rect, halo) for rect in rects]
    crops = [lr[t:b, l:r] for l, t, r, b in windows]
    if any(crop.shape != crops[0].shape for crop in crops):
        raise RuntimeError('All LR crop shapes must match for batching')

    stack = np.stack(crops).astype(np.float32)[:, None, :, :]
    x = torch.from_numpy(stack).to(device).div_(255.0)
    with torch.inference_mode():
        y = model(x)
        expected = (len(selected), 1, stack.shape[2] * 2, stack.shape[3] * 2)
        if tuple(y.shape) != expected:
            raise ValueError(f'Unexpected neural output shape: {tuple(y.shape)} != {expected}')
        if not bool(torch.isfinite(y).all()):
            raise ValueError('Nonfinite FSRCNN output')
        result = y.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()

    patches: dict[int, np.ndarray] = {}
    for k, (i, rect, win) in enumerate(zip(selected, rects, windows)):
        x0, y0, x1, y1 = rect
        left, top, _, _ = win
        ox, oy = x0 - 2 * left, y0 - 2 * top
        patch = result[k, 0, oy:oy + y1 - y0, ox:ox + x1 - x0]
        if patch.shape != (y1-y0, x1-x0):
            raise RuntimeError(f'Batched crop did not contain tile {i}')
        patches[i] = patch
    return patches, int(sum(c.size for c in crops))


def render_batched_fsrcnn(lr: np.ndarray,
                           tiles: list[tuple[int, int, int, int]],
                           selected: list[int], model: torch.nn.Module,
                           device: torch.device, feather: int = 8,
                           halo: int = 16) -> tuple[np.ndarray, int]:
    """Match Step 6C2's linear background and feathered SR blending."""
    _validate_frame(lr)
    if not isinstance(feather, int) or feather < 0:
        raise ValueError('Feather must be a nonnegative integer')
    patches, processed = batched_fsrcnn_tiles(lr, tiles, selected, model, device, halo)
    out = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_LINEAR)
    for i in selected:
        _blend(out, patches[i], tiles[i], feather)
    return out, processed
