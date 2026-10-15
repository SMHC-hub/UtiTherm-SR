"""FSRCNN x2 region-only inference with context halo (thermal grayscale).

Only the selected LR crops are sent through FSRCNN in ``mode='roi'``.
All geometry is fixed to the existing Step 6 4x4, 320x256 -> 640x512 setup.
The output starts from a full-frame linear interpolation baseline and blends
only selected neural-SR tiles, matching the Step 6A feather policy.

This does NOT guarantee an FSRCNN-trained-on-Bicubic-labels selector is useful.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from utitherm.sr.roi_bicubic import _blend, _check_tile, LR_SHAPE, HR_SHAPE


def _validate_frame(lr: np.ndarray) -> None:
    if lr.shape != LR_SHAPE or lr.dtype != np.uint8 or lr.ndim != 2:
        raise ValueError('Expected 320x256 grayscale uint8 LR input')


def _model_gray_x2(model: torch.nn.Module, src: np.ndarray, device: torch.device) -> np.ndarray:
    """Inference matches Step 4C's preprocessing and quantization."""
    x = torch.from_numpy(np.ascontiguousarray(src, dtype=np.float32))[None, None].to(device) / 255.0
    with torch.inference_mode():
        y = model(x)
        shape = (1, 1, src.shape[0] * 2, src.shape[1] * 2)
        if tuple(y.shape) != shape:
            raise ValueError(f'FSRCNN shape {tuple(y.shape)} != {shape}')
        if not torch.isfinite(y).all():
            raise ValueError('Non-finite FSRCNN output')
        y = y.clamp(0, 1).mul(255).round().to(torch.uint8)
    return y.cpu().numpy()[0, 0]


def full_fsrcnn(lr: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    _validate_frame(lr)
    return _model_gray_x2(model, lr, device)


def fsrcnn_tile(lr: np.ndarray, tile: tuple[int, int, int, int],
                model: torch.nn.Module, device: torch.device,
                halo: int = 16) -> tuple[np.ndarray, int]:
    """Return only the desired HR tile, using real surrounding LR pixels.

    Halo is essential because FSRCNN's deep convolutions have a wide receptive
    field. Cropping at the image edges (no artificial outside padding) preserves
    its genuine boundary behavior. The cropped output is expected to closely
    match full-frame SR and is *checked* by the benchmark.
    """
    _validate_frame(lr)
    _check_tile(tile)
    if not isinstance(halo, int) or halo < 10:
        raise ValueError('FSRCNN x2 requires halo >=10 LR pixels for this network')
    x0, y0, x1, y1 = tile
    xa, ya, xb, yb = x0 // 2, y0 // 2, x1 // 2, y1 // 2
    left, top = max(0, xa - halo), max(0, ya - halo)
    right, bottom = min(LR_SHAPE[1], xb + halo), min(LR_SHAPE[0], yb + halo)
    crop = lr[top:bottom, left:right]
    enlarged = _model_gray_x2(model, crop, device)
    ox, oy = 2 * (xa - left), 2 * (ya - top)
    patch = enlarged[oy:oy + y1 - y0, ox:ox + x1 - x0]
    if patch.shape != (y1 - y0, x1 - x0):
        raise RuntimeError('Neural crop did not contain the entire selected tile')
    return patch, int(crop.size)


def render_fsrcnn(lr: np.ndarray, tiles: list[tuple[int, int, int, int]],
                  selected: list[int], model: torch.nn.Module,
                  device: torch.device, mode: str, feather: int = 8,
                  halo: int = 16) -> tuple[np.ndarray, int]:
    """Return (HR image, total LR model-input pixels processed).

    Supported modes: 'linear', 'full', 'selective_fullframe', 'selective_roi'.
    This counts model input pixels, NOT FLOPs or actual speed.
    """
    _validate_frame(lr)
    if mode not in ('linear', 'full', 'selective_fullframe', 'selective_roi'):
        raise ValueError(f'Unknown FSRCNN render mode: {mode}')
    if mode.startswith('selective'):
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Selection must contain unique tiles')
        if any(not isinstance(i, int) or not 0 <= i < len(tiles) for i in selected):
            raise ValueError('Invalid selected tile index')
        for i in selected:
            _check_tile(tiles[i])
    if not isinstance(feather, int) or feather < 0:
        raise ValueError('feather must be nonnegative integer')
    if mode == 'linear':
        return cv2.resize(lr, (640, 512), interpolation=cv2.INTER_LINEAR), 0
    if mode == 'full':
        return full_fsrcnn(lr, model, device), int(lr.size)
    base = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_LINEAR)
    processed = 0
    full = None
    if mode == 'selective_fullframe':
        full = full_fsrcnn(lr, model, device)
        processed = int(lr.size)
    for i in selected:
        rect = tiles[i]
        if full is not None:
            x0, y0, x1, y1 = rect
            patch = full[y0:y1, x0:x1]
        else:
            patch, n = fsrcnn_tile(lr, rect, model, device, halo=halo)
            processed += n
        _blend(base, patch, rect, feather)
    return base, processed


def selected_region_difference(a: np.ndarray, b: np.ndarray,
                               tiles: list[tuple[int, int, int, int]],
                               selected: list[int]) -> dict[str, float | int]:
    """Numerically audit ROI-only vs full-model reference, no silent tolerance."""
    if a.shape != HR_SHAPE or b.shape != HR_SHAPE or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('Require same-shaped uint8 640x512 outputs')
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mask = np.zeros(HR_SHAPE, dtype=bool)
    for i in selected:
        x0, y0, x1, y1 = tiles[i]
        mask[y0:y1, x0:x1] = True
    relevant = diff[mask]
    return {
        'max_abs_difference': int(diff.max()),
        'changed_output_pixels': int(np.count_nonzero(diff)),
        'mean_abs_difference_in_selected_area': float(relevant.mean()) if relevant.size else 0.0,
    }
