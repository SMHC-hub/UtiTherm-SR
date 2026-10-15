"""Exact (for OpenCV INTER_CUBIC at x2) ROI-only upsampling.

Input: 320x256 grayscale LR image. Output: 640x512 grayscale image with
selected nonoverlapping 4x4 grid tiles enhanced and a feathered seam.

The whole-frame INTER_LINEAR base remains necessary; only the INTER_CUBIC
upsampling is restricted to selected tile neighborhoods. A two-source-pixel
halo reproduces the same interpolation as OpenCV whole-frame resize.
"""
from __future__ import annotations

import cv2
import numpy as np

LR_SHAPE = (256, 320)
HR_SHAPE = (512, 640)


def _check_tile(tile: tuple[int, int, int, int]):
    if len(tile) != 4 or any(not isinstance(v, (int, np.integer)) for v in tile):
        raise ValueError('Tile must contain four integer coordinates')
    x0, y0, x1, y1 = tile
    if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 512):
        raise ValueError('Tile outside the 640x512 output')
    if any(v % 2 for v in tile):
        raise ValueError('ROI tile coordinates must align with the x2 LR pixel grid')


def cubic_tile_from_lr(lr: np.ndarray, tile: tuple[int, int, int, int],
                       halo: int = 2) -> tuple[np.ndarray, int]:
    """Interpolate only a tile plus halo; return HR tile and cubic output pixels.

    Halo >=2 LR pixels is essential to match the 4-tap cubic footprint.
    The source cropping is LR-pixel aligned, preserving half-pixel positions.
    """
    if lr.shape != LR_SHAPE or lr.dtype != np.uint8 or lr.ndim != 2:
        raise ValueError('Require 320x256 grayscale uint8 LR')
    _check_tile(tile)
    if not isinstance(halo, int) or halo < 2:
        raise ValueError('INTER_CUBIC requires a halo of at least 2 LR pixels')
    x0, y0, x1, y1 = tile
    xa, ya, xb, yb = x0 // 2, y0 // 2, x1 // 2, y1 // 2
    left, top = max(0, xa - halo), max(0, ya - halo)
    right, bottom = min(320, xb + halo), min(256, yb + halo)
    crop = lr[top:bottom, left:right]
    zoom = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                      interpolation=cv2.INTER_CUBIC)
    sy, sx = 2 * (ya - top), 2 * (xa - left)
    tile_hr = zoom[sy:sy+(y1-y0), sx:sx+(x1-x0)]
    assert tile_hr.shape == (y1-y0, x1-x0)
    return tile_hr, int(zoom.size)


def _blend(dst: np.ndarray, patch: np.ndarray, tile: tuple[int,int,int,int],
           feather: int):
    """Match probe_roi_utility.enhanced_tile() exactly, without copying whole canvas."""
    x0,y0,x1,y1 = tile
    roi = dst[y0:y1,x0:x1]
    if feather == 0:
        roi[:] = patch
        return
    th, tw = y1-y0, x1-x0
    yy = np.minimum(np.arange(th), np.arange(th)[::-1]).astype(np.float32)
    xx = np.minimum(np.arange(tw), np.arange(tw)[::-1]).astype(np.float32)
    ay = np.clip(yy / feather, 0.0, 1.0)
    ax = np.clip(xx / feather, 0.0, 1.0)
    alpha = ay[:,None] * ax[None,:]
    result = np.rint(roi.astype(np.float32) * (1.0 - alpha) +
                     patch.astype(np.float32) * alpha)
    roi[:] = np.clip(result, 0, 255).astype(np.uint8)


def render_selected(lr: np.ndarray,
                    tiles: list[tuple[int,int,int,int]], selected: list[int],
                    feather: int = 8, halo: int = 2,
                    roi_only: bool = True) -> tuple[np.ndarray, int]:
    """Render selected tiles; return (HR image, bicubic output pixels processed).

    roi_only=False is a matched, optimized full-frame-cubic control; both
    implementations share the exact same blending code.
    """
    if lr.shape != LR_SHAPE or lr.dtype != np.uint8 or lr.ndim != 2:
        raise ValueError('Require 320x256 grayscale uint8 LR')
    if not isinstance(feather, int) or feather < 0:
        raise ValueError('feather must be nonnegative integer')
    if not selected or len(selected) != len(set(selected)) or not all(
            isinstance(i, int) and 0 <= i < len(tiles) for i in selected):
        raise ValueError('Select unique, valid tile indices')
    if not isinstance(halo, int) or halo < 2:
        raise ValueError('halo must be >=2 LR pixels')
    rects = [tiles[i] for i in selected]
    for rect in rects:
        _check_tile(rect)
    for j, a in enumerate(rects):
        for b in rects[j+1:]:
            if max(a[0],b[0]) < min(a[2],b[2]) and max(a[1],b[1]) < min(a[3],b[3]):
                raise ValueError('Selected tiles overlap')
    out = cv2.resize(lr, (640,512), interpolation=cv2.INTER_LINEAR)
    cubic = None if roi_only else cv2.resize(lr, (640,512), interpolation=cv2.INTER_CUBIC)
    pixels = 0 if roi_only else int(cubic.size)
    for rect in rects:
        x0,y0,x1,y1 = rect
        if roi_only:
            patch, count = cubic_tile_from_lr(lr, rect, halo)
            pixels += count
        else:
            patch = cubic[y0:y1,x0:x1]
        _blend(out,patch,rect,feather)
    return out, pixels
