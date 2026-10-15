"""Deterministic ×2 degradation and interpolation for baseline experiments."""
from PIL import Image


def downsample(image: Image.Image, scale: int = 2) -> Image.Image:
    if scale < 2:
        raise ValueError("scale must be at least 2")
    if image.width % scale or image.height % scale:
        raise ValueError("input width and height must be divisible by scale")
    return image.resize((image.width // scale, image.height // scale), Image.Resampling.BICUBIC)


def bicubic_upscale(image: Image.Image, scale: int = 2) -> Image.Image:
    if scale < 2:
        raise ValueError("scale must be at least 2")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.BICUBIC)
