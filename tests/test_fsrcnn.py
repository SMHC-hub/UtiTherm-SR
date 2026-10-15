"""Small, fast correctness tests. Run in existing UtiTherm Docker environment."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from utitherm.sr.fsrcnn import FSRCNN
from scripts.train_fsrcnn import ThermalPairs, group_split, mse_psnr, sequence_id


def test_fsrcnn_x2_shapes_and_finite():
    torch.manual_seed(42)
    model = FSRCNN()
    inp = torch.rand(2, 1, 16, 24)
    output = model(inp)
    assert output.shape == (2, 1, 32, 48)
    assert torch.isfinite(output).all()


def test_fsrcnn_backprop_reaches_upsampler():
    model = FSRCNN(d=8, s=4, m=1)
    pred = model(torch.rand(1, 1, 8, 8))
    pred.square().mean().backward()
    assert model.upsample.weight.grad is not None
    assert torch.isfinite(model.upsample.weight.grad).all()


def test_group_split_excludes_same_video():
    imgs = [Path(f"video-{k}-frame-{i:06d}.jpg") for k in 'abcd' for i in range(5)]
    a, b = group_split(imgs, seed=42, val_fraction=0.25)
    assert a and b
    assert set(a).isdisjoint(b)
    assert {sequence_id(p) for p in a}.isdisjoint({sequence_id(p) for p in b})
    assert group_split(imgs, 42, 0.25) == (a, b)


def test_patch_alignment_matches_pil_bicubic(tmp_path):
    # Whole-image downsample first, then crop; HR patch uses x2 coordinates.
    data = np.arange(96*128, dtype=np.uint32).reshape(96, 128).astype(np.uint8)
    file = tmp_path / "video-one-frame-000001.png"
    Image.fromarray(data).save(file)
    ds = ThermalPairs([file], lr_patch=16, training=False, seed=42)
    lr, hr = ds[0]
    digest = __import__("hashlib").sha256(f"{file.name}|42".encode()).digest()
    x = int.from_bytes(digest[:8], 'big') % (64 - 16 + 1)
    y = int.from_bytes(digest[8:16], 'big') % (48 - 16 + 1)
    expected_lr = Image.fromarray(data).resize((64, 48), Image.Resampling.BICUBIC)
    expected_lr = np.asarray(expected_lr.crop((x, y, x+16, y+16)), dtype=np.float32) / 255
    expected_hr = data[2*y:2*(y+16), 2*x:2*(x+16)].astype(np.float32)/255
    assert lr.shape == (1, 16, 16) and hr.shape == (1, 32, 32)
    assert np.allclose(lr.numpy()[0], expected_lr)
    assert np.allclose(hr.numpy()[0], expected_hr)


def test_psnr_and_bad_architecture():
    assert abs(mse_psnr(0.01) - 20) < 1e-9
    try:
        FSRCNN(scale=3)
    except ValueError:
        pass
    else:
        raise AssertionError("Only x2 is supported")
