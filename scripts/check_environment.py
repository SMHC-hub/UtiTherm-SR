"""Validate RTX 5070 access and resource visibility within Docker."""
from __future__ import annotations
import argparse
import json
import platform
import torch
import psutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is not available")
    args = ap.parse_args()
    has_cuda = torch.cuda.is_available()
    report = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": has_cuda,
        "ram_available_gib": round(psutil.virtual_memory().available / (1024**3), 2),
        "ram_total_visible_gib": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    if has_cuda:
        report.update({
            "gpu": torch.cuda.get_device_name(0),
            "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        })
        x = torch.randn(128, 128, device="cuda")
        report["gpu_tensor_test"] = bool(torch.isfinite(x @ x).all().item())
        torch.cuda.synchronize()
    print(json.dumps(report, indent=2))
    if args.require_gpu and not has_cuda:
        raise SystemExit("FAILED: CUDA GPU not available in Docker. Check Windows drivers, WSL2 and Docker Desktop GPU support.")
    if has_cuda and not report["gpu_tensor_test"]:
        raise SystemExit("FAILED: GPU tensor operations returned nonfinite values")


if __name__ == "__main__":
    main()
