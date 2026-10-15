"""Run FLIR audit: python scripts/audit_flir.py --root data/raw/flir"""
import argparse
import json
from pathlib import Path
from utitherm.data.flir_audit import inspect_flir


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only FLIR ADAS inventory and COCO schema audit")
    ap.add_argument("--root", type=Path, default=Path("data/raw/flir"))
    ap.add_argument("--output", type=Path, default=Path("reports/flir_audit.json"))
    ap.add_argument("--sample-size", type=int, default=20)
    args = ap.parse_args()
    report = inspect_flir(args.root, args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nAudit saved: {args.output}")
    if report["image_count"] == 0 or not report["coco_annotation_files"]:
        raise SystemExit("Dataset validation incomplete: missing images or MS COCO annotations.")


if __name__ == "__main__":
    main()
