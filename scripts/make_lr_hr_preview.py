"""Make 8-bit synthetic LR/HR/bicubic preview; does not touch annotations."""
import argparse
import json
from pathlib import Path
from utitherm.data.pairs import prepare_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Actual 8-bit thermal image folder identified by audit")
    ap.add_argument("--out", type=Path, default=Path("data/processed/pairs_preview"))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=20)
    args = ap.parse_args()
    report = prepare_pairs(args.src, args.out, scale=args.scale, max_images=args.max_images)
    print(json.dumps({k: v for k, v in report.items() if k != "entries"}, indent=2))
    print("Pairs and manifest saved. These are synthetic pairs, not genuine FLIR LR-HR sensor pairs.")


if __name__ == "__main__":
    main()
