"""Generate the deterministic 8-calibration/16-evaluation tabletop manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_tiny_dataset, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_tiny_dataset(args.output_dir, seed=args.seed)
    samples = load_manifest(manifest)
    print(f"Wrote {len(samples)} samples to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

