"""Export a self-contained packed INT4/INT8 Qwen2.5-VL checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import EXPORT_CONFIGS, export_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", choices=EXPORT_CONFIGS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.group_size <= 0 or args.max_shard_bytes <= 0:
        parser.error("group size and max shard bytes must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = export_checkpoint(
            model_path=args.model_path,
            source_run=args.source_run,
            config_name=args.config,
            output_dir=args.output_dir,
            group_size=args.group_size,
            scale_dtype=args.scale_dtype,
            max_shard_bytes=args.max_shard_bytes,
            resume=args.resume,
        )
    except Exception as exc:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        error = args.output_dir.with_suffix(".error.json")
        error.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
        raise
    print(f"Exported {result['linear_modules']} packed Linear modules to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

