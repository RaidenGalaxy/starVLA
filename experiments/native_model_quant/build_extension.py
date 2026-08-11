"""Build the AVX-512 W4A8 extension and record its reproducible identity."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extension import DEFAULT_BUILD_DIR, load_native_extension, write_build_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/native_model_quant/kernel_build.json"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    extension = load_native_extension(args.build_dir, verbose=args.verbose)
    payload = write_build_manifest(args.manifest, extension)
    if not payload["avx512_vnni"]:
        raise RuntimeError("extension built, but this CPU does not expose AVX-512 VNNI")
    print(f"Built {payload['extension_name']} ({payload['source_hash'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
