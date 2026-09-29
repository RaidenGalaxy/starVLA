"""Audit probe preprocessing against Qwen-OFT's native prediction path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.vla_layer_token_sensitivity.qwen_oft_probe import (
    _build_conditioned_inputs,
    _forward,
    _load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = json.loads(next(line for line in args.manifest.read_text().splitlines() if line.strip()))
    model = _load_model(
        SimpleNamespace(checkpoint=args.checkpoint, config=args.config, base_vlm=args.base_vlm)
    )
    state = np.asarray(sample["normalized_state"], dtype=np.float32)
    image = Image.open(sample["image_path"]).convert("RGB")
    configured_size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    if configured_size is None:
        configured_size = getattr(model.config.datasets.vla_data, "image_size", None)
    if configured_size is None:
        raise ValueError("checkpoint config has neither obs_image_size nor image_size")
    configured_size = tuple(int(value) for value in configured_size)
    resized = image.resize(configured_size)

    probe_inputs = _build_conditioned_inputs(
        model,
        Path(sample["image_path"]),
        str(sample["instruction"]),
        state=state,
    )
    _, probe_actions = _forward(model, probe_inputs)
    example = {
        "image": [resized],
        "lang": str(sample["instruction"]),
        "state": state.reshape(1, -1),
    }
    native_actions = model.predict_action([example])["normalized_actions"]
    max_abs = float(np.max(np.abs(probe_actions - native_actions)))
    payload = {
        "status": "pass" if max_abs <= args.tolerance else "fail",
        "sample_id": sample["sample_id"],
        "original_image_size": list(image.size),
        "configured_image_size": list(configured_size),
        "probe_action_shape": list(probe_actions.shape),
        "native_action_shape": list(native_actions.shape),
        "max_abs_action_difference": max_abs,
        "tolerance": args.tolerance,
        "note": (
            "Native Qwen-OFT checks obs_image_size only; this audit pre-resizes to the checkpoint's "
            "image_size so both paths receive the training-resolution image."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
