"""Resumable CPU scan of FP and globally W4-quantized Qwen-OFT layers.

For every manifest sample this runner records four model states: dense FP,
FP+one skipped layer, dense W4, and W4+one skipped layer.  W4 is a weight-only
fake quantization of all decoder Linear modules and is applied once per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.vla_layer_token_sensitivity.perturbations import (
    quantize_module_inplace,
    temporary_layer_skip,
)
from experiments.vla_layer_token_sensitivity.qwen_oft_probe import (
    GROUP_LABELS,
    GROUP_NAMES,
    _build_inputs,
    _decoder_layers,
    _forward,
    _load_model,
    _summarize,
    _token_groups,
)


SCHEMA_VERSION = 1
REFERENCE_LAYER = -1
REFERENCE_OPERATION = "reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,6,12,18,24,30,34")
    parser.add_argument("--base-states", default="fp,w4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path, limit: int | None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("manifest is empty")
    # Interleave episodes so ``--limit 4`` is a four-episode pilot rather than
    # four adjacent frames from the first episode.  The full 32-sample run is
    # unchanged and resumes the same sample keys.
    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    rows = [
        episode_rows[position]
        for position in range(max(len(values) for values in by_episode.values()))
        for episode_rows in by_episode.values()
        if position < len(episode_rows)
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    required = {"sample_id", "episode_index", "frame_index", "image_path", "instruction"}
    seen: set[str] = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
        seen.add(sample_id)
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        if not str(row["instruction"]).strip():
            raise ValueError(f"sample {sample_id} has an empty instruction")
    return rows


def _row_key(row: dict) -> tuple[str, str, int, str]:
    return (
        str(row["sample_id"]),
        str(row["base_state"]),
        int(row["layer"]),
        str(row["operation"]),
    )


def _read_existing(path: Path) -> dict[tuple[str, str, int, str], dict]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            temporary = path.with_suffix(path.suffix + ".repair")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            temporary.replace(path)
    return {_row_key(row): row for row in rows}


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _identity_summary(actions: np.ndarray) -> dict:
    return {
        "action_mse": 0.0,
        "action_max_abs": 0.0,
        "action_step_mse": [0.0] * int(np.prod(actions.shape[:-1])),
        "final_hidden_cosine_drift": {GROUP_LABELS[group]: 0.0 for group in GROUP_NAMES},
        "final_hidden_cka": {GROUP_LABELS[group]: 1.0 for group in GROUP_NAMES},
    }


def _reference_path(reference_dir: Path, sample_id: str) -> Path:
    return reference_dir / f"{sample_id}.npz"


def _save_reference(
    path: Path,
    *,
    hidden: np.ndarray,
    actions: np.ndarray,
    groups: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, hidden=hidden, actions=actions, groups=groups)
    temporary.replace(path)


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return payload["hidden"], payload["actions"], payload["groups"]


def _base_row(
    sample: dict,
    *,
    base_state: str,
    layer: int,
    operation: str,
    runtime_s: float,
    signature: dict,
) -> dict:
    fraction = sample.get("trajectory_fraction")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen_oft_fp_w4_layer_scan",
        "sample_id": str(sample["sample_id"]),
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "trajectory_fraction": None if fraction is None else float(fraction),
        "instruction": str(sample["instruction"]),
        "base_state": base_state,
        "layer": int(layer),
        "operation": operation,
        "runtime_s": float(runtime_s),
        "signature": signature,
    }


def _validate_existing(existing: dict, signature: dict) -> None:
    for row in existing.values():
        if row.get("signature") != signature:
            raise ValueError("existing output was produced by a different checkpoint/manifest/configuration")


def _fp_scan(
    *,
    model,
    samples: list[dict],
    layers,
    layer_indices: list[int],
    output: Path,
    reference_dir: Path,
    existing: dict,
    signature: dict,
    include_skips: bool,
) -> None:
    required_keys = [
        (str(sample["sample_id"]), "fp", layer, "skip")
        for sample in samples
        for layer in layer_indices
    ]
    for sample in samples:
        sample_id = str(sample["sample_id"])
        cache_path = _reference_path(reference_dir, sample_id)
        reference_key = (sample_id, "fp", REFERENCE_LAYER, REFERENCE_OPERATION)
        skip_missing = include_skips and any(
            (sample_id, "fp", layer, "skip") not in existing for layer in layer_indices
        )
        if reference_key in existing and cache_path.is_file() and not skip_missing:
            continue

        inputs = _build_inputs(model, Path(sample["image_path"]), str(sample["instruction"]))
        groups = _token_groups(model, inputs)
        started = time.perf_counter()
        reference_hidden, reference_actions = _forward(model, inputs)
        runtime = time.perf_counter() - started
        _save_reference(
            cache_path,
            hidden=reference_hidden,
            actions=reference_actions,
            groups=groups,
        )
        if reference_key not in existing:
            row = _base_row(
                sample,
                base_state="fp",
                layer=REFERENCE_LAYER,
                operation=REFERENCE_OPERATION,
                runtime_s=runtime,
                signature=signature,
            )
            row["token_counts"] = {
                GROUP_LABELS[group]: int(np.sum(groups == group)) for group in GROUP_NAMES
            }
            row["metrics"] = {"vs_fp": _identity_summary(reference_actions)}
            row["reference_actions"] = reference_actions.reshape(-1, reference_actions.shape[-1]).tolist()
            _append_row(output, row)
            existing[reference_key] = row
            print(f"complete {reference_key} ({runtime:.1f}s)", flush=True)

        if not include_skips:
            continue
        for layer_index in layer_indices:
            key = (sample_id, "fp", layer_index, "skip")
            if key in existing:
                continue
            started = time.perf_counter()
            with temporary_layer_skip(layers[layer_index]):
                hidden, actions = _forward(model, inputs)
            runtime = time.perf_counter() - started
            row = _base_row(
                sample,
                base_state="fp",
                layer=layer_index,
                operation="skip",
                runtime_s=runtime,
                signature=signature,
            )
            row["metrics"] = {
                "vs_fp": _summarize(reference_hidden, reference_actions, hidden, actions, groups)
            }
            _append_row(output, row)
            existing[key] = row
            print(f"complete {key} ({runtime:.1f}s)", flush=True)

    if include_skips:
        missing = [key for key in required_keys if key not in existing]
        if missing:
            raise RuntimeError(f"FP scan ended with {len(missing)} missing rows")


def _w4_scan(
    *,
    model,
    samples: list[dict],
    layers,
    layer_indices: list[int],
    output: Path,
    reference_dir: Path,
    existing: dict,
    signature: dict,
    group_size: int,
) -> None:
    pending = [
        (str(sample["sample_id"]), "w4", layer, "skip")
        for sample in samples
        for layer in layer_indices
        if (str(sample["sample_id"]), "w4", layer, "skip") not in existing
    ]
    pending_references = [
        sample
        for sample in samples
        if (str(sample["sample_id"]), "w4", REFERENCE_LAYER, REFERENCE_OPERATION) not in existing
    ]
    if not pending and not pending_references:
        return

    print("applying global decoder W4 fake quantization", flush=True)
    started = time.perf_counter()
    audit = quantize_module_inplace(layers, bits=4, group_size=group_size)
    quantization_runtime = time.perf_counter() - started
    audit_payload = {
        "bits": 4,
        "group_size": group_size,
        "linears": audit.linears,
        "parameters": audit.parameters,
        "weighted_mse": audit.weighted_mse,
        "runtime_s": quantization_runtime,
        "scope": "all_decoder_linear_weights",
    }
    print(
        f"global W4 ready: {audit.linears} linears, {audit.parameters} parameters "
        f"({quantization_runtime:.1f}s)",
        flush=True,
    )

    for sample in samples:
        sample_id = str(sample["sample_id"])
        reference_key = (sample_id, "w4", REFERENCE_LAYER, REFERENCE_OPERATION)
        missing_skips = [
            layer for layer in layer_indices if (sample_id, "w4", layer, "skip") not in existing
        ]
        if reference_key in existing and not missing_skips:
            continue
        fp_hidden, fp_actions, fp_groups = _load_reference(_reference_path(reference_dir, sample_id))
        inputs = _build_inputs(model, Path(sample["image_path"]), str(sample["instruction"]))
        groups = _token_groups(model, inputs)
        if not np.array_equal(groups, fp_groups):
            raise RuntimeError(f"token grouping changed between FP and W4 for {sample_id}")

        started = time.perf_counter()
        w4_hidden, w4_actions = _forward(model, inputs)
        runtime = time.perf_counter() - started
        if reference_key not in existing:
            row = _base_row(
                sample,
                base_state="w4",
                layer=REFERENCE_LAYER,
                operation=REFERENCE_OPERATION,
                runtime_s=runtime,
                signature=signature,
            )
            row["metrics"] = {
                "vs_fp": _summarize(fp_hidden, fp_actions, w4_hidden, w4_actions, groups)
            }
            row["quantization"] = audit_payload
            _append_row(output, row)
            existing[reference_key] = row
            print(f"complete {reference_key} ({runtime:.1f}s)", flush=True)

        for layer_index in missing_skips:
            key = (sample_id, "w4", layer_index, "skip")
            started = time.perf_counter()
            with temporary_layer_skip(layers[layer_index]):
                hidden, actions = _forward(model, inputs)
            runtime = time.perf_counter() - started
            row = _base_row(
                sample,
                base_state="w4",
                layer=layer_index,
                operation="skip",
                runtime_s=runtime,
                signature=signature,
            )
            row["metrics"] = {
                "vs_base": _summarize(w4_hidden, w4_actions, hidden, actions, groups),
                "vs_fp": _summarize(fp_hidden, fp_actions, hidden, actions, groups),
            }
            _append_row(output, row)
            existing[key] = row
            print(f"complete {key} ({runtime:.1f}s)", flush=True)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("layers must not contain duplicates")
    base_states = [value.strip().lower() for value in args.base_states.split(",") if value.strip()]
    if not base_states or set(base_states) - {"fp", "w4"}:
        raise ValueError("base-states must contain fp and/or w4")

    samples = _read_manifest(args.manifest, args.limit)
    signature = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "config": str(args.config.resolve()),
        "base_vlm": str(args.base_vlm.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "group_size": args.group_size,
    }
    existing = _read_existing(args.output)
    _validate_existing(existing, signature)

    model_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        config=args.config,
        base_vlm=args.base_vlm,
    )
    model = _load_model(model_args)
    layers = _decoder_layers(model)
    if any(layer < 0 or layer >= len(layers) for layer in layer_indices):
        raise ValueError(f"layer indices must be within [0, {len(layers) - 1}]")

    reference_dir = args.output.parent / f"{args.output.stem}_references"
    _fp_scan(
        model=model,
        samples=samples,
        layers=layers,
        layer_indices=layer_indices,
        output=args.output,
        reference_dir=reference_dir,
        existing=existing,
        signature=signature,
        include_skips="fp" in base_states,
    )
    if "w4" in base_states:
        _w4_scan(
            model=model,
            samples=samples,
            layers=layers,
            layer_indices=layer_indices,
            output=args.output,
            reference_dir=reference_dir,
            existing=existing,
            signature=signature,
            group_size=args.group_size,
        )

    print(
        json.dumps(
            {
                "status": "complete",
                "samples": len(samples),
                "layers": layer_indices,
                "base_states": base_states,
                "rows_total": len(existing),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
