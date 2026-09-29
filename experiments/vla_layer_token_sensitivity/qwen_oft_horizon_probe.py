"""Resumable schema-v2 Qwen-OFT horizon-conditioned FP/W4 layer scan."""

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
from experiments.vla_layer_token_sensitivity.qwen_oft_horizon_data import persistence_baseline
from experiments.vla_layer_token_sensitivity.qwen_oft_probe import (
    GROUP_LABELS,
    GROUP_NAMES,
    _build_conditioned_inputs,
    _condition_instruction,
    _decoder_layers,
    _forward,
    _load_model,
    _summarize,
    _token_groups,
)
from starVLA.model.framework.share_tools import state2str_transform


SCHEMA_VERSION = 2
REFERENCE_LAYER = -1
REFERENCE_OPERATION = "reference"
STEP_GROUPS = {
    "early": tuple(range(0, 5)),
    "middle": tuple(range(5, 10)),
    "late": tuple(range(10, 15)),
    "terminal": (15,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="34")
    parser.add_argument("--base-states", default="fp,w4")
    parser.add_argument("--input-condition", default="state,no_state")
    parser.add_argument("--action-step-groups", default="early,middle,late,terminal")
    parser.add_argument("--schema-version", type=int, default=SCHEMA_VERSION)
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


def _parse_conditions(value: str) -> list[str]:
    conditions = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("input-condition must contain unique state/no_state values")
    if set(conditions) - {"state", "no_state"}:
        raise ValueError("input-condition only supports state and no_state")
    return conditions


def _parse_step_groups(value: str) -> dict[str, tuple[int, ...]]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("action-step-groups must contain unique group names")
    unknown = set(names) - set(STEP_GROUPS)
    if unknown:
        raise ValueError(f"unknown action-step groups: {sorted(unknown)}")
    return {name: STEP_GROUPS[name] for name in names}


def _read_manifest(path: Path, limit: int | None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("manifest is empty")
    required = {
        "sample_id",
        "episode_index",
        "frame_index",
        "image_path",
        "instruction",
        "normalized_state",
        "normalized_action_chunk",
    }
    seen: set[str] = set()
    episodes: set[int] = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
        sample_id = str(row["sample_id"])
        episode = int(row["episode_index"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        if episode in episodes:
            raise ValueError("schema-v2 confirmation scan requires one sample per episode")
        seen.add(sample_id)
        episodes.add(episode)
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        state = np.asarray(row["normalized_state"], dtype=np.float32)
        actions = np.asarray(row["normalized_action_chunk"], dtype=np.float32)
        if state.shape != (8,) or actions.shape != (16, 7):
            raise ValueError(f"sample {sample_id} must contain state [8] and actions [16,7]")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"sample {sample_id} contains non-finite normalized values")
    return rows


def _row_key(row: dict) -> tuple[str, str, str, int, str]:
    return (
        str(row["sample_id"]),
        str(row["input_condition"]),
        str(row["base_state"]),
        int(row["layer"]),
        str(row["operation"]),
    )


def _read_existing(path: Path) -> dict[tuple[str, str, str, int, str], dict]:
    if not path.exists():
        return {}
    rows: list[dict] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            repaired = path.with_suffix(path.suffix + ".repair")
            with repaired.open("w", encoding="utf-8") as handle:
                for valid in rows:
                    handle.write(json.dumps(valid, sort_keys=True) + "\n")
            repaired.replace(path)
            break
        if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("schema-v2 output cannot be mixed with older scan rows")
        rows.append(row)
    indexed = {_row_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("existing schema-v2 output contains duplicate result keys")
    return indexed


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reference_path(reference_dir: Path, sample_id: str, condition: str) -> Path:
    return reference_dir / f"{sample_id}__{condition}.npz"


def _save_reference(path: Path, hidden: np.ndarray, actions: np.ndarray, groups: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, hidden=hidden, actions=actions, groups=groups)
    temporary.replace(path)


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return payload["hidden"], payload["actions"], payload["groups"]


def _with_horizons(summary: dict, step_groups: dict[str, tuple[int, ...]]) -> dict:
    step_mse = np.asarray(summary["action_step_mse"], dtype=np.float64)
    if step_mse.shape != (16,):
        raise ValueError("action_step_mse must have 16 entries")
    summary["action_horizon_mse"] = {
        name: float(step_mse[list(indices)].mean()) for name, indices in step_groups.items()
    }
    return summary


def _target_summary(
    actions: np.ndarray,
    target: np.ndarray,
    step_groups: dict[str, tuple[int, ...]],
) -> dict:
    prediction = np.asarray(actions, dtype=np.float64).reshape(-1, actions.shape[-1])
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    squared = np.square(prediction - target)
    step_mse = squared.mean(axis=-1)
    return {
        "action_mse": float(squared.mean()),
        "action_max_abs": float(np.max(np.abs(prediction - target))),
        "action_step_mse": step_mse.tolist(),
        "action_dim_mse": squared.mean(axis=0).tolist(),
        "action_squared_error": squared.tolist(),
        "action_horizon_mse": {
            name: float(step_mse[list(indices)].mean()) for name, indices in step_groups.items()
        },
    }


def _input_state(sample: dict, condition: str) -> np.ndarray | None:
    if condition == "no_state":
        return None
    return np.asarray(sample["normalized_state"], dtype=np.float32)


def _input_audit(model, sample: dict, condition: str) -> dict:
    state = _input_state(sample, condition)
    conditioned = _condition_instruction(model, str(sample["instruction"]), state)
    return {
        "conditioned_instruction": conditioned,
        "state_bins": None if state is None else state2str_transform(state).split(),
    }


def _base_row(
    sample: dict,
    *,
    condition: str,
    base_state: str,
    layer: int,
    operation: str,
    runtime_s: float,
    signature: dict,
    input_audit: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen_oft_horizon_conditioned_fp_w4_scan",
        "sample_id": str(sample["sample_id"]),
        "split": str(sample.get("split", "unknown")),
        "episode_index": int(sample["episode_index"]),
        "task_index": int(sample.get("task_index", -1)),
        "frame_index": int(sample["frame_index"]),
        "instruction": str(sample["instruction"]),
        "input_condition": condition,
        "base_state": base_state,
        "layer": int(layer),
        "operation": operation,
        "runtime_s": float(runtime_s),
        "input_audit": input_audit,
        "signature": signature,
    }


def _build_inputs(model, sample: dict, condition: str):
    return _build_conditioned_inputs(
        model,
        Path(sample["image_path"]),
        str(sample["instruction"]),
        state=_input_state(sample, condition),
    )


def _metrics_vs_reference(
    reference_hidden: np.ndarray,
    reference_actions: np.ndarray,
    hidden: np.ndarray,
    actions: np.ndarray,
    groups: np.ndarray,
    step_groups: dict[str, tuple[int, ...]],
) -> dict:
    return _with_horizons(
        _summarize(reference_hidden, reference_actions, hidden, actions, groups),
        step_groups,
    )


def _validate_existing(existing: dict, signature: dict) -> None:
    for row in existing.values():
        if row.get("signature") != signature:
            raise ValueError("existing output has a different checkpoint/manifest/configuration signature")


def _fp_scan(
    *,
    model,
    samples: list[dict],
    conditions: list[str],
    layers,
    layer_indices: list[int],
    output: Path,
    reference_dir: Path,
    existing: dict,
    signature: dict,
    step_groups: dict[str, tuple[int, ...]],
    include_skips: bool,
) -> None:
    for sample in samples:
        target = np.asarray(sample["normalized_action_chunk"], dtype=np.float32)
        baseline = persistence_baseline(sample["normalized_state"], target.shape[0])
        for condition in conditions:
            sample_id = str(sample["sample_id"])
            reference_key = (sample_id, condition, "fp", REFERENCE_LAYER, REFERENCE_OPERATION)
            cache_path = _reference_path(reference_dir, sample_id, condition)
            inputs = _build_inputs(model, sample, condition)
            groups = _token_groups(model, inputs)
            if cache_path.is_file() and reference_key in existing:
                reference_hidden, reference_actions, cached_groups = _load_reference(cache_path)
                if not np.array_equal(groups, cached_groups):
                    raise RuntimeError(f"cached token groups changed for {sample_id}/{condition}")
            else:
                started = time.perf_counter()
                reference_hidden, reference_actions = _forward(model, inputs)
                runtime = time.perf_counter() - started
                _save_reference(cache_path, reference_hidden, reference_actions, groups)
                if reference_key not in existing:
                    row = _base_row(
                        sample,
                        condition=condition,
                        base_state="fp",
                        layer=REFERENCE_LAYER,
                        operation=REFERENCE_OPERATION,
                        runtime_s=runtime,
                        signature=signature,
                        input_audit=_input_audit(model, sample, condition),
                    )
                    row["token_counts"] = {
                        GROUP_LABELS[group]: int(np.sum(groups == group)) for group in GROUP_NAMES
                    }
                    row["metrics"] = {
                        "vs_fp": _metrics_vs_reference(
                            reference_hidden,
                            reference_actions,
                            reference_hidden,
                            reference_actions,
                            groups,
                            step_groups,
                        ),
                        "vs_ground_truth": _target_summary(reference_actions, target, step_groups),
                    }
                    row["baseline_vs_ground_truth"] = _target_summary(baseline, target, step_groups)
                    row["reference_actions"] = reference_actions.reshape(16, 7).tolist()
                    _append_row(output, row)
                    existing[reference_key] = row
                    print(f"complete {reference_key} ({runtime:.1f}s)", flush=True)

            if not include_skips:
                continue
            for layer_index in layer_indices:
                key = (sample_id, condition, "fp", layer_index, "skip")
                if key in existing:
                    continue
                started = time.perf_counter()
                with temporary_layer_skip(layers[layer_index]):
                    hidden, actions = _forward(model, inputs)
                runtime = time.perf_counter() - started
                row = _base_row(
                    sample,
                    condition=condition,
                    base_state="fp",
                    layer=layer_index,
                    operation="skip",
                    runtime_s=runtime,
                    signature=signature,
                    input_audit=_input_audit(model, sample, condition),
                )
                row["metrics"] = {
                    "vs_fp": _metrics_vs_reference(
                        reference_hidden, reference_actions, hidden, actions, groups, step_groups
                    ),
                    "vs_ground_truth": _target_summary(actions, target, step_groups),
                }
                _append_row(output, row)
                existing[key] = row
                print(f"complete {key} ({runtime:.1f}s)", flush=True)


def _w4_scan(
    *,
    model,
    samples: list[dict],
    conditions: list[str],
    layers,
    layer_indices: list[int],
    output: Path,
    reference_dir: Path,
    existing: dict,
    signature: dict,
    step_groups: dict[str, tuple[int, ...]],
    group_size: int,
) -> None:
    pending = any(
        (str(sample["sample_id"]), condition, "w4", layer, "skip") not in existing
        for sample in samples
        for condition in conditions
        for layer in layer_indices
    ) or any(
        (str(sample["sample_id"]), condition, "w4", REFERENCE_LAYER, REFERENCE_OPERATION)
        not in existing
        for sample in samples
        for condition in conditions
    )
    if not pending:
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
        target = np.asarray(sample["normalized_action_chunk"], dtype=np.float32)
        for condition in conditions:
            sample_id = str(sample["sample_id"])
            fp_hidden, fp_actions, fp_groups = _load_reference(
                _reference_path(reference_dir, sample_id, condition)
            )
            inputs = _build_inputs(model, sample, condition)
            groups = _token_groups(model, inputs)
            if not np.array_equal(groups, fp_groups):
                raise RuntimeError(f"FP/W4 token groups changed for {sample_id}/{condition}")
            reference_key = (sample_id, condition, "w4", REFERENCE_LAYER, REFERENCE_OPERATION)
            missing_skips = [
                layer
                for layer in layer_indices
                if (sample_id, condition, "w4", layer, "skip") not in existing
            ]
            if reference_key in existing and not missing_skips:
                continue
            started = time.perf_counter()
            w4_hidden, w4_actions = _forward(model, inputs)
            runtime = time.perf_counter() - started
            if reference_key not in existing:
                row = _base_row(
                    sample,
                    condition=condition,
                    base_state="w4",
                    layer=REFERENCE_LAYER,
                    operation=REFERENCE_OPERATION,
                    runtime_s=runtime,
                    signature=signature,
                    input_audit=_input_audit(model, sample, condition),
                )
                row["metrics"] = {
                    "vs_fp": _metrics_vs_reference(
                        fp_hidden, fp_actions, w4_hidden, w4_actions, groups, step_groups
                    ),
                    "vs_ground_truth": _target_summary(w4_actions, target, step_groups),
                }
                row["quantization"] = audit_payload
                _append_row(output, row)
                existing[reference_key] = row
                print(f"complete {reference_key} ({runtime:.1f}s)", flush=True)

            for layer_index in missing_skips:
                key = (sample_id, condition, "w4", layer_index, "skip")
                started = time.perf_counter()
                with temporary_layer_skip(layers[layer_index]):
                    hidden, actions = _forward(model, inputs)
                runtime = time.perf_counter() - started
                row = _base_row(
                    sample,
                    condition=condition,
                    base_state="w4",
                    layer=layer_index,
                    operation="skip",
                    runtime_s=runtime,
                    signature=signature,
                    input_audit=_input_audit(model, sample, condition),
                )
                row["metrics"] = {
                    "vs_base": _metrics_vs_reference(
                        w4_hidden, w4_actions, hidden, actions, groups, step_groups
                    ),
                    "vs_fp": _metrics_vs_reference(
                        fp_hidden, fp_actions, hidden, actions, groups, step_groups
                    ),
                    "vs_ground_truth": _target_summary(actions, target, step_groups),
                }
                _append_row(output, row)
                existing[key] = row
                print(f"complete {key} ({runtime:.1f}s)", flush=True)


def main() -> None:
    args = parse_args()
    if args.schema_version != SCHEMA_VERSION:
        raise ValueError(f"only schema-version {SCHEMA_VERSION} is supported")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    if not layer_indices or len(layer_indices) != len(set(layer_indices)):
        raise ValueError("layers must contain unique indices")
    conditions = _parse_conditions(args.input_condition)
    step_groups = _parse_step_groups(args.action_step_groups)
    base_states = [item.strip().lower() for item in args.base_states.split(",") if item.strip()]
    if not base_states or set(base_states) - {"fp", "w4"}:
        raise ValueError("base-states only supports fp,w4")
    samples = _read_manifest(args.manifest, args.limit)
    signature = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "config": str(args.config.resolve()),
        "base_vlm": str(args.base_vlm.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "group_size": args.group_size,
        "action_step_groups": {name: list(indices) for name, indices in step_groups.items()},
    }
    existing = _read_existing(args.output)
    _validate_existing(existing, signature)
    model = _load_model(
        SimpleNamespace(checkpoint=args.checkpoint, config=args.config, base_vlm=args.base_vlm)
    )
    layers = _decoder_layers(model)
    if any(layer < 0 or layer >= len(layers) for layer in layer_indices):
        raise ValueError(f"layer indices must be within [0,{len(layers) - 1}]")
    reference_dir = args.output.parent / f"{args.output.stem}_references"
    _fp_scan(
        model=model,
        samples=samples,
        conditions=conditions,
        layers=layers,
        layer_indices=layer_indices,
        output=args.output,
        reference_dir=reference_dir,
        existing=existing,
        signature=signature,
        step_groups=step_groups,
        include_skips="fp" in base_states,
    )
    if "w4" in base_states:
        _w4_scan(
            model=model,
            samples=samples,
            conditions=conditions,
            layers=layers,
            layer_indices=layer_indices,
            output=args.output,
            reference_dir=reference_dir,
            existing=existing,
            signature=signature,
            step_groups=step_groups,
            group_size=args.group_size,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "schema_version": SCHEMA_VERSION,
                "samples": len(samples),
                "input_conditions": conditions,
                "layers": layer_indices,
                "rows_total": len(existing),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
