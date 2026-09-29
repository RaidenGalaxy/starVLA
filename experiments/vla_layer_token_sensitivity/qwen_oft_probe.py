"""CPU teacher-forced probe for the legacy StarVLA QwenOFT checkpoint.

This script measures teacher action divergence and final-state drift after
single-layer skip or W8/W4/W3 fake-quant perturbations.  It is intentionally an
open-loop diagnostic: it does not claim packed-kernel speed or robot success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from experiments.vla_layer_token_sensitivity.metrics import (
    cosine_drift_by_group,
    linear_cka_by_group,
)
from experiments.vla_layer_token_sensitivity.perturbations import (
    temporary_groupwise_weight_quantization,
    temporary_layer_skip,
)
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.vlm.QWen2_5 import IMAGE_TOKEN_INDEX


GROUP_VISION = 0
GROUP_LANGUAGE = 1
GROUP_ACTION = 2
GROUP_NAMES = (GROUP_VISION, GROUP_LANGUAGE, GROUP_ACTION)
GROUP_LABELS = {
    GROUP_VISION: "vision",
    GROUP_LANGUAGE: "language",
    GROUP_ACTION: "action_query",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers",
        default="0,6,12,18,24,30,34",
        help="Comma-separated zero-based decoder layer indices.",
    )
    parser.add_argument("--ops", default="skip,w4", help="Comma-separated skip/w8/w4/w3 operations.")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def _load_model(args: argparse.Namespace) -> Qwenvl_OFT:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.base_vlm.is_dir():
        raise FileNotFoundError(args.base_vlm)

    config = OmegaConf.load(args.config)
    apply_config_compat(config)
    config.framework.qwenvl.base_vlm = str(args.base_vlm.resolve())
    config.framework.qwenvl.attn_implementation = "sdpa"
    model = Qwenvl_OFT(config).float().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def _decoder_layers(model: Qwenvl_OFT):
    return model.qwen_vl_interface.model.model.language_model.layers


def _build_inputs(model: Qwenvl_OFT, image_path: Path, instruction: str):
    return _build_conditioned_inputs(model, image_path, instruction, state=None)


def _condition_instruction(
    model: Qwenvl_OFT,
    instruction: str,
    state: np.ndarray | list[float] | None,
) -> str:
    """Apply Qwen-OFT's native discretised-state text conditioning."""

    if state is None:
        return instruction
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (8,):
        raise ValueError(f"normalized state must have shape [8], got {state_array.shape}")
    return model.add_discretized_state_to_instruction(
        [instruction],
        [state_array.reshape(1, -1)],
    )[0]


def _prompted_instruction(model: Qwenvl_OFT, instruction: str) -> str:
    action_tokens = model.action_token * model.chunk_len
    suffix = f" Please predict the next {model.chunk_len} robot actions: <action>{action_tokens}<action>."
    return instruction + suffix


def _build_conditioned_inputs(
    model: Qwenvl_OFT,
    image_path: Path,
    instruction: str,
    *,
    state: np.ndarray | list[float] | None,
):
    image = Image.open(image_path).convert("RGB")
    image_size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    if image_size is None:
        image_size = getattr(model.config.datasets.vla_data, "image_size", None)
    if image_size is not None:
        image = image.resize(tuple(int(value) for value in image_size))
    conditioned = _condition_instruction(model, instruction, state)
    return model.qwen_vl_interface.build_qwenvl_inputs(
        images=[[image]],
        instructions=[_prompted_instruction(model, conditioned)],
    )


def _token_groups(model: Qwenvl_OFT, inputs) -> np.ndarray:
    input_ids = inputs["input_ids"]
    attention = inputs["attention_mask"].bool()
    groups = torch.full_like(input_ids, GROUP_LANGUAGE)
    groups[input_ids == IMAGE_TOKEN_INDEX] = GROUP_VISION
    groups[input_ids == model.action_token_id] = GROUP_ACTION
    groups[~attention] = -1
    present = set(groups[groups >= 0].unique().tolist())
    missing = set(GROUP_NAMES) - present
    if missing:
        raise RuntimeError(f"missing required token groups: {sorted(missing)}")
    return groups.cpu().numpy()


@torch.inference_mode()
def _forward(model: Qwenvl_OFT, inputs) -> tuple[np.ndarray, np.ndarray]:
    outputs = model.qwen_vl_interface.model(
        **inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    action_queries = model._gather_action_token_embeddings(
        hidden,
        inputs["input_ids"],
        action_token_id=model.action_token_id,
    )
    actions = model.action_model.predict_action(action_queries)
    if not torch.isfinite(hidden).all():
        raise RuntimeError("VLM produced non-finite hidden states")
    if not torch.isfinite(actions).all():
        raise RuntimeError("action head produced non-finite actions")
    return hidden.float().cpu().numpy(), actions.float().cpu().numpy()


def _renamed(metrics: dict[int, float]) -> dict[str, float]:
    return {GROUP_LABELS[int(group)]: float(value) for group, value in metrics.items()}


def _summarize(
    reference_hidden: np.ndarray,
    reference_actions: np.ndarray,
    candidate_hidden: np.ndarray,
    candidate_actions: np.ndarray,
    groups: np.ndarray,
) -> dict:
    action_delta = candidate_actions - reference_actions
    action_squared_error = np.square(action_delta).reshape(-1, action_delta.shape[-1])
    action_mask = groups == GROUP_ACTION
    if action_mask.shape[0] != 1:
        raise ValueError("per-position action-query drift currently expects batch size 1")
    action_reference = reference_hidden[action_mask]
    action_candidate = candidate_hidden[action_mask]
    if action_reference.shape[0] != reference_actions.shape[-2]:
        raise ValueError("action-query token count does not match action horizon")
    dot = np.sum(action_reference * action_candidate, axis=-1)
    norm = np.linalg.norm(action_reference, axis=-1) * np.linalg.norm(action_candidate, axis=-1)
    action_query_drift = 1.0 - np.clip(dot / np.maximum(norm, 1e-12), -1.0, 1.0)
    return {
        "action_mse": float(np.mean(np.square(action_delta))),
        "action_max_abs": float(np.max(np.abs(action_delta))),
        "action_step_mse": np.mean(action_squared_error, axis=-1).tolist(),
        "action_dim_mse": np.mean(action_squared_error, axis=0).tolist(),
        "action_squared_error": action_squared_error.tolist(),
        "action_query_cosine_drift": action_query_drift.tolist(),
        "final_hidden_cosine_drift": _renamed(
            cosine_drift_by_group(
                reference_hidden,
                candidate_hidden,
                groups,
                group_names=GROUP_NAMES,
            )
        ),
        "final_hidden_cka": _renamed(
            linear_cka_by_group(
                reference_hidden,
                candidate_hidden,
                groups,
                group_names=GROUP_NAMES,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    layers_to_probe = [int(value) for value in args.layers.split(",") if value.strip()]
    operations = [value.strip().lower() for value in args.ops.split(",") if value.strip()]
    invalid_ops = set(operations) - {"skip", "w8", "w4", "w3"}
    if invalid_ops:
        raise ValueError(f"unsupported operations: {sorted(invalid_ops)}")

    model = _load_model(args)
    inputs = _build_inputs(model, args.image, args.instruction)
    groups = _token_groups(model, inputs)
    layers = _decoder_layers(model)
    if any(layer < 0 or layer >= len(layers) for layer in layers_to_probe):
        raise ValueError(f"layer indices must be within [0, {len(layers) - 1}]")

    reference_hidden, reference_actions = _forward(model, inputs)
    rows = []
    for layer_index in layers_to_probe:
        layer = layers[layer_index]
        for operation in operations:
            audit = None
            if operation == "skip":
                context = temporary_layer_skip(layer)
            else:
                bits = int(operation[1:])
                context = temporary_groupwise_weight_quantization(
                    layer,
                    bits=bits,
                    group_size=args.group_size,
                )
            with context as audit:
                hidden, actions = _forward(model, inputs)
            row = {
                "layer": layer_index,
                "operation": operation,
                **_summarize(reference_hidden, reference_actions, hidden, actions, groups),
            }
            if audit is not None:
                row["quantization"] = {
                    "linears": audit.linears,
                    "parameters": audit.parameters,
                    "weighted_mse": audit.weighted_mse,
                    "group_size": args.group_size,
                }
            rows.append(row)

    payload = {
        "status": "complete",
        "kind": "cpu_open_loop_teacher_divergence",
        "checkpoint": str(args.checkpoint.resolve()),
        "base_vlm": str(args.base_vlm.resolve()),
        "image": str(args.image.resolve()),
        "instruction": args.instruction,
        "token_counts": {
            GROUP_LABELS[group]: int(np.sum(groups == group)) for group in GROUP_NAMES
        },
        "action_shape": list(reference_actions.shape),
        "layers_total": len(layers),
        "rows": rows,
        "limitations": [
            "single-sample open-loop diagnostic",
            "fake-quantized weights use dequantized CPU compute",
            "teacher action divergence is not robot task success",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
