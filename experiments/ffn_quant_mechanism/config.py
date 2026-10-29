from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.ffn_quant_causal.config import (
    INFINITEVL_MODEL_ID,
    INFINITEVL_REVISION,
    MODEL_SPECS,
)

FORMAT_VERSION = 1
PROTOCOL_NAME = "ffn_quant_mechanism"
STAGES = (
    "audit",
    "calibrate",
    "localize",
    "transport",
    "gdn_mediation",
    "confirm",
    "report",
)
MODEL_KINDS = ("infinitevl", "qwen")
CONDITIONS = ("B", "W", "A", "R", "G", "RG", "M")
MEDIATOR_GROUPS = ("q", "write", "decay", "output_gate")
GDN_SWA_LAYER_PAIRS = tuple((index, index + 1) for index in range(3, 32, 4))
SCREEN_LAYER_PAIRS = ((3, 4), (15, 16), (31, 32))
DOSE_GRID = (0.5, 1.0, 2.0)
TRANSPORT_CLIP = (0.25, 4.0)
STATE_PREFIXES = (64, 128, 256, 512)


@dataclasses.dataclass(frozen=True)
class MechanismConfig:
    output_dir: Path
    parent_output: Path = Path("outputs/ffn_quant_causal_v3")
    qwen_model_path: Path = Path("playground/Pretrained_models/Qwen2.5-VL-3B-Instruct")
    calibration_manifest: Path = Path(
        "data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"
    )
    evaluation_manifest: Path = Path("data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl")
    device: str = "cpu"
    torch_threads: int = 16
    interop_threads: int = 1
    seed: int = 42
    group_size: int = 128
    weight_bits: int = 4
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 32
    wallclock_hours: float = 120.0
    bootstrap_samples: int = 10_000
    transport_clip: tuple[float, float] = TRANSPORT_CLIP
    mediator_min_share: float = 0.15
    mediator_max_confirm: int = 2

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError("the mechanism protocol is CPU-only")
        if self.torch_threads <= 0 or self.interop_threads != 1:
            raise ValueError("use positive Torch threads and one interop thread")
        if self.group_size != 128 or self.weight_bits != 4:
            raise ValueError("the primary mechanism protocol fixes asymmetric W4 g128")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("invalid pixel bounds")
        if self.max_new_tokens <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("generation/bootstrap counts must be positive")
        low, high = self.transport_clip
        if not 0 < low <= 1 <= high:
            raise ValueError("transport clip must contain one and be positive")
        if not 0 < self.mediator_min_share <= 1:
            raise ValueError("mediator_min_share must be in (0, 1]")
        if not 0 <= self.mediator_max_confirm <= len(MEDIATOR_GROUPS):
            raise ValueError("invalid mediator confirmation limit")
        if self.wallclock_hours <= 0:
            raise ValueError("wallclock_hours must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        for key in (
            "output_dir",
            "parent_output",
            "qwen_model_path",
            "calibration_manifest",
            "evaluation_manifest",
        ):
            result[key] = str(result[key])
        result.update(
            {
                "format_version": FORMAT_VERSION,
                "protocol": PROTOCOL_NAME,
                "stages": list(STAGES),
                "models": {key: value.to_dict() for key, value in MODEL_SPECS.items()},
                "infinitevl_model_id": INFINITEVL_MODEL_ID,
                "infinitevl_revision": INFINITEVL_REVISION,
                "conditions": list(CONDITIONS),
                "mediator_groups": list(MEDIATOR_GROUPS),
                "gdn_swa_layer_pairs": [list(value) for value in GDN_SWA_LAYER_PAIRS],
                "screen_layer_pairs": [list(value) for value in SCREEN_LAYER_PAIRS],
                "dose_grid": list(DOSE_GRID),
                "state_prefixes": list(STATE_PREFIXES),
                "quantization": {
                    "scope": "decoder_ffn_gate_up_down_only",
                    "weight_quantizer": "asymmetric_groupwise_fake_quant",
                    "weight_bits": self.weight_bits,
                    "activation_bits": None,
                    "group_size": self.group_size,
                    "compute": "float32_cpu",
                },
                "claims": {
                    "diagnostic_counterfactuals_are_deployment_methods": False,
                    "speed_or_memory_claim": False,
                    "confirm_history": "historically_reused_evaluation",
                },
            }
        )
        return result

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
