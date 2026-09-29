from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "hustvl/InfiniteVL"
MODEL_REVISION = "3d705618aabb1ac60bae75fab6ad0e4679aff58d"
FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class QuantScheme:
    name: str
    weight_bits: int | None
    activation_bits: int | None = None
    group_size: int = 128
    zero_point: bool = True
    per_output_channel: bool = False

    def __post_init__(self) -> None:
        if self.weight_bits not in (None, 2, 3, 4, 8):
            raise ValueError("weight_bits must be None, 2, 3, 4, or 8")
        if self.activation_bits not in (None, 8):
            raise ValueError("only A16 (None) and A8 are supported")
        if self.weight_bits is None and self.activation_bits is not None:
            raise ValueError("activation quantization requires weight quantization")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.activation_bits == 8 and (
            self.weight_bits != 4 or self.zero_point or not self.per_output_channel
        ):
            raise ValueError("W4A8 is symmetric per-output-channel W4 plus per-token A8")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


SCHEMES = {
    "bf16": QuantScheme("bf16_value_fp32_compute", None),
    "fp16": QuantScheme("fp16_rounded_value_fp32_compute", None),
    "w8a16": QuantScheme("rtn_w8a16", 8),
    "w4a16": QuantScheme("rtn_w4a16", 4),
    "w4a8": QuantScheme(
        "rtn_w4a8", 4, activation_bits=8, zero_point=False,
        per_output_channel=True,
    ),
    "w3a16": QuantScheme("rtn_w3a16", 3),
    "w2a16": QuantScheme("rtn_w2a16", 2),
}


@dataclasses.dataclass(frozen=True)
class ProtocolConfig:
    output_dir: Path
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    calibration_manifest: Path = Path(
        "data/vlm_quant_benchmark/multimodal_calibration/"
        "multimodal_calibration_4x32.jsonl"
    )
    evaluation_manifest: Path = Path(
        "data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl"
    )
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 16
    seed: int = 42
    wallclock_hours: float = 120.0
    qig_budget_hours: float = 18.0
    group_size: int = 128
    vvs_threshold: float = 0.05
    vvs_max_layers: int = 2
    vvs_bootstrap: int = 500
    qig_steps: tuple[int, ...] = (4, 8, 16)
    qig_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    qig_spearman: float = 0.98
    qig_jaccard: float = 0.90
    bootstrap_samples: int = 2000

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        for key in ("output_dir", "calibration_manifest", "evaluation_manifest"):
            result[key] = str(result[key])
        result["format_version"] = FORMAT_VERSION
        result["result_labels"] = [
            "cpu_reference_backend", "fake_quantized", "no_speed_or_memory_claim",
        ]
        return result

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


EXPECTED_COUNTS = {
    "decoder_ffn": 108,
    "swa_attention": 36,
    "gdn_main": 135,
    "gdn_dynamics": 54,
    "decoder_total": 333,
    "decoder_safe": 279,
}
