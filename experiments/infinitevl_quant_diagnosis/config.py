from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.infinitevl_quant.config import MODEL_ID, MODEL_REVISION


FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class DiagnosticScheme:
    key: str
    weight_bits: int | None
    activation_bits: int | None
    group_size: int = 128
    zero_point: bool = True
    per_output_channel: bool = False

    def __post_init__(self) -> None:
        if self.weight_bits not in (None, 2, 3, 4):
            raise ValueError("weight_bits must be None, 2, 3, or 4")
        if self.activation_bits not in (None, 8):
            raise ValueError("activation_bits must be None or 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.weight_bits is None and self.activation_bits is None:
            raise ValueError("a diagnostic QDQ scheme must quantize weights or activations")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


SCHEMES = {
    "a8_only": DiagnosticScheme("a8_only", None, 8),
    "w4_asym_a16": DiagnosticScheme("w4_asym_a16", 4, None),
    "w4_asym_a8": DiagnosticScheme("w4_asym_a8", 4, 8),
    "w4_sym_channel_a16": DiagnosticScheme(
        "w4_sym_channel_a16", 4, None, zero_point=False, per_output_channel=True,
    ),
    "w4_sym_channel_a8": DiagnosticScheme(
        "w4_sym_channel_a8", 4, 8, zero_point=False, per_output_channel=True,
    ),
    "w3_asym_a16": DiagnosticScheme("w3_asym_a16", 3, None),
    "w2_asym_a16": DiagnosticScheme("w2_asym_a16", 2, None),
}


@dataclasses.dataclass(frozen=True)
class DiagnosisConfig:
    output_dir: Path
    source_output: Path
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

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        for key in (
            "output_dir", "source_output", "calibration_manifest", "evaluation_manifest",
        ):
            result[key] = str(result[key])
        result.update({
            "format_version": FORMAT_VERSION,
            "labels": [
                "cpu_reference_backend", "fake_quantized", "no_speed_or_memory_claim",
                "causal_diagnosis",
            ],
        })
        return result

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

