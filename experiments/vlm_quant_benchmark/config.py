from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any


class MethodName(StrEnum):
    BF16_REFERENCE = "bf16_reference"
    RTN = "rtn"
    AWQ = "awq"
    GPTQ = "gptq"
    SMOOTHQUANT = "smoothquant"
    LLM_INT8 = "llm_int8"
    TENDER = "tender"
    OMNIQUANT = "omniquant"
    SPQR = "spqr"
    SQUEEZELLM = "squeezellm"


class QuantScope(StrEnum):
    DECODER_CANONICAL = "decoder_canonical"
    FULL_VLM_EXTENDED = "full_vlm_extended"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    method: MethodName
    scope: QuantScope = QuantScope.DECODER_CANONICAL
    weight_bits: int = 4
    activation_bits: int | None = None
    group_size: int = 128
    zero_point: bool = True
    calibration: str = "text"
    calibration_samples: int = 128
    calibration_tokens: int = 2048
    seed: int = 42
    max_new_tokens: int = 128
    awq_alpha_grid: tuple[float, ...] = tuple(i / 20 for i in range(21))
    awq_clip_grid: tuple[float, ...] = (1.0, 0.95, 0.90, 0.85)
    smoothquant_alpha: float = 0.5
    llm_int8_threshold: float = 6.0
    tender_groups: int = 4
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.weight_bits not in (4, 8):
            raise ValueError("weight_bits must be 4 or 8")
        if self.activation_bits not in (None, 4, 8):
            raise ValueError("activation_bits must be None, 4, or 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.calibration_samples <= 0 or self.calibration_tokens <= 0:
            raise ValueError("calibration limits must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["method"] = self.method.value
        value["scope"] = self.scope.value
        return value

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def default_matrix() -> list[ExperimentConfig]:
    """Return method-matched configurations; unlike schemes are not conflated."""
    configs = [ExperimentConfig(MethodName.BF16_REFERENCE)]
    for scope in QuantScope:
        configs.extend(
            [
                ExperimentConfig(MethodName.RTN, scope=scope),
                ExperimentConfig(MethodName.AWQ, scope=scope),
                ExperimentConfig(MethodName.GPTQ, scope=scope),
                ExperimentConfig(MethodName.RTN, scope=scope, weight_bits=8, activation_bits=8),
                ExperimentConfig(MethodName.SMOOTHQUANT, scope=scope, weight_bits=8, activation_bits=8),
                ExperimentConfig(MethodName.LLM_INT8, scope=scope, weight_bits=8, activation_bits=8),
                ExperimentConfig(MethodName.TENDER, scope=scope, weight_bits=8, activation_bits=8),
                ExperimentConfig(MethodName.TENDER, scope=scope, weight_bits=4, activation_bits=4),
            ]
        )
    return configs

