from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any


class MethodName(StrEnum):
    BF16_VALUE_FP32_COMPUTE = "bf16_value_fp32_compute"
    # Compatibility alias. New configs serialize the unambiguous value above.
    BF16_REFERENCE = "bf16_value_fp32_compute"
    RTN = "rtn"
    AWQ = "awq"
    MBQ = "mbq"
    GPTQ = "gptq"
    SMOOTHQUANT = "smoothquant"
    LLM_INT8 = "llm_int8"
    VECTORWISE_W8 = "vectorwise_w8"
    TENDER = "tender"
    OMNIQUANT = "omniquant"
    SPQR = "spqr"
    SQUEEZELLM = "squeezellm"

    @classmethod
    def _missing_(cls, value):
        if value == "bf16_reference":
            return cls.BF16_VALUE_FP32_COMPUTE
        return None


class QuantScope(StrEnum):
    DECODER_CANONICAL = "decoder_canonical"
    FULL_VLM_EXTENDED = "full_vlm_extended"


class ImplementationTier(StrEnum):
    REFERENCE = "reference"
    NAIVE_BASELINE = "naive_baseline"
    OFFICIAL_ANCHOR = "official_anchor"
    LOCAL_VALIDATED = "local_algorithm_simulation"
    CPU_RESEARCH = "cpu_research_approximation"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    method: MethodName
    scope: QuantScope = QuantScope.DECODER_CANONICAL
    weight_bits: int | None = 4
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
    mbq_scale_grid: tuple[float, ...] = tuple(i / 20 for i in range(20))
    mbq_loss_mode: str = "mae"
    mbq_reweight: bool = True
    mbq_distort: bool = False
    mbq_upstream_commit: str = "a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1"
    mbq_algorithm_version: int = 2
    weight_granularity: str = "groupwise"
    smoothquant_alpha: float = 0.5
    llm_int8_threshold: float = 6.0
    tender_groups: int = 4
    omniquant_steps: int = 20
    squeezellm_kmeans_steps: int = 5
    scheme_id: str = ""
    implementation_tier: ImplementationTier | str = ImplementationTier.LOCAL_VALIDATED
    calibration_protocol: str = "c4_128x2048"
    model_revision: str = "local"
    processor_revision: str = "local"
    calibration_manifest_sha256: str = ""
    evaluation_manifest_sha256: str = ""
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    judge_nproc: int = 4
    judge_retries: int = 3
    format_version: int = 2

    def __post_init__(self) -> None:
        if self.weight_bits not in (None, 3, 4, 8):
            raise ValueError("weight_bits must be None, 3, 4, or 8")
        if self.activation_bits not in (None, 8):
            raise ValueError("active configurations support A16 or A8 only")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.calibration_samples <= 0 or self.calibration_tokens <= 0:
            raise ValueError("calibration limits must be positive")
        if self.omniquant_steps <= 0 or self.squeezellm_kmeans_steps <= 0:
            raise ValueError("second-wave optimization steps must be positive")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("pixel bounds must satisfy 0 < min_pixels <= max_pixels")
        if self.judge_nproc <= 0 or self.judge_retries < 0:
            raise ValueError("invalid judge concurrency/retry settings")
        if not self.mbq_scale_grid or any(
            value < 0 or value >= 1 for value in self.mbq_scale_grid
        ):
            raise ValueError("mbq_scale_grid values must satisfy 0 <= value < 1")
        if self.mbq_loss_mode != "mae":
            raise ValueError("the validated MBQ path supports MAE reconstruction only")
        if self.mbq_algorithm_version != 2:
            raise ValueError("unsupported MBQ algorithm version")
        if self.weight_granularity not in {"groupwise", "per_output_channel"}:
            raise ValueError("unsupported weight granularity")
        if self.method == MethodName.MBQ:
            if self.scope != QuantScope.DECODER_CANONICAL:
                raise ValueError("MBQ supports decoder_canonical scope only")
            if self.weight_bits not in (3, 4):
                raise ValueError("MBQ supports W3 or W4 weights in this benchmark")
            if self.weight_bits == 3 and self.activation_bits is not None:
                raise ValueError("W3 is evaluated as W3A16 only")
            if not self.mbq_reweight or self.mbq_distort:
                raise ValueError("validated MBQ requires reweight=True and distort=False")
            if self.activation_bits is None:
                if not self.zero_point or self.weight_granularity != "groupwise":
                    raise ValueError("MBQ W3A16/W4A16 requires asymmetric group-wise weights")
            elif self.activation_bits == 8:
                if self.zero_point or self.weight_granularity != "per_output_channel":
                    raise ValueError(
                        "MBQ W4A8 requires symmetric per-output-channel weights"
                    )
            else:
                raise ValueError("MBQ supports W3A16, W4A16, or W4A8 only")
        tier = ImplementationTier(self.implementation_tier)
        object.__setattr__(self, "implementation_tier", tier)
        if not self.scheme_id:
            object.__setattr__(self, "scheme_id", self._default_scheme_id())

    def _default_scheme_id(self) -> str:
        if self.method == MethodName.BF16_VALUE_FP32_COMPUTE:
            return "bf16_value_fp32_compute"
        activations = self.activation_bits or 16
        quantizer = "asym" if self.zero_point else "sym"
        return f"w{self.weight_bits}a{activations}_g{self.group_size}_{quantizer}"

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["method"] = self.method.value
        value["scope"] = self.scope.value
        value["implementation_tier"] = ImplementationTier(self.implementation_tier).value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        fields = {item.name for item in dataclasses.fields(cls)}
        payload = {key: item for key, item in value.items() if key in fields}
        payload["method"] = MethodName(payload["method"])
        payload["scope"] = QuantScope(payload.get("scope", QuantScope.DECODER_CANONICAL))
        for key in ("awq_alpha_grid", "awq_clip_grid", "mbq_scale_grid"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

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
    configs = [ExperimentConfig(
        MethodName.BF16_VALUE_FP32_COMPUTE,
        weight_bits=None,
        zero_point=False,
        implementation_tier=ImplementationTier.REFERENCE,
    )]
    for scope in QuantScope:
        configs.extend(
            [
                ExperimentConfig(
                    MethodName.RTN, scope=scope,
                    implementation_tier=ImplementationTier.NAIVE_BASELINE,
                ),
                ExperimentConfig(MethodName.AWQ, scope=scope),
                ExperimentConfig(MethodName.GPTQ, scope=scope),
                ExperimentConfig(
                    MethodName.RTN, scope=scope, weight_bits=8, activation_bits=8,
                    zero_point=False,
                    implementation_tier=ImplementationTier.NAIVE_BASELINE,
                ),
                ExperimentConfig(MethodName.SMOOTHQUANT, scope=scope, weight_bits=8, activation_bits=8),
                ExperimentConfig(MethodName.LLM_INT8, scope=scope, weight_bits=8, activation_bits=8),
                ExperimentConfig(MethodName.TENDER, scope=scope, weight_bits=8, activation_bits=8),
            ]
        )
    return configs
