from __future__ import annotations

import dataclasses
import hashlib
import json
import math
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
    VG_QIG_MBQ = "vg_qig_mbq"
    VG_UNIFORM_MBQ = "vg_uniform_mbq"
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
    mbq_gradient_aggregation: str = "pooled_elements"
    mbq_objective_normalization: str = "token_weighted"
    mbq_median_floor: bool = True
    mbq_reconstruction_level: str = "linear_group"
    mbq_modality_mode: str = "vision_answer"
    # Optional factorial controls. When unset, both inherit mbq_modality_mode so
    # existing configurations, digests, and search semantics remain unchanged.
    mbq_activation_mode: str | None = None
    mbq_objective_mode: str | None = None
    # Scales only the prompt-language term in a three-class reconstruction
    # objective. The default is serialization-compatible with v5 artifacts.
    mbq_prompt_objective_scale: float = 1.0
    mbq_upstream_commit: str = "a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1"
    mbq_algorithm_version: int = 4
    # Vulnerability-gated QIG-MBQ controls.  They are omitted from serialized
    # non-VG configurations so legacy config digests remain stable.
    vg_algorithm_version: int = 1
    vg_vvs_samples: int = 16
    vg_qig_samples: int = 32
    vg_vvs_threshold: float = 0.05
    vg_vvs_max_layers: int = 4
    vg_gate_bootstrap_samples: int = 2000
    vg_gate_inclusion_probability: float = 0.5
    vg_ig_step_grid: tuple[int, ...] = (8, 16, 32)
    vg_ig_rank_threshold: float = 0.98
    vg_ig_topk_jaccard_threshold: float = 0.90
    vg_iqr_multiplier: float = 1.5
    vg_coordinate_sweeps: int = 2
    # Optional bounded layer set for the real-model CPU smoke test only.
    vg_candidate_layers: tuple[int, ...] = ()
    weight_granularity: str = "groupwise"
    smoothquant_alpha: float = 0.5
    llm_int8_threshold: float = 6.0
    tender_groups: int = 4
    omniquant_steps: int = 20
    squeezellm_kmeans_steps: int = 5
    squeezellm_sparse_fraction: float = 0.0045
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
        if self.weight_bits not in (None, 2, 3, 4, 8):
            raise ValueError("weight_bits must be None, 2, 3, 4, or 8")
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
        if self.mbq_gradient_aggregation not in {
            "pooled_elements", "sample_mean", "source_balanced",
        }:
            raise ValueError("unsupported MBQ gradient aggregation")
        if self.mbq_objective_normalization not in {
            "token_weighted", "modality_mean",
        }:
            raise ValueError("unsupported MBQ objective normalization")
        if self.mbq_reconstruction_level not in {"linear_group", "block_output"}:
            raise ValueError("unsupported MBQ reconstruction level")
        if self.mbq_modality_mode not in {
            "vision_answer", "vision_prompt_answer",
        }:
            raise ValueError("unsupported MBQ modality mode")
        for name, value in (
            ("activation", self.mbq_activation_mode),
            ("objective", self.mbq_objective_mode),
        ):
            if value is not None and value not in {
                "vision_answer", "vision_prompt_answer",
            }:
                raise ValueError(f"unsupported MBQ {name} mode")
        if (
            not math.isfinite(self.mbq_prompt_objective_scale)
            or self.mbq_prompt_objective_scale < 0
        ):
            raise ValueError("MBQ prompt objective scale must be finite and non-negative")
        if (
            self.mbq_prompt_objective_scale != 1.0
            and self.effective_mbq_objective_mode != "vision_prompt_answer"
        ):
            raise ValueError(
                "non-default MBQ prompt objective scale requires a prompt-aware objective"
            )
        if self.mbq_algorithm_version not in {4, 5}:
            raise ValueError("unsupported MBQ algorithm version")
        if (
            self.mbq_modality_mode == "vision_prompt_answer"
            and self.mbq_algorithm_version != 5
        ):
            raise ValueError("three-token-class MBQ requires algorithm version 5")
        if (
            "vision_prompt_answer" in {
                self.effective_mbq_activation_mode,
                self.effective_mbq_objective_mode,
            }
            and self.mbq_modality_mode != "vision_prompt_answer"
        ):
            raise ValueError(
                "prompt-aware MBQ search requires vision_prompt_answer raw statistics"
            )
        if self.weight_granularity not in {"groupwise", "per_output_channel"}:
            raise ValueError("unsupported weight granularity")
        if not 0 <= self.squeezellm_sparse_fraction < 1:
            raise ValueError("squeezellm_sparse_fraction must satisfy 0 <= value < 1")
        vg_methods = {MethodName.VG_QIG_MBQ, MethodName.VG_UNIFORM_MBQ}
        if self.method == MethodName.MBQ or self.method in vg_methods:
            if self.scope != QuantScope.DECODER_CANONICAL:
                raise ValueError("MBQ supports decoder_canonical scope only")
            if self.weight_bits not in (2, 3, 4):
                raise ValueError("MBQ supports W2, W3, or W4 weights in this benchmark")
            if self.weight_bits == 2 and self.activation_bits is not None:
                raise ValueError("W2A16 only")
            if self.weight_bits == 3 and self.activation_bits is not None:
                raise ValueError("W3A16 only")
            if self.mbq_distort:
                raise ValueError("validated MBQ requires distort=False")
            if self.activation_bits is None:
                if not self.zero_point or self.weight_granularity != "groupwise":
                    raise ValueError("MBQ W2A16/W3A16/W4A16 requires asymmetric group-wise weights")
            elif self.activation_bits == 8:
                if self.zero_point or self.weight_granularity != "per_output_channel":
                    raise ValueError(
                        "MBQ W4A8 requires symmetric per-output-channel weights"
                    )
            else:
                raise ValueError("MBQ supports W2A16, W3A16, W4A16, or W4A8 only")
        if self.method in vg_methods:
            if self.activation_bits not in (None, 8):
                raise ValueError("VG-QIG-MBQ supports A16 or A8 activations")
            if self.implementation_tier != ImplementationTier.CPU_RESEARCH:
                raise ValueError(
                    "VG-QIG-MBQ must be labelled cpu_research_approximation"
                )
            if self.vg_algorithm_version != 1:
                raise ValueError("unsupported VG-QIG-MBQ algorithm version")
            if self.vg_vvs_samples <= 0 or self.vg_qig_samples <= 0:
                raise ValueError("VG sample counts must be positive")
            if self.vg_vvs_threshold < 0 or self.vg_vvs_max_layers < 0:
                raise ValueError("invalid VG vulnerability gate")
            if self.vg_gate_bootstrap_samples <= 0:
                raise ValueError("VG bootstrap count must be positive")
            if not 0 <= self.vg_gate_inclusion_probability <= 1:
                raise ValueError("VG inclusion probability must be in [0, 1]")
            if not self.vg_ig_step_grid or any(x <= 0 for x in self.vg_ig_step_grid):
                raise ValueError("VG IG step grid must contain positive values")
            if tuple(sorted(set(self.vg_ig_step_grid))) != self.vg_ig_step_grid:
                raise ValueError("VG IG step grid must be strictly increasing")
            if not 0 <= self.vg_ig_rank_threshold <= 1:
                raise ValueError("VG IG rank threshold must be in [0, 1]")
            if not 0 <= self.vg_ig_topk_jaccard_threshold <= 1:
                raise ValueError("VG IG Jaccard threshold must be in [0, 1]")
            if self.vg_iqr_multiplier < 0 or self.vg_coordinate_sweeps <= 0:
                raise ValueError("invalid VG QIG/search controls")
            if any(layer < 0 for layer in self.vg_candidate_layers):
                raise ValueError("VG candidate layer indices must be non-negative")
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
        # Omitting unset extension fields preserves all pre-factorial digests.
        if value["mbq_activation_mode"] is None:
            value.pop("mbq_activation_mode")
        if value["mbq_objective_mode"] is None:
            value.pop("mbq_objective_mode")
        value["method"] = self.method.value
        value["scope"] = self.scope.value
        if value["mbq_prompt_objective_scale"] == 1.0:
            value.pop("mbq_prompt_objective_scale")
        if self.method not in {MethodName.VG_QIG_MBQ, MethodName.VG_UNIFORM_MBQ}:
            for field in (
                "vg_algorithm_version", "vg_vvs_samples", "vg_qig_samples",
                "vg_vvs_threshold", "vg_vvs_max_layers",
                "vg_gate_bootstrap_samples", "vg_gate_inclusion_probability",
                "vg_ig_step_grid", "vg_ig_rank_threshold",
                "vg_ig_topk_jaccard_threshold", "vg_iqr_multiplier",
                "vg_coordinate_sweeps",
                "vg_candidate_layers",
            ):
                value.pop(field)
        value["implementation_tier"] = ImplementationTier(self.implementation_tier).value
        return value

    @property
    def effective_mbq_activation_mode(self) -> str:
        return self.mbq_activation_mode or self.mbq_modality_mode

    @property
    def effective_mbq_objective_mode(self) -> str:
        return self.mbq_objective_mode or self.mbq_modality_mode

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        fields = {item.name for item in dataclasses.fields(cls)}
        payload = {key: item for key, item in value.items() if key in fields}
        payload["method"] = MethodName(payload["method"])
        payload["scope"] = QuantScope(payload.get("scope", QuantScope.DECODER_CANONICAL))
        for key in (
            "awq_alpha_grid", "awq_clip_grid", "mbq_scale_grid",
            "vg_ig_step_grid", "vg_candidate_layers",
        ):
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
