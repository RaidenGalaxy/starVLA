from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
EXPECTED_HIDDEN_LAYERS = 36
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_FFN_LINEAR_COUNT = EXPECTED_HIDDEN_LAYERS * len(FFN_PROJECTIONS)

INFINITEVL_MODEL_ID = "hustvl/InfiniteVL"
INFINITEVL_REVISION = "3d705618aabb1ac60bae75fab6ad0e4679aff58d"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

STAGES = (
    "audit",
    "rtn",
    "trace",
    "counterfactual",
    "mbq_qig",
    "confirm",
    "report",
)

RESULT_LABELS = (
    "cpu_reference_backend",
    "fake_quantized",
    "ffn_only",
    "matched_cross_model_protocol",
    "within_protocol_holdout192",
    "historically_reused_evaluation",
    "no_speed_or_memory_claim",
)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Immutable logical identity expected from one side of the comparison."""

    key: str
    model_id: str
    revision: str
    model_type: str
    hidden_layers: int = EXPECTED_HIDDEN_LAYERS
    hidden_size: int = 2048
    intermediate_size: int = 11008

    def __post_init__(self) -> None:
        if self.key not in {"infinitevl", "qwen"}:
            raise ValueError(f"unsupported comparison model key: {self.key}")
        if not self.model_id or not self.revision or not self.model_type:
            raise ValueError("model identity fields must be non-empty")
        if self.hidden_layers != EXPECTED_HIDDEN_LAYERS:
            raise ValueError(
                f"matched protocol requires {EXPECTED_HIDDEN_LAYERS} decoder layers"
            )
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("model dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


MODEL_SPECS = {
    "infinitevl": ModelSpec(
        key="infinitevl",
        model_id=INFINITEVL_MODEL_ID,
        revision=INFINITEVL_REVISION,
        model_type="infinite_vl",
    ),
    "qwen": ModelSpec(
        key="qwen",
        model_id=QWEN_MODEL_ID,
        # The local Qwen checkpoint is content-addressed by audit.py.  This
        # label deliberately avoids claiming an unverified upstream revision.
        revision="local_content_addressed_checkpoint",
        model_type="qwen2_5_vl",
    ),
}


@dataclasses.dataclass(frozen=True)
class CausalConfig:
    output_dir: Path
    qwen_model_path: Path = Path(
        "playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    )
    calibration_manifest: Path = Path(
        "data/vlm_quant_benchmark/multimodal_calibration/"
        "multimodal_calibration_4x32.jsonl"
    )
    evaluation_manifest: Path = Path(
        "data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl"
    )
    device: str = "cpu"
    torch_threads: int = 16
    interop_threads: int = 1
    seed: int = 42
    group_size: int = 128
    max_new_tokens: int = 32
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    wallclock_hours: float = 120.0
    qig_configuration_hours: float = 18.0
    bootstrap_samples: int = 10_000
    weight_bits: tuple[int, ...] = (8, 4, 3, 2)
    search_weight_bits: int = 4
    mbq_scale_grid: tuple[float, ...] = tuple(index / 20 for index in range(20))
    qig_steps: tuple[int, ...] = (8, 16, 32)
    qig_coordinate_sweeps: int = 2
    qig_spearman: float = 0.98
    qig_jaccard: float = 0.90
    vvs_threshold: float = 0.05
    vvs_max_layers: int = 4
    vvs_bootstrap: int = 2_000

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError("the causal comparison protocol is CPU-only")
        if self.torch_threads <= 0 or self.interop_threads != 1:
            raise ValueError("use positive Torch threads and exactly one interop thread")
        if self.group_size != 128:
            raise ValueError("the matched protocol fixes group_size=128")
        if self.weight_bits != (8, 4, 3, 2):
            raise ValueError("the matched RTN grid is fixed to W8/W4/W3/W2")
        if self.search_weight_bits != 4:
            raise ValueError("the causal recovery comparison preregisters W4A16")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("invalid pixel bounds")
        if self.wallclock_hours <= 0 or self.qig_configuration_hours <= 0:
            raise ValueError("wall-clock limits must be positive")
        if not self.mbq_scale_grid or any(
            value < 0 or value >= 1 for value in self.mbq_scale_grid
        ):
            raise ValueError("MBQ scale grid values must satisfy 0 <= value < 1")
        if tuple(sorted(set(self.qig_steps))) != self.qig_steps or any(
            step <= 0 for step in self.qig_steps
        ):
            raise ValueError("QIG steps must be unique, positive, and increasing")
        if self.qig_coordinate_sweeps <= 0 or self.vvs_max_layers < 0:
            raise ValueError("invalid QIG/VVS search limits")
        if not 0 <= self.qig_spearman <= 1 or not 0 <= self.qig_jaccard <= 1:
            raise ValueError("QIG convergence thresholds must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        for key in (
            "output_dir", "qwen_model_path", "calibration_manifest",
            "evaluation_manifest",
        ):
            result[key] = str(result[key])
        result.update({
            "format_version": FORMAT_VERSION,
            "models": {key: value.to_dict() for key, value in MODEL_SPECS.items()},
            "stages": list(STAGES),
            "result_labels": list(RESULT_LABELS),
            "quantization": {
                "scope": "decoder_ffn_gate_up_down_only",
                "expected_linears_per_model": EXPECTED_FFN_LINEAR_COUNT,
                "weight_quantizer": "asymmetric_groupwise_fake_quant",
                "activation_quantizer": None,
                "compute_dtype": "float32_cpu",
                "reference_values": "bf16_checkpoint_values_widened_to_float32",
            },
            "split_contract": {
                "mbq_calibration": 128,
                "vvs_subset": 16,
                "qig_subset": 32,
                "qig_acceptance": 32,
                "screen": 32,
                "diagnostic8": 8,
                "confirm": 192,
                "confirm_report_label": "within_protocol_holdout192",
                "confirm_evaluation_history": "historically_reused_evaluation",
                "full_descriptive": 256,
            },
        })
        return result

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
