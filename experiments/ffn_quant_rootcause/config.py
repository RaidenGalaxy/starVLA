from __future__ import annotations

import dataclasses
from pathlib import Path

from experiments.infinitevl_quant.progress import json_digest

STAGES = ("audit", "ratio", "local", "dose", "propagation", "confirm", "report")
MODELS = ("infinitevl", "qwen")
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEEDS = (42, 43, 44)
PROJECTIONS = ("gate", "up", "down")


@dataclasses.dataclass(frozen=True)
class Config:
    output_dir: Path = Path("outputs/ffn_quant_rootcause_v1")
    parent_output: Path = Path("outputs/ffn_quant_causal_v3")
    mechanism_output: Path = Path("outputs/ffn_quant_mechanism_v1")
    infinitevl_model_path: Path | None = None
    qwen_model_path: Path = Path("playground/Pretrained_models/Qwen2.5-VL-3B-Instruct")
    calibration_manifest: Path = Path(
        "data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"
    )
    evaluation_manifest: Path = Path("data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl")
    threads: int = 16
    wallclock_hours: float = 120.0
    bootstrap_samples: int = 10_000
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 32
    group_size: int = 128
    chunk_tokens: int = 128
    linear_epsilon: float = 0.01

    def __post_init__(self):
        output = self.output_dir.resolve()
        protected = [self.parent_output.resolve(), self.mechanism_output.resolve(), self.qwen_model_path.resolve()]
        if self.infinitevl_model_path is not None:
            protected.append(self.infinitevl_model_path.resolve())
        if any(output == p or output in p.parents or p in output.parents for p in protected):
            raise ValueError("output must be separate from historical evidence and checkpoint directories")
        if self.threads <= 0 or not 0 < self.wallclock_hours <= 120:
            raise ValueError("positive threads and at most 120 hours per round are required")
        if self.bootstrap_samples <= 0 or self.group_size != 128 or self.chunk_tokens <= 0:
            raise ValueError("invalid bootstrap/chunk count or non-g128 protocol")
        if not 0 < self.linear_epsilon < 1 or self.max_new_tokens <= 0:
            raise ValueError("invalid finite-difference epsilon or generation limit")

    def payload(self):
        result = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in dataclasses.asdict(self).items()}
        result.update(
            protocol="ffn_quant_rootcause",
            format_version=1,
            compute="float32_cpu",
            activation_quantization=False,
            quantizer="asymmetric_groupwise_fake_quant",
            seeds=list(SEEDS),
            doses=list(DOSES),
            layer_indexing="zero_based",
            confirmation="historically_reused_evaluation",
            cache_policy="new_baselines_no_historical_metric_import",
        )
        return result

    def digest(self):
        return json_digest(self.payload())
