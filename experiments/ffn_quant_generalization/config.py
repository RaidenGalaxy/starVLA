from __future__ import annotations

import dataclasses
from pathlib import Path

from experiments.infinitevl_quant.progress import json_digest

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "infinitevl": (36, 2048, 11008),
    "qwen25vl_3b": (36, 2048, 11008),
    "qwen25vl_7b": (28, 3584, 18944),
}
POLICIES = ("sensitive_first", "sensitive_protected")
SEEDS = (42, 43, 44)
DATASETS = ("ChartQA_TEST", "GQA_TESTDEV_BALANCED")
SOURCES = {
    "model7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "chartqa": ("HuggingFaceM4/ChartQA", "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"),
    "gqa": ("lmms-lab-encoder/GQA", "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"),
}
STAGES = ("audit", "rank", "sweep", "confirm", "analyze", "report")


@dataclasses.dataclass(frozen=True)
class Config:
    output_dir: Path = ROOT / "outputs/ffn_quant_generalization_v1"
    history_dir: Path = ROOT / "outputs/ffn_quant_compression_v1"
    calibration_manifest: Path = (
        ROOT / "data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"
    )
    evaluation_manifest: Path = ROOT / "data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl"
    threads: int = 16
    total_hours: float = 240.0
    round_hours: float = 120.0
    reserve_hours: float = 12.0
    bootstrap_samples: int = 10_000
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 32
    group_size: int = 128
    seed: int = 20261026

    def __post_init__(self):
        if self.threads != 16 or self.total_hours != 240 or self.round_hours != 120:
            raise ValueError("the frozen protocol requires 16 threads, 240 total and 120 hours per round")
        if not 0 < self.reserve_hours < self.total_hours or self.group_size != 128:
            raise ValueError("invalid reserve or quantizer")
        if self.bootstrap_samples != 10_000:
            raise ValueError("the registered bootstrap/randomization count is 10000")
        out, old = self.output_dir.resolve(), self.history_dir.resolve()
        if out == old or out in old.parents or old in out.parents:
            raise ValueError("new output must be separate from historical evidence")

    def payload(self):
        return {
            **{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in dataclasses.asdict(self).items()},
            "format_version": 1,
            "protocol": "ffn_quant_generalization",
            "models": MODELS,
            "sources": SOURCES,
            "compute": "bf16_value_float32_cpu_fake_quant",
            "quantizer": "W4_asymmetric_RTN_g128_gamma1_FFN_only",
            "coverage": "N_W4/N_FFN; separate BF16 bit savings = 0.75*coverage",
        }

    def digest(self):
        return json_digest(self.payload())
