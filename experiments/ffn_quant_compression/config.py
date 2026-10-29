from __future__ import annotations

import dataclasses
from pathlib import Path

from experiments.ffn_quant_rootcause.config import Config as RootConfig

MODELS = ("infinitevl", "qwen")
QA_COUNTS = (0, 3, 18, 33, 35, 36)
SEEDS = (42, 43, 44)
POLICIES = ("sensitive_first", "sensitive_protected")
STAGES = ("audit", "sweep", "analyze", "report")
PARAMETERS_PER_LAYER = 3 * 2048 * 11008


@dataclasses.dataclass(frozen=True)
class Config(RootConfig):
    output_dir: Path = Path("outputs/ffn_quant_compression_v1")
    rootcause_output: Path = Path("outputs/ffn_quant_rootcause_v1")
    reserve_hours: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        out, source = self.output_dir.resolve(), self.rootcause_output.resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("output must be separate from frozen root-cause evidence")
        if not 0 <= self.reserve_hours < self.wallclock_hours:
            raise ValueError("reserve must be shorter than the single wallclock budget")

    def payload(self):
        value = super().payload()
        value.update(
            protocol="ffn_quant_compression",
            format_version=1,
            qa_counts=list(QA_COUNTS),
            random_seeds=list(SEEDS),
            compression="1-(16*N_reference+4*N_w4)/(16*N_ffn)",
            reference_weight_bits=16,
            quantized_weight_bits=4,
            budget="one_persistent_deadline_including_all_stages",
            qa_history="historically_reused_evaluation",
            reference_cache_policy="explicit_identity_and_sentinel_verified_import",
            cache_policy="explicit_verified_historical_metric_import",
            gamma=1.0,
            random_generator="numpy.default_rng_PCG64",
            nll_aggregation="sum_target_token_losses_over_sum_target_tokens",
            qa_aggregation="four_dataset_macro_average",
        )
        value.pop("doses", None)
        return value
