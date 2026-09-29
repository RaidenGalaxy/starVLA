from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.infinitevl_quant.config import MODEL_ID, MODEL_REVISION


FORMAT_VERSION = 1
FORMATS = ("w8a16", "w4a16", "w3a16")
SCREEN_SCOPES = ("swa", "gdn_main", "attn_main", "gdn_dynamics", "attn_all")
CONFIRM_SCOPES = ("attn_main", "attn_all")
EXPECTED_SCOPE_COUNTS = {
    "swa": 36,
    "gdn_main": 135,
    "attn_main": 171,
    "gdn_dynamics": 54,
    "attn_all": 225,
    "ffn": 108,
}


@dataclasses.dataclass(frozen=True)
class FFNReferenceConfig:
    output_dir: Path
    source_rootcause_output: Path
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
    bootstrap_samples: int = 2000
    wallclock_hours: float = 24.0
    reproduction_atol: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        for key in (
            "output_dir", "source_rootcause_output", "calibration_manifest",
            "evaluation_manifest",
        ):
            result[key] = str(result[key])
        result.update({
            "format_version": FORMAT_VERSION,
            "formats": list(FORMATS),
            "screen_scopes": list(SCREEN_SCOPES),
            "confirm_scopes": list(CONFIRM_SCOPES),
            "labels": [
                "cpu_reference_backend", "fake_quantized",
                "ffn_reference_value_fp32_compute", "no_speed_or_memory_claim",
            ],
        })
        return result

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

