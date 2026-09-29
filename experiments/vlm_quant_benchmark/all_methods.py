"""Combined first- and second-wave method dispatcher."""
from __future__ import annotations

from .config import MethodName
from .mbq import MBQSearchResult, apply_mbq
from .safe_methods import apply_method as apply_first_wave
from .second_wave_methods import apply_second_wave


SECOND_WAVE = {MethodName.OMNIQUANT, MethodName.SPQR, MethodName.SQUEEZELLM}


def apply_method(model, selected, config, calibration=None):
    if config.method in {
        MethodName.MBQ, MethodName.VG_QIG_MBQ, MethodName.VG_UNIFORM_MBQ,
    }:
        if not isinstance(calibration, MBQSearchResult):
            raise ValueError(f"{config.method.value} requires a validated scale artifact")
        applied = apply_mbq(model, selected, config, calibration)
        applied.audit["method"] = config.method.value
        return applied
    if config.method in SECOND_WAVE:
        return apply_second_wave(model, selected, config, calibration)
    return apply_first_wave(model, selected, config, calibration)
