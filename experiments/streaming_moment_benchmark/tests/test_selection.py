from __future__ import annotations

import numpy as np

from experiments.streaming_moment_benchmark.frames import FEATURE_NAMES, FrameObservation, FrameRef
from experiments.streaming_moment_benchmark.selection import OnlineMemorySelector
from experiments.streaming_moment_benchmark.utility import UtilityModel


def _frame(index: int, *, utility_feature: float = 0.0, motion: float = 0.0) -> FrameObservation:
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    features[0] = utility_feature
    features[-2] = motion
    return FrameObservation(
        FrameRef(f"f{index}", index, index / 10, index / 10, 0, "source"),
        features,
        np.zeros((4, 4, 3), dtype=np.uint8),
    )


def test_fractional_encode_budget_and_hard_memory_cap() -> None:
    selector = OnlineMemorySelector("uniform_fifo", memory_frames=1, encode_fps=0.5, window_sec=1)
    selector.ingest_window([_frame(index) for index in range(10)])
    selector.ingest_window([_frame(index) for index in range(10, 20)])
    result = selector.result()
    assert result.frames_admitted == 1
    assert len(result.final_refs) == 1


def test_motion_topk_evicts_lower_priority_frames() -> None:
    selector = OnlineMemorySelector("motion_topk", memory_frames=2, encode_fps=2, window_sec=1)
    selector.ingest_window([_frame(0, motion=0.1), _frame(1, motion=0.2)])
    selector.ingest_window([_frame(10, motion=0.9), _frame(11, motion=0.8)])
    assert {item.frame_id for item in selector.result().final_refs} == {"f10", "f11"}


def test_future_utility_keeps_old_high_value_event() -> None:
    size = len(FEATURE_NAMES)
    coefficients = np.zeros(size)
    coefficients[0] = 1.0
    model = UtilityModel(
        FEATURE_NAMES,
        np.zeros(size),
        np.ones(size),
        coefficients,
        0.0,
        {"runtime_query_visible": False},
    )
    selector = OnlineMemorySelector(
        "future_utility", memory_frames=4, encode_fps=2, window_sec=1, utility_model=model
    )
    selector.ingest_window([_frame(0, utility_feature=10), _frame(1, utility_feature=9)])
    for window in range(1, 5):
        selector.ingest_window([_frame(window * 10), _frame(window * 10 + 1)])
    retained = {item.frame_id for item in selector.result().final_refs}
    assert {"f0", "f1"}.issubset(retained)
    assert len(retained) <= 4
