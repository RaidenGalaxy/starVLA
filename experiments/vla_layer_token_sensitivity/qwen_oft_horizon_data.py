"""Bridge/Qwen-OFT state and action normalization for horizon experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATE_RAW_DIM = 7
STATE_MODEL_DIM = 8
ACTION_DIM = 7


def load_bridge_statistics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "bridge_dataset" not in payload:
        raise ValueError("statistics do not contain bridge_dataset")
    statistics = payload["bridge_dataset"]
    for name, dimension in (("state", STATE_MODEL_DIM), ("action", ACTION_DIM)):
        if name not in statistics:
            raise ValueError(f"statistics do not contain bridge_dataset.{name}")
        for key in ("q01", "q99"):
            values = np.asarray(statistics[name][key], dtype=np.float64)
            if values.shape != (dimension,):
                raise ValueError(f"{name}.{key} must have shape [{dimension}]")
    return statistics


def _q99_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    if values.shape[-1] != q01.size or q01.shape != q99.shape:
        raise ValueError("q99 values and statistics have incompatible shapes")
    normalized = values.copy()
    mask = q01 != q99
    normalized[..., mask] = 2.0 * (
        (values[..., mask] - q01[mask]) / (q99[mask] - q01[mask])
    ) - 1.0
    return np.clip(normalized, -2.2, 2.2).astype(np.float32)


def normalize_bridge_state(raw_state: np.ndarray | list[float], statistics: dict) -> np.ndarray:
    """Convert raw Bridge 7-D state to normalized Qwen-OFT 8-D state."""

    raw = np.asarray(raw_state, dtype=np.float64)
    if raw.shape != (STATE_RAW_DIM,):
        raise ValueError(f"raw Bridge state must have shape [{STATE_RAW_DIM}], got {raw.shape}")
    padded = np.insert(raw, 6, 0.0)
    state_stats = statistics["state"]
    normalized = _q99_normalize(padded, state_stats["q01"], state_stats["q99"])
    normalized[6] = 0.0
    normalized[7] = float(raw[6] > 0.5)
    return normalized


def normalize_bridge_actions(raw_actions: np.ndarray | list, statistics: dict) -> np.ndarray:
    """Normalize a Bridge action chunk using training-time q99/binary rules."""

    raw = np.asarray(raw_actions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[-1] != ACTION_DIM:
        raise ValueError(f"raw Bridge actions must have shape [T,{ACTION_DIM}], got {raw.shape}")
    action_stats = statistics["action"]
    normalized = _q99_normalize(raw, action_stats["q01"], action_stats["q99"])
    normalized[:, 6] = (raw[:, 6] > 0.5).astype(np.float32)
    return normalized


def persistence_baseline(normalized_state: np.ndarray | list[float], horizon: int) -> np.ndarray:
    """Zero delta pose with the current binary gripper repeated over the horizon."""

    state = np.asarray(normalized_state, dtype=np.float32)
    if state.shape != (STATE_MODEL_DIM,):
        raise ValueError(f"normalized state must have shape [{STATE_MODEL_DIM}]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    baseline = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    baseline[:, 6] = state[7]
    return baseline
