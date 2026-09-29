"""A lightweight future-query utility scorer and its auditable ridge fit."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import atomic_json


@dataclasses.dataclass(frozen=True)
class UtilityModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    metadata: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {values.shape[-1]}")
        normalized = (values - self.mean) / self.scale
        return normalized @ self.coefficients + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> UtilityModel:
        if int(row.get("format_version", 0)) != 1:
            raise ValueError("unsupported utility model format")
        names = tuple(str(item) for item in row["feature_names"])
        mean = np.asarray(row["mean"], dtype=np.float64)
        scale = np.asarray(row["scale"], dtype=np.float64)
        coefficients = np.asarray(row["coefficients"], dtype=np.float64)
        if mean.shape != scale.shape or mean.shape != coefficients.shape or mean.shape != (len(names),):
            raise ValueError("utility model vector shapes do not match")
        if not all(np.isfinite(item).all() for item in (mean, scale, coefficients)) or not math.isfinite(
            float(row["intercept"])
        ):
            raise ValueError("utility model contains non-finite values")
        if np.any(scale <= 0):
            raise ValueError("utility model scale must be positive")
        return cls(names, mean, scale, coefficients, float(row["intercept"]), dict(row.get("metadata", {})))

    @classmethod
    def load(cls, path: Path) -> UtilityModel:
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def fit_ridge_utility(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    alpha: float = 1.0,
    balance_binary: bool = True,
    metadata: dict[str, Any] | None = None,
) -> UtilityModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
        raise ValueError("utility fit needs at least two aligned examples")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature names do not match the matrix")
    if alpha < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("utility fit inputs must be finite and alpha non-negative")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    weights = np.ones(len(y), dtype=np.float64)
    if balance_binary and set(np.unique(y)).issubset({0.0, 1.0}) and len(np.unique(y)) == 2:
        positive = y == 1
        weights[positive] = len(y) / (2 * int(positive.sum()))
        weights[~positive] = len(y) / (2 * int((~positive).sum()))
    design = np.column_stack([normalized, np.ones(len(y))])
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[-1, -1] = 0.0
    weighted = design * np.sqrt(weights[:, None])
    weighted_targets = y * np.sqrt(weights)
    solution = np.linalg.pinv(weighted.T @ weighted + regularizer) @ weighted.T @ weighted_targets
    prediction = design @ solution
    model_metadata = dict(metadata or {})
    model_metadata.update(
        {
            "examples": len(y),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "train_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "alpha": alpha,
            "balanced_binary": balance_binary,
        }
    )
    return UtilityModel(tuple(feature_names), mean, scale, solution[:-1], float(solution[-1]), model_metadata)


def save_utility_model(path: Path, model: UtilityModel) -> None:
    atomic_json(path, model.to_dict())
