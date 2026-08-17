from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from .selection import SelectedLinear


@dataclasses.dataclass
class ActivationStats:
    abs_sum: torch.Tensor
    abs_max: torch.Tensor
    rows: int
    reservoir: torch.Tensor
    reservoir_priorities: torch.Tensor | None = None

    @property
    def abs_mean(self) -> torch.Tensor:
        return self.abs_sum / max(self.rows, 1)


def calibration_cache_key(
    *,
    model_revision: str,
    calibration_digest: str,
    selected_names: list[str],
    min_pixels: int,
    max_pixels: int,
    reservoir_rows: int,
    seed: int,
) -> str:
    payload = {
        "format_version": 1,
        "model_revision": model_revision,
        "calibration_digest": calibration_digest,
        "selected_names": selected_names,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "reservoir_rows": reservoir_rows,
        "seed": seed,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


class ActivationCollector:
    """Stream statistics and retain a deterministic uniform row reservoir."""

    def __init__(
        self,
        selected: list[SelectedLinear],
        reservoir_rows: int = 32,
        *,
        seed: int = 42,
    ) -> None:
        self.selected = selected
        self.reservoir_rows = reservoir_rows
        self.seed = seed
        self.stats: dict[str, ActivationStats] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._generators: dict[str, torch.Generator] = {}

    def __enter__(self) -> "ActivationCollector":
        for item in self.selected:
            digest = hashlib.sha256(f"{self.seed}:{item.name}".encode()).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
            self._generators[item.name] = generator
            self._handles.append(item.module.register_forward_pre_hook(self._hook(item.name)))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._generators.clear()

    def _reservoir(
        self,
        name: str,
        value: torch.Tensor,
        old: ActivationStats | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        priorities = torch.rand(value.shape[0], generator=self._generators[name])
        if old is not None and old.reservoir_priorities is not None:
            value = torch.cat((old.reservoir, value), dim=0)
            priorities = torch.cat((old.reservoir_priorities, priorities), dim=0)
        keep = min(self.reservoir_rows, priorities.numel())
        selected = torch.topk(priorities, keep, sorted=True).indices
        return value[selected].clone(), priorities[selected].clone()

    def _hook(self, name: str):
        def collect(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            value = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu()
            if value.numel() == 0:
                return
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite activation at {name}")
            absolute = value.abs()
            old = self.stats.get(name)
            reservoir, priorities = self._reservoir(name, value, old)
            self.stats[name] = ActivationStats(
                abs_sum=absolute.sum(dim=0) if old is None else old.abs_sum + absolute.sum(dim=0),
                abs_max=absolute.amax(dim=0) if old is None else torch.maximum(
                    old.abs_max, absolute.amax(dim=0)
                ),
                rows=value.shape[0] if old is None else old.rows + value.shape[0],
                reservoir=reservoir,
                reservoir_priorities=priorities,
            )

        return collect


def save_calibration_cache(
    path: Path,
    *,
    key: str,
    selected: list[SelectedLinear],
    stats: dict[str, ActivationStats],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "key": key,
        "module_names": [item.name for item in selected],
        "stats": {name: dataclasses.asdict(value) for name, value in stats.items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_calibration_cache(
    path: Path,
    *,
    key: str,
    selected: list[SelectedLinear],
) -> dict[str, ActivationStats] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = [item.name for item in selected]
    if payload.get("format_version") != 1 or payload.get("key") != key:
        return None
    if payload.get("module_names") != names:
        return None
    stats = {
        name: ActivationStats(**value)
        for name, value in payload["stats"].items()
    }
    for item in selected:
        value = stats.get(item.name)
        if value is None or value.abs_sum.numel() != item.module.in_features:
            return None
        if not torch.isfinite(value.abs_sum).all() or not torch.isfinite(value.reservoir).all():
            return None
    return stats


@torch.inference_mode()
def collect_calibration(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    reservoir_rows: int = 32,
    seed: int = 42,
) -> dict[str, ActivationStats]:
    model.eval()
    with ActivationCollector(
        selected, reservoir_rows=reservoir_rows, seed=seed,
    ) as collector:
        for batch in batches:
            model(**batch, use_cache=False, return_dict=True)
    missing = [item.name for item in selected if item.name not in collector.stats]
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"calibration did not execute {len(missing)} selected modules: {preview}")
    return collector.stats


def collect_calibration_cached(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    cache_dir: Path,
    key: str,
    reservoir_rows: int = 32,
    seed: int = 42,
) -> tuple[dict[str, ActivationStats], bool]:
    path = cache_dir / f"{key}.pt"
    cached = load_calibration_cache(path, key=key, selected=selected)
    if cached is not None:
        return cached, True
    stats = collect_calibration(
        model, selected, batches,
        reservoir_rows=reservoir_rows, seed=seed,
    )
    save_calibration_cache(path, key=key, selected=selected, stats=stats)
    return stats, False
