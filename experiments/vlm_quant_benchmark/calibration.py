from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
from torch import nn

from .selection import SelectedLinear


@dataclasses.dataclass
class ActivationStats:
    abs_sum: torch.Tensor
    abs_max: torch.Tensor
    rows: int
    reservoir: torch.Tensor

    @property
    def abs_mean(self) -> torch.Tensor:
        return self.abs_sum / max(self.rows, 1)


class ActivationCollector:
    """Stream input-channel statistics and a bounded reconstruction reservoir."""

    def __init__(self, selected: list[SelectedLinear], reservoir_rows: int = 32) -> None:
        self.selected = selected
        self.reservoir_rows = reservoir_rows
        self.stats: dict[str, ActivationStats] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationCollector":
        for item in self.selected:
            self._handles.append(item.module.register_forward_pre_hook(self._hook(item.name)))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

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
            keep = value[: self.reservoir_rows]
            if old is None:
                self.stats[name] = ActivationStats(
                    abs_sum=absolute.sum(dim=0),
                    abs_max=absolute.amax(dim=0),
                    rows=value.shape[0],
                    reservoir=keep.clone(),
                )
                return
            room = max(self.reservoir_rows - old.reservoir.shape[0], 0)
            reservoir = old.reservoir
            if room:
                reservoir = torch.cat((reservoir, keep[:room]), dim=0)
            self.stats[name] = ActivationStats(
                abs_sum=old.abs_sum + absolute.sum(dim=0),
                abs_max=torch.maximum(old.abs_max, absolute.amax(dim=0)),
                rows=old.rows + value.shape[0],
                reservoir=reservoir,
            )


@torch.inference_mode()
def collect_calibration(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    reservoir_rows: int = 32,
) -> dict[str, ActivationStats]:
    model.eval()
    with ActivationCollector(selected, reservoir_rows=reservoir_rows) as collector:
        for batch in batches:
            model(**batch, use_cache=False, return_dict=True)
    missing = [item.name for item in selected if item.name not in collector.stats]
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"calibration did not execute {len(missing)} selected modules: {preview}")
    return collector.stats

