"""Qwen-style decoder FFN tracing and activation distribution summaries.

The tracer deliberately observes the input to ``post_attention_layernorm`` as
the FFN residual.  Inferring that residual from a layer output is ambiguous for
models that return tuples or attach additional outputs to a decoder layer.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_TRACE_FIELDS = (
    "ffn_input",
    "gate",
    "up",
    "product",
    "down_output",
    "residual",
)


@dataclasses.dataclass(frozen=True)
class FFNTrace:
    """One invocation of a decoder FFN, stored detached on CPU."""

    layer_name: str
    call_index: int
    ffn_input: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    product: torch.Tensor
    down_output: torch.Tensor
    residual: torch.Tensor

    def stage_tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in _TRACE_FIELDS}


@dataclasses.dataclass
class _PendingTrace:
    residual: torch.Tensor | None = None
    ffn_input: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    up: torch.Tensor | None = None
    product: torch.Tensor | None = None
    down_output: torch.Tensor | None = None


def _first_tensor(arguments: tuple[Any, ...], *, location: str) -> torch.Tensor:
    if not arguments or not isinstance(arguments[0], torch.Tensor):
        raise TypeError(f"{location} must receive a tensor as its first positional argument")
    return arguments[0]


def _as_named_layers(
    layers: Mapping[str, nn.Module] | Iterable[tuple[str, nn.Module]],
) -> list[tuple[str, nn.Module]]:
    items = list(layers.items() if isinstance(layers, Mapping) else layers)
    if not items:
        raise ValueError("at least one decoder layer is required")
    names: set[str] = set()
    for name, layer in items:
        if not isinstance(name, str) or not name:
            raise ValueError("layer names must be non-empty strings")
        if name in names:
            raise ValueError(f"duplicate layer name: {name}")
        if not isinstance(layer, nn.Module):
            raise TypeError(f"{name} is not a torch module")
        names.add(name)
        _validate_qwen_layer(name, layer)
    return items


def _validate_qwen_layer(name: str, layer: nn.Module) -> None:
    norm = getattr(layer, "post_attention_layernorm", None)
    mlp = getattr(layer, "mlp", None)
    if not isinstance(norm, nn.Module) or not isinstance(mlp, nn.Module):
        raise TypeError(
            f"{name} must expose post_attention_layernorm and mlp modules"
        )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(mlp, projection, None)
        if not isinstance(module, nn.Module) or not isinstance(
            getattr(module, "weight", None), torch.Tensor
        ):
            raise TypeError(f"{name}.mlp.{projection} must expose a tensor weight")


class QwenFFNTracer(contextlib.AbstractContextManager["QwenFFNTracer"]):
    """Capture exact SwiGLU stages from one or more Qwen-style decoder layers.

    ``max_tokens`` bounds retained activations per invocation.  When the bound
    is exceeded, all leading dimensions are flattened and deterministic,
    evenly spaced token rows (including both sequence ends) are kept for every
    stage.  This preserves alignment without systematically retaining only the
    leading visual tokens of a multimodal prompt.
    """

    def __init__(
        self,
        layers: Mapping[str, nn.Module] | Iterable[tuple[str, nn.Module]],
        *,
        max_tokens: int | None = 256,
        max_calls_per_layer: int = 1,
        on_trace: Callable[[FFNTrace], None] | None = None,
        storage_dtype: torch.dtype = torch.float32,
        reject_nonfinite: bool = True,
        swiglu_rtol: float = 1e-4,
        swiglu_atol: float = 1e-6,
    ) -> None:
        self.layers = _as_named_layers(layers)
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        if max_calls_per_layer < 0:
            raise ValueError("max_calls_per_layer must be non-negative")
        if not storage_dtype.is_floating_point:
            raise TypeError("storage_dtype must be floating point")
        if swiglu_rtol < 0 or swiglu_atol < 0:
            raise ValueError("SwiGLU tolerances must be non-negative")
        self.max_tokens = max_tokens
        self.max_calls_per_layer = max_calls_per_layer
        self.on_trace = on_trace
        self.storage_dtype = storage_dtype
        self.reject_nonfinite = reject_nonfinite
        self.swiglu_rtol = swiglu_rtol
        self.swiglu_atol = swiglu_atol
        self.traces: dict[str, list[FFNTrace]] = {
            name: [] for name, _layer in self.layers
        }
        self.call_counts: dict[str, int] = {
            name: 0 for name, _layer in self.layers
        }
        self._pending: dict[str, _PendingTrace] = {
            name: _PendingTrace() for name, _layer in self.layers
        }
        self._handles: list[Any] = []
        self._active = False

    def _snapshot(self, value: torch.Tensor, *, location: str) -> torch.Tensor:
        if value.ndim == 0:
            raise ValueError(f"{location} must have at least one dimension")
        detached = value.detach()
        if self.reject_nonfinite and not torch.isfinite(detached).all():
            raise ValueError(f"{location} contains NaN or Inf")
        if self.max_tokens is not None:
            token_rows = detached.numel() // detached.shape[-1]
            if token_rows > self.max_tokens:
                flat = detached.reshape(-1, detached.shape[-1])
                indices = torch.linspace(
                    0, token_rows - 1, self.max_tokens, device=flat.device,
                ).round().long()
                detached = flat.index_select(0, indices)
        return detached.to(device="cpu", dtype=self.storage_dtype).clone()

    def _set_residual(self, name: str, arguments: tuple[Any, ...]) -> None:
        pending = self._pending[name]
        if pending.ffn_input is not None:
            raise RuntimeError(f"nested or incomplete FFN invocation for {name}")
        value = _first_tensor(arguments, location=f"{name}.post_attention_layernorm")
        pending.residual = self._snapshot(value, location=f"{name}.residual")

    def _start_mlp(self, name: str, arguments: tuple[Any, ...]) -> None:
        pending = self._pending[name]
        if pending.residual is None:
            raise RuntimeError(
                f"{name}.mlp ran without a preceding post_attention_layernorm call"
            )
        if pending.ffn_input is not None:
            raise RuntimeError(f"nested MLP invocation for {name}")
        value = _first_tensor(arguments, location=f"{name}.mlp")
        pending.ffn_input = self._snapshot(value, location=f"{name}.ffn_input")

    def _capture_projection(
        self, name: str, field: str, output: Any,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{name}.{field} output must be a tensor")
        pending = self._pending[name]
        if pending.ffn_input is None:
            raise RuntimeError(f"{name}.{field} ran outside an MLP invocation")
        if getattr(pending, field) is not None:
            raise RuntimeError(f"{name}.{field} ran more than once in one MLP invocation")
        setattr(
            pending,
            field,
            self._snapshot(output, location=f"{name}.{field}"),
        )

    def _capture_product(self, name: str, arguments: tuple[Any, ...]) -> None:
        value = _first_tensor(arguments, location=f"{name}.mlp.down_proj")
        self._capture_projection(name, "product", value)

    def _finish_mlp(self, name: str, output: Any) -> None:
        pending = self._pending[name]
        required = {
            field: getattr(pending, field)
            for field in _TRACE_FIELDS
        }
        missing = [field for field, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"incomplete FFN trace for {name}: missing {missing}")
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{name}.mlp output must be a tensor")
        mlp_output = self._snapshot(output, location=f"{name}.mlp_output")

        # The checks below also document the precise Qwen/SwiGLU contract on
        # which offline replay relies.
        tensors = {key: value for key, value in required.items() if value is not None}
        if tensors["gate"].shape != tensors["up"].shape:
            raise RuntimeError(f"gate/up shape mismatch for {name}")
        if tensors["product"].shape != tensors["gate"].shape:
            raise RuntimeError(f"SwiGLU product shape mismatch for {name}")
        if tensors["ffn_input"].shape != tensors["residual"].shape:
            raise RuntimeError(f"FFN input/residual shape mismatch for {name}")
        if tensors["down_output"].shape != tensors["residual"].shape:
            raise RuntimeError(f"down output/residual shape mismatch for {name}")
        expected_product = F.silu(tensors["gate"].float()) * tensors["up"].float()
        if not torch.allclose(
            tensors["product"].float(),
            expected_product,
            rtol=self.swiglu_rtol,
            atol=self.swiglu_atol,
        ):
            maximum = float((tensors["product"].float() - expected_product).abs().max())
            raise RuntimeError(
                f"{name} does not implement SiLU(gate_proj(x)) * up_proj(x); "
                f"max error={maximum:.6g}"
            )
        if not torch.allclose(
            tensors["down_output"], mlp_output,
            rtol=self.swiglu_rtol, atol=self.swiglu_atol,
        ):
            raise RuntimeError(f"{name}.mlp output differs from down_proj output")

        call_index = self.call_counts[name]
        trace = FFNTrace(
            layer_name=name,
            call_index=call_index,
            **tensors,
        )
        self._pending[name] = _PendingTrace()
        self.call_counts[name] += 1
        if len(self.traces[name]) < self.max_calls_per_layer:
            self.traces[name].append(trace)
        if self.on_trace is not None:
            self.on_trace(trace)

    def __enter__(self) -> "QwenFFNTracer":
        if self._active:
            raise RuntimeError("tracer is already active")
        self._active = True
        try:
            for name, layer in self.layers:
                norm = layer.post_attention_layernorm
                mlp = layer.mlp
                self._handles.append(
                    norm.register_forward_pre_hook(
                        lambda _module, args, key=name: self._set_residual(key, args)
                    )
                )
                self._handles.append(
                    mlp.register_forward_pre_hook(
                        lambda _module, args, key=name: self._start_mlp(key, args)
                    )
                )
                self._handles.append(
                    mlp.gate_proj.register_forward_hook(
                        lambda _module, _args, output, key=name: self._capture_projection(
                            key, "gate", output
                        )
                    )
                )
                self._handles.append(
                    mlp.up_proj.register_forward_hook(
                        lambda _module, _args, output, key=name: self._capture_projection(
                            key, "up", output
                        )
                    )
                )
                self._handles.append(
                    mlp.down_proj.register_forward_pre_hook(
                        lambda _module, args, key=name: self._capture_product(key, args)
                    )
                )
                self._handles.append(
                    mlp.down_proj.register_forward_hook(
                        lambda _module, _args, output, key=name: self._capture_projection(
                            key, "down_output", output
                        )
                    )
                )
                self._handles.append(
                    mlp.register_forward_hook(
                        lambda _module, _args, output, key=name: self._finish_mlp(
                            key, output
                        )
                    )
                )
        except Exception:
            self._remove_hooks()
            self._active = False
            raise
        return self

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        self._remove_hooks()
        self._active = False
        if exc_type is None:
            incomplete = [
                name for name, pending in self._pending.items()
                if any(getattr(pending, field) is not None for field in _TRACE_FIELDS)
            ]
            if incomplete:
                raise RuntimeError(f"incomplete FFN traces at context exit: {incomplete}")
        return None

    def flattened(self) -> list[FFNTrace]:
        """Return traces in layer registration order, then invocation order."""

        return [
            trace
            for name, _layer in self.layers
            for trace in self.traces[name]
        ]

    def clear(self) -> None:
        if self._active:
            raise RuntimeError("cannot clear an active tracer")
        for traces in self.traces.values():
            traces.clear()
        self.call_counts = {name: 0 for name, _layer in self.layers}
        self._pending = {name: _PendingTrace() for name, _layer in self.layers}

    def retention_audit(self) -> dict[str, dict[str, int]]:
        """Report total, retained, and streamed-away invocations per layer."""

        return {
            name: {
                "calls": self.call_counts[name],
                "retained": len(self.traces[name]),
                "not_retained": self.call_counts[name] - len(self.traces[name]),
            }
            for name, _layer in self.layers
        }


@torch.no_grad()
def tensor_distribution_summary(value: torch.Tensor) -> dict[str, int | float | bool]:
    """Return JSON-ready outlier and energy statistics for one tensor."""

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError("summary requires a non-empty tensor")
    flat = value.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum())
    result: dict[str, int | float | bool] = {
        "numel": flat.numel(),
        "finite": finite_count == flat.numel(),
        "finite_fraction": finite_count / flat.numel(),
    }
    if finite_count == 0:
        result.update({
            "mean": math.nan,
            "rms": math.nan,
            "mean_abs": math.nan,
            "abs_max": math.nan,
            "abs_p99": math.nan,
            "abs_p999": math.nan,
            "kurtosis": math.nan,
            "top_0_1pct_energy_fraction": math.nan,
        })
        return result
    flat = flat[finite]
    absolute = flat.abs()
    power = flat.square()
    mean = flat.mean()
    centered = flat - mean
    variance = centered.square().mean()
    kurtosis = (
        float(centered.pow(4).mean() / variance.square())
        if float(variance) > 0.0 else 0.0
    )
    top_count = max(1, math.ceil(flat.numel() * 0.001))
    total_energy = float(power.sum())
    top_energy = float(torch.topk(power, k=top_count, sorted=False).values.sum())
    result.update({
        "mean": float(mean),
        "rms": math.sqrt(float(power.mean())),
        "mean_abs": float(absolute.mean()),
        "abs_max": float(absolute.max()),
        "abs_p99": float(torch.quantile(absolute, 0.99)),
        "abs_p999": float(torch.quantile(absolute, 0.999)),
        "kurtosis": kurtosis,
        "top_0_1pct_energy_fraction": top_energy / total_energy if total_energy else 0.0,
    })
    return result


@torch.no_grad()
def trace_summary(trace: FFNTrace) -> dict[str, Any]:
    """Summarize all captured stages and the FFN/residual branch geometry."""

    stages = {
        name: tensor_distribution_summary(value)
        for name, value in trace.stage_tensors().items()
    }
    residual = trace.residual.float().reshape(-1)
    down = trace.down_output.float().reshape(-1)
    residual_power = float(torch.dot(residual, residual))
    down_power = float(torch.dot(down, down))
    denominator = math.sqrt(residual_power * down_power)
    cosine = float(torch.dot(residual, down)) / denominator if denominator else (
        1.0 if residual_power == down_power else 0.0
    )
    residual_rms = stages["residual"]["rms"]
    down_rms = stages["down_output"]["rms"]
    assert isinstance(residual_rms, float) and isinstance(down_rms, float)
    merged_rms = math.sqrt(float((residual + down).square().mean()))
    return {
        "layer_name": trace.layer_name,
        "call_index": trace.call_index,
        "stages": stages,
        "branch": {
            "ffn_to_residual_rms": down_rms / max(residual_rms, torch.finfo(torch.float32).tiny),
            "ffn_residual_cosine": cosine,
            "merged_rms": merged_rms,
            "cancellation_ratio": merged_rms / max(
                residual_rms + down_rms, torch.finfo(torch.float32).tiny
            ),
        },
    }
