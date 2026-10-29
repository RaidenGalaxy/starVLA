from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from experiments.infinitevl_quant.quantization import quantize_weight


@dataclasses.dataclass(frozen=True)
class ShadowObservation:
    layer: int
    gamma: float
    residual: torch.Tensor
    ffn_input: torch.Tensor
    reference: Mapping[str, torch.Tensor]
    quantized: Mapping[str, torch.Tensor]
    output: torch.Tensor


Observer = Callable[[ShadowObservation], None]


def _finite_scalar(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def tensor_rms(value: torch.Tensor) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError("RMS requires a non-empty tensor")
    flat = value.detach().float().reshape(-1)
    if not torch.isfinite(flat).all():
        raise FloatingPointError("RMS input contains NaN or Inf")
    return math.sqrt(float(flat.square().mean()))


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine tensors must have identical shapes")
    x = left.detach().float().reshape(-1)
    y = right.detach().float().reshape(-1)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError("cosine inputs contain NaN or Inf")
    denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if denominator:
        return float(torch.dot(x, y)) / denominator
    return 1.0 if float(torch.linalg.vector_norm(x)) == float(torch.linalg.vector_norm(y)) else 0.0


def distribution_summary(value: torch.Tensor) -> dict[str, int | float]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError("distribution summary requires a non-empty tensor")
    flat = value.detach().float().reshape(-1).cpu()
    if not torch.isfinite(flat).all():
        raise FloatingPointError("distribution input contains NaN or Inf")
    absolute = flat.abs()
    centered = flat - flat.mean()
    variance = centered.square().mean()
    power = flat.square()
    top_count = max(1, math.ceil(flat.numel() * 0.001))
    total_power = float(power.sum())
    top_power = float(torch.topk(power, top_count, sorted=False).values.sum())
    return {
        "numel": flat.numel(),
        "rms": math.sqrt(float(power.mean())),
        "abs_max": float(absolute.max()),
        "abs_p99": float(torch.quantile(absolute, 0.99)),
        "abs_p999": float(torch.quantile(absolute, 0.999)),
        "kurtosis": (float(centered.pow(4).mean() / variance.square()) if float(variance) > 0 else 0.0),
        "top_0_1pct_energy_fraction": top_power / total_power if total_power else 0.0,
    }


class ShadowFFN(nn.Module):
    """Compute reference and QDQ FFN outputs on the same live input.

    ``gamma=0`` returns the source module output directly and ``gamma=1``
    returns the QDQ output directly.  These explicit branches avoid arithmetic
    roundoff from ``reference + gamma * (candidate-reference)`` at the two
    parity endpoints.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        layer: int,
        gamma: float,
        bits: int = 4,
        group_size: int = 128,
        scales: Mapping[str, torch.Tensor] | None = None,
        observer: Observer | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.layer = int(layer)
        self.gamma = _finite_scalar(gamma, label="gamma")
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.observer = observer
        self._residual: torch.Tensor | None = None
        activation = getattr(source, "act_fn", None)
        self._activation: Callable[[torch.Tensor], torch.Tensor] = activation if callable(activation) else F.silu
        scale_map = {} if scales is None else dict(scales)
        for projection in ("gate", "up", "down"):
            module = getattr(source, f"{projection}_proj", None)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"source FFN lacks Linear {projection}_proj")
            scale = scale_map.get(projection)
            if scale is not None:
                scale = scale.detach().float().cpu()
                if scale.ndim != 1 or scale.numel() != module.in_features:
                    raise ValueError(f"invalid {projection} input scale")
                if not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError(f"non-positive/non-finite {projection} scale")
                prepared = module.weight.detach().float().cpu() * scale.unsqueeze(0)
            else:
                prepared = module.weight.detach().float().cpu()
            quantized = quantize_weight(
                prepared,
                bits=self.bits,
                group_size=self.group_size,
                zero_point=True,
            )
            self.register_buffer(f"{projection}_weight", quantized.dequantized)
            self.register_buffer(
                f"{projection}_bias",
                None if module.bias is None else module.bias.detach().float().cpu().clone(),
            )
            self.register_buffer(
                f"{projection}_scale",
                None if scale is None else scale.clone(),
            )
        self.quantization_audit = {
            "layer": self.layer,
            "bits": self.bits,
            "group_size": self.group_size,
            "zero_point": True,
            "input_scaled": bool(scale_map),
        }

    def bind_residual(self, value: torch.Tensor) -> None:
        if self._residual is not None:
            raise RuntimeError(f"nested or incomplete shadow FFN call at layer {self.layer}")
        if not isinstance(value, torch.Tensor):
            raise TypeError("shadow FFN residual must be a tensor")
        self._residual = value

    def _linear(self, value: torch.Tensor, projection: str) -> torch.Tensor:
        compute = value.float()
        scale = getattr(self, f"{projection}_scale")
        if scale is not None:
            compute = compute / scale.to(compute.device)
        return F.linear(
            compute,
            getattr(self, f"{projection}_weight").to(compute.device),
            None
            if getattr(self, f"{projection}_bias") is None
            else (getattr(self, f"{projection}_bias").to(compute.device)),
        )

    def _quantized(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self._linear(value, "gate")
        up = self._linear(value, "up")
        product = self._activation(gate) * up
        down = self._linear(product, "down")
        return {"gate": gate, "up": up, "product": product, "down_output": down}

    def _reference_stages(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self.source.gate_proj(value)
        up = self.source.up_proj(value)
        product = self._activation(gate) * up
        down = self.source.down_proj(product)
        return {"gate": gate, "up": up, "product": product, "down_output": down}

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self._residual
        self._residual = None
        if residual is None:
            raise RuntimeError(f"shadow FFN layer {self.layer} has no bound residual")
        reference = self._reference_stages(value)
        quantized = self._quantized(value)
        if self.gamma == 0.0:
            output = reference["down_output"]
        elif self.gamma == 1.0:
            output = quantized["down_output"]
        else:
            output = reference["down_output"] + self.gamma * (quantized["down_output"] - reference["down_output"])
        if self.observer is not None:
            self.observer(
                ShadowObservation(
                    layer=self.layer,
                    gamma=self.gamma,
                    residual=residual,
                    ffn_input=value,
                    reference=reference,
                    quantized=quantized,
                    output=output,
                )
            )
        return output


@dataclasses.dataclass(frozen=True)
class ShadowAudit:
    layers: int
    bits: int
    group_size: int
    gammas: Mapping[int, float]
    scales_supplied: bool


class ShadowFFNContext(contextlib.AbstractContextManager[ShadowAudit]):
    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        decoder_prefix: str,
        gammas: Mapping[int, float] | float,
        bits: int = 4,
        group_size: int = 128,
        scales: Mapping[str, torch.Tensor] | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not layers:
            raise ValueError("at least one decoder layer is required")
        self.layers = list(layers)
        self.decoder_prefix = decoder_prefix
        self.gammas = {
            index: _finite_scalar(
                gammas[index] if isinstance(gammas, Mapping) else gammas,
                label=f"gamma[{index}]",
            )
            for index in range(len(self.layers))
        }
        if isinstance(gammas, Mapping) and set(gammas) != set(self.gammas):
            raise ValueError("gamma mapping must contain every decoder layer exactly once")
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.scales = None if scales is None else dict(scales)
        self.observer = observer
        self._originals: dict[int, nn.Module] = {}
        self._wrappers: dict[int, ShadowFFN] = {}
        self._handles: list[Any] = []

    def _projection_scales(self, layer: int) -> dict[str, torch.Tensor] | None:
        if self.scales is None:
            return None
        result = {}
        for projection in ("gate", "up", "down"):
            name = f"{self.decoder_prefix}.{layer}.mlp.{projection}_proj"
            if name not in self.scales:
                raise ValueError(f"missing shadow input scale {name}")
            result[projection] = self.scales[name]
        return result

    def __enter__(self) -> ShadowAudit:
        if self._originals:
            raise RuntimeError("shadow FFN context is already active")
        try:
            for index, layer in enumerate(self.layers):
                source = getattr(layer, "mlp", None)
                norm = getattr(layer, "post_attention_layernorm", None)
                if not isinstance(source, nn.Module) or not isinstance(norm, nn.Module):
                    raise TypeError(f"decoder layer {index} lacks MLP/post-attention norm")
                wrapper = ShadowFFN(
                    source,
                    layer=index,
                    gamma=self.gammas[index],
                    bits=self.bits,
                    group_size=self.group_size,
                    scales=self._projection_scales(index),
                    observer=self.observer,
                )
                self._originals[index] = source
                self._wrappers[index] = wrapper
                layer.mlp = wrapper

                def bind(_module, args, *, target=wrapper):
                    if not args or not isinstance(args[0], torch.Tensor):
                        raise TypeError("post-attention norm must receive a tensor")
                    target.bind_residual(args[0])

                self._handles.append(norm.register_forward_pre_hook(bind))
        except Exception:
            self._restore()
            raise
        return ShadowAudit(
            layers=len(self.layers),
            bits=self.bits,
            group_size=self.group_size,
            gammas=dict(self.gammas),
            scales_supplied=self.scales is not None,
        )

    def _restore(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for index, source in self._originals.items():
            self.layers[index].mlp = source
        self._originals.clear()
        self._wrappers.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        originals = dict(self._originals)
        self._restore()
        failures = [
            index for index, module in originals.items() if getattr(self.layers[index], "mlp", None) is not module
        ]
        if failures:
            raise RuntimeError(f"shadow FFN context failed to restore layers {failures}")
        return None


@contextlib.contextmanager
def shadow_ffns(
    layers: Sequence[nn.Module],
    *,
    decoder_prefix: str,
    gammas: Mapping[int, float] | float,
    bits: int = 4,
    group_size: int = 128,
    scales: Mapping[str, torch.Tensor] | None = None,
    observer: Observer | None = None,
) -> Iterator[ShadowAudit]:
    with ShadowFFNContext(
        layers,
        decoder_prefix=decoder_prefix,
        gammas=gammas,
        bits=bits,
        group_size=group_size,
        scales=scales,
        observer=observer,
    ) as audit:
        yield audit


def observation_summary(observation: ShadowObservation) -> dict[str, Any]:
    reference = observation.reference
    quantized = observation.quantized
    error = quantized["down_output"] - reference["down_output"]
    residual_rms = tensor_rms(observation.residual)
    absolute_error_rms = tensor_rms(error)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "layer": observation.layer,
        "gamma": observation.gamma,
        "stages": {
            "residual": distribution_summary(observation.residual),
            "ffn_input": distribution_summary(observation.ffn_input),
            **{name: distribution_summary(value) for name, value in reference.items()},
        },
        "quantized_stage_error": {
            name: {
                "absolute_rms": tensor_rms(quantized[name] - value),
                "relative_rms": tensor_rms(quantized[name] - value) / max(tensor_rms(value), tiny),
            }
            for name, value in reference.items()
        },
        "absolute_error_rms": absolute_error_rms,
        "relative_error_to_residual_rms": absolute_error_rms / max(residual_rms, tiny),
        "error_residual_cosine": tensor_cosine(error, observation.residual),
    }
