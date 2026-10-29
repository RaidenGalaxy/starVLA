from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import MEDIATOR_GROUPS, STATE_PREFIXES

_ACTIVE_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "ffn_quant_mechanism_gdn_layer",
    default=None,
)


MEDIATOR_KEYS = ("q", "k", "v", "g", "beta", "output_gate")


def _clone(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().float().cpu().clone()
    if not torch.isfinite(result).all():
        raise FloatingPointError("GDN mediator contains NaN or Inf")
    return result


@dataclasses.dataclass
class GDNReference:
    layers: dict[int, dict[str, torch.Tensor]] = dataclasses.field(default_factory=dict)

    def record_recurrence(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if layer in self.layers and any(key in self.layers[layer] for key in ("q", "k", "v", "g", "beta")):
            raise RuntimeError(f"GDN layer {layer} recurrence ran more than once")
        row = self.layers.setdefault(layer, {})
        row.update(
            {
                "q": _clone(q),
                "k": _clone(k),
                "v": _clone(v),
                "g": _clone(g),
                "beta": _clone(beta),
            }
        )

    def record_output_gate(self, layer: int, gate: torch.Tensor) -> None:
        row = self.layers.setdefault(layer, {})
        if "output_gate" in row:
            raise RuntimeError(f"GDN layer {layer} output gate ran more than once")
        row["output_gate"] = _clone(gate)

    def validate(self, expected_layers: int = 27) -> None:
        if len(self.layers) != expected_layers:
            raise RuntimeError(f"captured {len(self.layers)} GDN references, expected {expected_layers}")
        incomplete = {
            layer: sorted(set(MEDIATOR_KEYS) - set(row))
            for layer, row in self.layers.items()
            if set(row) != set(MEDIATOR_KEYS)
        }
        if incomplete:
            raise RuntimeError(f"incomplete GDN mediator references: {incomplete}")

    def payload(self, *, sample_id: str) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": 1,
            "sample_id": sample_id,
            "layers": {str(layer): row for layer, row in sorted(self.layers.items())},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, sample_id: str) -> "GDNReference":
        if payload.get("format_version") != 1 or payload.get("sample_id") != sample_id:
            raise RuntimeError("GDN reference identity mismatch")
        raw = payload.get("layers")
        if not isinstance(raw, Mapping):
            raise RuntimeError("GDN reference does not contain a layer mapping")
        reference = cls(
            {
                int(layer): {
                    str(key): value.float().cpu() for key, value in row.items() if isinstance(value, torch.Tensor)
                }
                for layer, row in raw.items()
                if isinstance(row, Mapping)
            }
        )
        reference.validate()
        return reference


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), p=2, dim=-1, eps=1e-6)


def _state_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left, right = reference.float(), candidate.float()
    delta = right - left
    norm = torch.linalg.vector_norm(left).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = F.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1e-12)
    return {
        "normalized_frobenius": float(torch.linalg.vector_norm(delta) / norm),
        "cosine_drift": float(1.0 - cosine),
        "max_abs_error": float(delta.abs().max()),
    }


def _validate_mediator_shapes(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> None:
    for key in ("q", "k", "v", "g", "beta"):
        if key not in reference:
            raise RuntimeError(f"reference GDN mediators omit {key}")
        if candidate[key].shape != reference[key].shape:
            raise RuntimeError(
                f"GDN mediator shape mismatch for {key}: {tuple(candidate[key].shape)} != {tuple(reference[key].shape)}"
            )


@torch.no_grad()
def recurrent_gdn_intervention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    reference: Mapping[str, torch.Tensor],
    repair_groups: Sequence[str] = (),
    carry_clamp: bool = False,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    state_prefixes: Sequence[int] = STATE_PREFIXES,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Run a paired recurrent GDN with explicit mediator and carry repairs.

    The current-token candidate output is computed before an optional carry
    clamp.  Only the state handed to the next token is replaced by the paired
    BF16 state, which makes the temporal path intervention unambiguous.
    """

    if cu_seqlens is not None:
        raise NotImplementedError("packed GDN sequences are outside this protocol")
    groups = frozenset(repair_groups)
    unknown = groups - set(MEDIATOR_GROUPS)
    if unknown:
        raise ValueError(f"unknown GDN mediator repair groups: {sorted(unknown)}")
    candidate = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    _validate_mediator_shapes(candidate, reference)
    if any(value.ndim != 4 for value in (q, k, v)):
        raise ValueError("q/k/v must have [batch,time,heads,dim] shape")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g/beta must have [batch,time,heads] shape")
    if initial_state is not None:
        raise NotImplementedError("GDN causal mediation is teacher-forced/use_cache=False only")

    def choose(key: str, group: str) -> torch.Tensor:
        return reference[key].to(q.device) if group in groups else candidate[key].float()

    cq = choose("q", "q")
    ck = choose("k", "write")
    cv = choose("v", "write")
    cbeta = choose("beta", "write")
    cg = choose("g", "decay")
    rq, rk, rv = (reference[key].to(q.device).float() for key in ("q", "k", "v"))
    rg = reference["g"].to(q.device).float()
    rbeta = reference["beta"].to(q.device).float()
    if use_qk_l2norm_in_kernel:
        cq, ck, rq, rk = map(_normalize, (cq, ck, rq, rk))
    else:
        cq, ck, rq, rk = (value.float() for value in (cq, ck, rq, rk))
    cv, cg, cbeta = cv.float(), cg.float(), cbeta.float()
    rv, rg, rbeta = rv.float(), rg.float(), rbeta.float()
    key_dim = cq.shape[-1]
    factor = key_dim**-0.5 if scale is None else float(scale)
    cq, rq = cq * factor, rq * factor
    batch, steps, heads, _ = cq.shape
    value_dim = cv.shape[-1]
    candidate_state = torch.zeros(
        batch,
        heads,
        key_dim,
        value_dim,
        device=q.device,
        dtype=torch.float32,
    )
    reference_state = torch.zeros_like(candidate_state)
    outputs = []
    requested = set(int(value) for value in state_prefixes if int(value) > 0)
    state_rows: dict[str, dict[str, Any]] = {}
    for index in range(steps):
        reference_state = reference_state * rg[:, index].exp()[..., None, None]
        rkey = rk[:, index]
        rresidual = rv[:, index] - torch.einsum(
            "bhkv,bhk->bhv",
            reference_state,
            rkey,
        )
        rresidual = rresidual * rbeta[:, index, :, None]
        reference_state = reference_state + rkey.unsqueeze(-1) * rresidual.unsqueeze(-2)

        candidate_state = candidate_state * cg[:, index].exp()[..., None, None]
        key = ck[:, index]
        residual = cv[:, index] - torch.einsum(
            "bhkv,bhk->bhv",
            candidate_state,
            key,
        )
        residual = residual * cbeta[:, index, :, None]
        candidate_state = candidate_state + key.unsqueeze(-1) * residual.unsqueeze(-2)
        outputs.append(
            torch.einsum(
                "bhkv,bhk->bhv",
                candidate_state,
                cq[:, index],
            )
        )
        pre_clamp = candidate_state
        if carry_clamp:
            candidate_state = reference_state.clone()
        prefix = index + 1
        if prefix in requested or prefix == steps:
            row = {
                "pre_clamp": _state_error(reference_state, pre_clamp),
                "carried": _state_error(reference_state, candidate_state),
            }
            if prefix in requested:
                state_rows[str(prefix)] = row
            if prefix == steps:
                state_rows["final"] = row
    output = (
        torch.stack(outputs, dim=1)
        if outputs
        else cv.new_empty(
            batch,
            0,
            heads,
            value_dim,
        )
    )
    if not torch.isfinite(output).all() or not torch.isfinite(candidate_state).all():
        raise FloatingPointError("GDN intervention produced NaN or Inf")
    audit = {
        "repair_groups": sorted(groups),
        "carry_clamp": bool(carry_clamp),
        "sequence_tokens": steps,
        "states": state_rows,
    }
    return output, candidate_state if output_final_state else None, audit


@dataclasses.dataclass(frozen=True)
class GDNInterventionAudit:
    mode: str
    gdn_layers: int
    repair_groups: tuple[str, ...]
    carry_clamp: bool
    backend: str


class GDNInterventionContext(contextlib.AbstractContextManager[GDNInterventionAudit]):
    """Patch one loaded InfiniteVL instance and restore it exactly on exit."""

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_reference: bool = False,
        reference: GDNReference | None = None,
        repair_groups: Sequence[str] = (),
        carry_clamp: bool = False,
        state_prefixes: Sequence[int] = STATE_PREFIXES,
    ) -> None:
        if capture_reference == (reference is not None):
            raise ValueError("choose exactly one of capture_reference or reference")
        self.model = model
        self.capture_reference = bool(capture_reference)
        self.reference = GDNReference() if capture_reference else reference
        assert self.reference is not None
        self.repair_groups = tuple(sorted(set(repair_groups)))
        unknown = set(self.repair_groups) - set(MEDIATOR_GROUPS)
        if unknown:
            raise ValueError(f"unknown mediator groups: {sorted(unknown)}")
        self.carry_clamp = bool(carry_clamp)
        self.state_prefixes = tuple(int(value) for value in state_prefixes)
        self.state_audits: dict[int, dict[str, Any]] = {}
        self._handles: list[Any] = []
        self._tokens: list[contextvars.Token] = []
        self._remote: Any = None
        self._original_chunk: Any = None
        self._original_recurrent: Any = None
        self._gdn_layers: list[tuple[int, nn.Module]] = []

    def _active(self) -> int:
        layer = _ACTIVE_LAYER.get()
        if layer is None:
            raise RuntimeError("GDN kernel ran outside a registered GatedDeltaNet module")
        return layer

    def _dispatch(self, original, *args, **kwargs):
        layer = self._active()
        names = ("q", "k", "v", "g", "beta")
        values = {name: kwargs.get(name, args[index] if index < len(args) else None) for index, name in enumerate(names)}
        if any(not isinstance(value, torch.Tensor) for value in values.values()):
            raise TypeError("GDN kernel did not receive tensor q/k/v/g/beta")
        if self.capture_reference:
            self.reference.record_recurrence(layer, **values)
            return original(*args, **kwargs)
        layer_reference = self.reference.layers.get(layer)
        if layer_reference is None:
            raise RuntimeError(f"missing BF16 GDN reference for layer {layer}")
        passthrough = {
            key: kwargs[key]
            for key in (
                "scale",
                "initial_state",
                "output_final_state",
                "cu_seqlens",
                "use_qk_l2norm_in_kernel",
            )
            if key in kwargs
        }
        output, final_state, audit = recurrent_gdn_intervention(
            **values,
            reference=layer_reference,
            repair_groups=self.repair_groups,
            carry_clamp=self.carry_clamp,
            state_prefixes=self.state_prefixes,
            **passthrough,
        )
        self.state_audits[layer] = audit
        return output, final_state

    def __enter__(self) -> GDNInterventionAudit:
        module_name = self.model.__class__.__module__
        remote = sys.modules.get(module_name)
        if remote is None:
            raise RuntimeError(f"cannot locate loaded InfiniteVL module {module_name}")
        for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
            if not callable(getattr(remote, name, None)):
                raise RuntimeError(f"InfiniteVL module lacks callable {name}")
        self._remote = remote
        self._original_chunk = remote.chunk_gated_delta_rule
        self._original_recurrent = remote.fused_recurrent_gated_delta_rule
        self._gdn_layers = sorted(
            [
                (int(module.layer_idx), module)
                for module in self.model.modules()
                if module.__class__.__name__ == "GatedDeltaNet"
            ],
            key=lambda item: item[0],
        )
        if len(self._gdn_layers) != 27:
            raise RuntimeError(f"expected 27 GDN modules, got {len(self._gdn_layers)}")
        try:
            for layer, module in self._gdn_layers:

                def before(_module, _args, *, index=layer):
                    self._tokens.append(_ACTIVE_LAYER.set(index))

                def after(_module, _args, _output):
                    if not self._tokens:
                        raise RuntimeError("unbalanced GDN module hooks")
                    _ACTIVE_LAYER.reset(self._tokens.pop())

                self._handles.append(module.register_forward_pre_hook(before))
                self._handles.append(module.register_forward_hook(after))

                def output_gate(_module, _args, output, *, index=layer):
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("GDN output gate must return a tensor")
                    if self.capture_reference:
                        self.reference.record_output_gate(index, output)
                        return None
                    if "output_gate" in self.repair_groups:
                        expected = self.reference.layers[index]["output_gate"].to(output.device)
                        if expected.shape != output.shape:
                            raise RuntimeError("GDN output-gate reference shape mismatch")
                        return expected.to(output.dtype)
                    return None

                self._handles.append(module.g_proj.register_forward_hook(output_gate))

            remote.chunk_gated_delta_rule = lambda *args, **kwargs: self._dispatch(
                self._original_chunk,
                *args,
                **kwargs,
            )
            remote.fused_recurrent_gated_delta_rule = lambda *args, **kwargs: self._dispatch(
                self._original_recurrent,
                *args,
                **kwargs,
            )
        except Exception:
            self._restore()
            raise
        return GDNInterventionAudit(
            mode="capture_reference" if self.capture_reference else "intervene",
            gdn_layers=len(self._gdn_layers),
            repair_groups=self.repair_groups,
            carry_clamp=self.carry_clamp,
            backend=("native_chunk_or_recurrent_capture" if self.capture_reference else "paired_recurrent_intervention"),
        )

    def _restore(self) -> None:
        if self._remote is not None:
            if self._original_chunk is not None:
                self._remote.chunk_gated_delta_rule = self._original_chunk
            if self._original_recurrent is not None:
                self._remote.fused_recurrent_gated_delta_rule = self._original_recurrent
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        while self._tokens:
            _ACTIVE_LAYER.reset(self._tokens.pop())

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        remote, chunk, recurrent = (
            self._remote,
            self._original_chunk,
            self._original_recurrent,
        )
        self._restore()
        if remote is not None and (
            remote.chunk_gated_delta_rule is not chunk or remote.fused_recurrent_gated_delta_rule is not recurrent
        ):
            raise RuntimeError("GDN intervention failed to restore remote kernels")
        if exc_type is None and self.capture_reference:
            self.reference.validate()
        return None


@contextlib.contextmanager
def capture_gdn_reference(model: nn.Module) -> Iterator[GDNReference]:
    context = GDNInterventionContext(model, capture_reference=True)
    with context:
        yield context.reference


@contextlib.contextmanager
def intervene_gdn(
    model: nn.Module,
    reference: GDNReference,
    *,
    repair_groups: Sequence[str] = (),
    carry_clamp: bool = False,
    state_prefixes: Sequence[int] = STATE_PREFIXES,
) -> Iterator[GDNInterventionContext]:
    context = GDNInterventionContext(
        model,
        reference=reference,
        repair_groups=repair_groups,
        carry_clamp=carry_clamp,
        state_prefixes=state_prefixes,
    )
    with context:
        yield context


def aggregate_state_audits(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[int, str, str, str], list[float]] = {}
    for sample in rows:
        for layer, audit in sample["state_audits"].items():
            for prefix, state_row in audit["states"].items():
                for location in ("pre_clamp", "carried"):
                    for metric, value in state_row[location].items():
                        values.setdefault(
                            (int(layer), str(prefix), location, metric),
                            [],
                        ).append(float(value))
    aggregates = [
        {
            "layer": layer,
            "prefix": prefix,
            "location": location,
            "metric": metric,
            "samples": len(items),
            "mean": sum(items) / len(items),
        }
        for (layer, prefix, location, metric), items in sorted(values.items())
    ]
    return {
        "analysis": "paired_gdn_state_drift",
        "samples": len(rows),
        "aggregates": aggregates,
    }
