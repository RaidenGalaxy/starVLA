from __future__ import annotations

import contextlib
import contextvars
import math
import statistics
import sys
import time
import types
from collections.abc import Iterator
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


_CURRENT_GDN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "infinitevl_current_gdn", default=None,
)
_STATE_PROBE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "infinitevl_state_probe", default=None,
)
_FORCE_RECURRENT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "infinitevl_force_recurrent", default=False,
)

FLA_REFERENCE_COMMIT = "954438d1fcb5e1bb05c22f9908de9c5c2df74ae5"
FLA_REFERENCE_SHA256 = "d1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb"
FLA_REFERENCE_URL = (
    "https://github.com/fla-org/flash-linear-attention/blob/"
    f"{FLA_REFERENCE_COMMIT}/fla/ops/gated_delta_rule/naive.py"
)


def _l2_normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), p=2, dim=-1, eps=1e-6)


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Differentiable FP32 reference recurrence matching FLA's delta rule."""
    if cu_seqlens is not None:
        raise NotImplementedError("packed sequences are outside the CPU pilot")
    if not all(value.ndim == 4 for value in (q, k, v)):
        raise ValueError("q, k, and v must have [batch, time, heads, dim] shape")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, and v batch/time/head dimensions must match")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must have [batch, time, heads] shape")
    if not all(torch.isfinite(value).all() for value in (q, k, v, g, beta)):
        raise ValueError("Gated Delta Rule inputs contain NaN or Inf")

    qf = _l2_normalize(q) if use_qk_l2norm_in_kernel else q.float()
    kf = _l2_normalize(k) if use_qk_l2norm_in_kernel else k.float()
    vf, gf, betaf = v.float(), g.float(), beta.float()
    batch, steps, heads, key_dim = qf.shape
    qf = qf * (key_dim ** -0.5 if scale is None else float(scale))
    value_dim = vf.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch, heads, key_dim, value_dim, device=q.device, dtype=torch.float32,
        )
    else:
        expected = (batch, heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(f"initial state has {tuple(initial_state.shape)}, expected {expected}")
        state = initial_state.float()

    outputs: list[torch.Tensor] = []
    probe = _STATE_PROBE.get()
    layer_name = _CURRENT_GDN.get()
    requested = set() if probe is None else set(probe["prefixes"])
    for index in range(steps):
        state = state * gf[:, index].exp()[..., None, None]
        key = kf[:, index]
        residual = vf[:, index] - torch.einsum("bhkv,bhk->bhv", state, key)
        residual = residual * betaf[:, index, :, None]
        state = state + key.unsqueeze(-1) * residual.unsqueeze(-2)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, qf[:, index]))
        prefix = index + 1
        if layer_name is not None and prefix in requested:
            probe["states"].setdefault(layer_name, {})[prefix] = state.detach().cpu().clone()
    output = torch.stack(outputs, dim=1) if outputs else vf.new_empty(
        batch, 0, heads, value_dim,
    )
    if not torch.isfinite(output).all() or not torch.isfinite(state).all():
        raise FloatingPointError("CPU Gated Delta Rule produced NaN or Inf")
    return output, state if output_final_state else None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FP32 port of FLA's pinned ``naive_chunk_gated_delta_rule``.

    InfiniteVL asks the FLA kernel to normalize q/k internally, so the port
    performs that normalization before applying the upstream chunk equations.
    QIG and state tracing explicitly select the recurrent implementation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if _STATE_PROBE.get() is not None or _FORCE_RECURRENT.get():
        return recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=initial_state,
            scale=scale,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    if cu_seqlens is not None:
        raise NotImplementedError("packed sequences are outside the CPU pilot")
    if not all(value.ndim == 4 for value in (q, k, v)):
        raise ValueError("q, k, and v must have [batch, time, heads, dim] shape")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, and v batch/time/head dimensions must match")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must have [batch, time, heads] shape")
    if not all(torch.isfinite(value).all() for value in (q, k, v, g, beta)):
        raise ValueError("Gated Delta Rule inputs contain NaN or Inf")
    if q.shape[1] == 0:
        state = None if initial_state is None else initial_state.float()
        return v.float(), state if output_final_state else None

    qf = _l2_normalize(q) if use_qk_l2norm_in_kernel else q.float()
    kf = _l2_normalize(k) if use_qk_l2norm_in_kernel else k.float()
    qf, kf = qf.transpose(1, 2).contiguous(), kf.transpose(1, 2).contiguous()
    vf = v.transpose(1, 2).contiguous().float()
    betaf = beta.transpose(1, 2).contiguous().float()
    gf = g.transpose(1, 2).contiguous().float()
    original_steps = qf.shape[-2]
    pad_length = (chunk_size - original_steps % chunk_size) % chunk_size
    if pad_length:
        qf = F.pad(qf, (0, 0, 0, pad_length))
        kf = F.pad(kf, (0, 0, 0, pad_length))
        vf = F.pad(vf, (0, 0, 0, pad_length))
        betaf = F.pad(betaf, (0, pad_length))
        gf = F.pad(gf, (0, pad_length))

    batch, heads, padded_steps, key_dim = qf.shape
    value_dim = vf.shape[-1]
    chunks = padded_steps // chunk_size
    qf = qf * (key_dim ** -0.5 if scale is None else float(scale))
    vf = vf * betaf[..., None]
    k_beta = kf * betaf[..., None]

    def split_chunks(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, heads, chunks, chunk_size, value.shape[-1])

    q_chunks = split_chunks(qf)
    k_chunks = split_chunks(kf)
    v_chunks = split_chunks(vf)
    kb_chunks = split_chunks(k_beta)
    decay = gf.reshape(batch, heads, chunks, chunk_size).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    lower_decay = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    diagonal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    inverse = -(
        (kb_chunks @ k_chunks.transpose(-1, -2)) * lower_decay
    ).masked_fill(diagonal_mask, 0)
    # This in-place triangular recurrence is the exact operation used by the
    # pinned upstream PyTorch reference. QIG never enters this path.
    for index in range(1, chunk_size):
        inverse[..., index, :index] = (
            inverse[..., index, :index].clone()
            + (
                inverse[..., index, :index, None].clone()
                * inverse[..., :index, :index].clone()
            ).sum(-2)
        )
    inverse = inverse + torch.eye(
        chunk_size, dtype=torch.float32, device=q.device,
    )
    value_cumsum = inverse @ v_chunks
    key_cumdecay = inverse @ (kb_chunks * decay_exp)

    state = kf.new_zeros(batch, heads, key_dim, value_dim)
    if initial_state is not None:
        expected = (batch, heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial state has {tuple(initial_state.shape)}, expected {expected}"
            )
        state = initial_state.float()
    output = torch.zeros_like(value_cumsum)
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for index in range(chunks):
        q_i, k_i, value_i = (
            q_chunks[:, :, index],
            k_chunks[:, :, index],
            value_cumsum[:, :, index],
        )
        attention = (
            (q_i @ k_i.transpose(-1, -2)) * lower_decay[:, :, index]
        ).masked_fill(causal_mask, 0)
        value_prime = key_cumdecay[:, :, index] @ state
        value_new = value_i - value_prime
        output_inter = (q_i * decay[:, :, index, :, None].exp()) @ state
        output[:, :, index] = output_inter + attention @ value_new
        final_decay = decay[:, :, index, -1]
        state = (
            state * final_decay[..., None, None].exp()
            + (
                k_i
                * (final_decay[..., None] - decay[:, :, index]).exp()[..., None]
            ).transpose(-1, -2)
            @ value_new
        )
    output = output.reshape(batch, heads, padded_steps, value_dim)
    output = output[:, :, :original_steps].transpose(1, 2).contiguous()
    if not torch.isfinite(output).all() or not torch.isfinite(state).all():
        raise FloatingPointError("CPU chunk Gated Delta Rule produced NaN or Inf")
    return output, state if output_final_state else None


@contextlib.contextmanager
def qig_recurrent_reference() -> Iterator[None]:
    """Force differentiable recurrent GDN execution within QIG attribution."""
    token = _FORCE_RECURRENT.set(True)
    try:
        yield
    finally:
        _FORCE_RECURRENT.reset(token)


class CPURMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **_: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps,
        )
        return (normalized * self.weight.float()).to(value.dtype)


class CPUFusedRMSNormGated(CPURMSNorm):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, norm_before_gate: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(hidden_size, eps=eps, **kwargs)
        if not norm_before_gate:
            raise NotImplementedError("InfiniteVL expects norm_before_gate=True")

    def forward(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return super().forward(value) * F.silu(gate.float()).to(value.dtype)


class CPUShortConvolution(nn.Module):
    """Depthwise causal convolution with FLA-compatible state shape [B,C,K]."""
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        activation: str | None = None,
        bias: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or kernel_size <= 0:
            raise ValueError("hidden_size and kernel_size must be positive")
        self.hidden_size = int(hidden_size)
        self.kernel_size = (int(kernel_size),)
        self._kernel_size = int(kernel_size)
        self.activation = activation
        # Match nn.Conv1d / upstream FLA state_dict layout exactly.
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation in (None, "identity"):
            return value
        if self.activation == "silu":
            return F.silu(value)
        raise ValueError(f"unsupported short-convolution activation: {self.activation}")

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            raise NotImplementedError("packed short convolution is outside the CPU pilot")
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError("x must have [batch, time, hidden_size] shape")
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        channels = x.transpose(1, 2).float()
        if cache is None:
            padded = F.pad(channels, (self._kernel_size - 1, 0))
        else:
            expected = (x.shape[0], self.hidden_size, self._kernel_size)
            if tuple(cache.shape) != expected:
                raise ValueError(f"cache has {tuple(cache.shape)}, expected {expected}")
            padded = torch.cat((cache.float()[:, :, 1:], channels), dim=-1)
        result = F.conv1d(
            padded, self.weight.float(), bias=(
                None if self.bias is None else self.bias.float()
            ), groups=self.hidden_size,
        ).transpose(1, 2)
        if residual is not None:
            result = result + residual.float()
        result = self._activate(result).to(x.dtype)
        state_source = torch.cat((
            torch.zeros(
                x.shape[0], self.hidden_size, self._kernel_size,
                device=x.device, dtype=torch.float32,
            ) if cache is None else cache.float(),
            channels,
        ), dim=-1)
        # Cache tensors are part of the recurrent state contract and remain
        # FP32 even when callers provide lower-precision activations.
        final_state = state_source[:, :, -self._kernel_size:]
        if cache is not None and x.shape[1] == 1:
            cache.copy_(final_state)
            final_state = cache
        return result, final_state if output_final_state else None


def _get_unpad_data(attention_mask: torch.Tensor):
    lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    cumulative = F.pad(lengths.cumsum(0), (1, 0))
    return indices, cumulative, int(lengths.max().item())


def _index_first_axis(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value[indices]


def _pad_input(
    value: torch.Tensor, indices: torch.Tensor, batch: int, sequence: int,
) -> torch.Tensor:
    output = value.new_zeros((batch * sequence, *value.shape[1:]))
    output[indices] = value
    return output.reshape(batch, sequence, *value.shape[1:])


def install_fla_cpu_shims() -> None:
    """Install import-compatible FLA modules before loading remote model code."""
    fla = types.ModuleType("fla")
    layers = types.ModuleType("fla.layers")
    layer_utils = types.ModuleType("fla.layers.utils")
    modules = types.ModuleType("fla.modules")
    ops = types.ModuleType("fla.ops")
    gated = types.ModuleType("fla.ops.gated_delta_rule")
    layer_utils.get_unpad_data = _get_unpad_data
    layer_utils.index_first_axis = _index_first_axis
    layer_utils.pad_input = _pad_input
    modules.FusedRMSNormGated = CPUFusedRMSNormGated
    modules.RMSNorm = CPURMSNorm
    modules.ShortConvolution = CPUShortConvolution
    gated.chunk_gated_delta_rule = chunk_gated_delta_rule
    gated.fused_recurrent_gated_delta_rule = recurrent_gated_delta_rule
    fla.layers, fla.modules, fla.ops = layers, modules, ops
    layers.utils, ops.gated_delta_rule = layer_utils, gated
    sys.modules.update({
        "fla": fla,
        "fla.layers": layers,
        "fla.layers.utils": layer_utils,
        "fla.modules": modules,
        "fla.ops": ops,
        "fla.ops.gated_delta_rule": gated,
    })


@contextlib.contextmanager
def capture_gdn_states(
    model: nn.Module, prefixes: tuple[int, ...],
) -> Iterator[dict[str, dict[int, torch.Tensor]]]:
    if not prefixes or any(value <= 0 for value in prefixes):
        raise ValueError("state-probe prefixes must be positive")
    storage: dict = {"prefixes": tuple(sorted(set(prefixes))), "states": {}}
    token = _STATE_PROBE.set(storage)
    handles = []
    module_tokens: list[contextvars.Token] = []

    def before(name: str) -> Callable:
        def hook(_module, _args):
            module_tokens.append(_CURRENT_GDN.set(name))
        return hook

    def after(_module, _args, _output):
        if module_tokens:
            _CURRENT_GDN.reset(module_tokens.pop())

    for name, module in model.named_modules():
        if module.__class__.__name__ == "GatedDeltaNet":
            handles.append(module.register_forward_pre_hook(before(name)))
            handles.append(module.register_forward_hook(after))
    try:
        yield storage["states"]
    finally:
        for handle in handles:
            handle.remove()
        while module_tokens:
            _CURRENT_GDN.reset(module_tokens.pop())
        _STATE_PROBE.reset(token)


def choose_torch_threads(candidates: tuple[int, ...] = (8, 16)) -> dict:
    """Choose once using a deterministic FFN-shaped CPU microbenchmark."""
    if not candidates:
        raise ValueError("at least one thread candidate is required")
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(8, 2048, generator=generator)
    gate = torch.randn(4096, 2048, generator=generator)
    up = torch.randn(4096, 2048, generator=generator)
    down = torch.randn(2048, 4096, generator=generator)
    timings = {}
    for threads in candidates:
        torch.set_num_threads(threads)
        samples = []
        for _ in range(4):
            started = time.perf_counter()
            F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
            samples.append(time.perf_counter() - started)
        timings[str(threads)] = samples[1:]
    selected = min(
        candidates, key=lambda value: statistics.median(timings[str(value)]),
    )
    torch.set_num_threads(selected)
    return {
        "selected": selected,
        "interop_threads": torch.get_num_interop_threads(),
        "seconds": timings,
        "median_seconds": {
            key: statistics.median(value) for key, value in timings.items()
        },
        "benchmark": "8x2048 -> gate/up 4096 -> down 2048, FP32",
    }
