from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from .official_awq import (
    SafeTensorStore,
    dequantize_awq_gemm,
    pack_awq_gemm,
    unpack_awq_gemm,
)
from .reporting import atomic_json


EXPECTED_PACKED_MODULES = 252
EXPECTED_REVISION = "e7b623934290c5a4da0ee3c6e1e57bfb6b5abbf2"


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def select_distinct_shapes(store: SafeTensorStore, count: int) -> list[str]:
    selected: list[str] = []
    shapes: set[tuple[int, int]] = set()
    for name in store.keys():
        if not name.endswith(".qweight"):
            continue
        value = store.get(name)
        shape = (value.shape[1] * 8, value.shape[0])
        if shape in shapes:
            continue
        selected.append(name.removesuffix(".qweight"))
        shapes.add(shape)
        if len(selected) == count:
            return selected
    raise ValueError(f"checkpoint has only {len(selected)} distinct packed layer shapes")


@torch.inference_mode()
def validate_layer(store: SafeTensorStore, module_name: str, seed: int) -> dict:
    qweight = store.get(f"{module_name}.qweight")
    qzeros = store.get(f"{module_name}.qzeros")
    scales = store.get(f"{module_name}.scales")
    unpacked_weight = unpack_awq_gemm(qweight)
    unpacked_zeros = unpack_awq_gemm(qzeros)
    if not torch.equal(pack_awq_gemm(unpacked_weight), qweight):
        raise AssertionError(f"checkpoint pack/unpack mismatch: {module_name}.qweight")
    if not torch.equal(pack_awq_gemm(unpacked_zeros), qzeros):
        raise AssertionError(f"checkpoint pack/unpack mismatch: {module_name}.qzeros")

    weight = dequantize_awq_gemm(qweight, qzeros, scales, group_size=128)
    if not torch.isfinite(weight).all():
        raise ValueError(f"non-finite dequantized weight: {module_name}")
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(3, weight.shape[1], generator=generator)
    actual = F.linear(inputs, weight)
    expanded_scales = scales.float().repeat_interleave(128, dim=0)
    expanded_zeros = unpacked_zeros.float().repeat_interleave(128, dim=0)
    manual_weight_in_out = (
        unpacked_weight.float() - expanded_zeros
    ) * expanded_scales
    expected = inputs @ manual_weight_in_out
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    return {
        "module": module_name,
        "qweight_shape": list(qweight.shape),
        "qzeros_shape": list(qzeros.shape),
        "scales_shape": list(scales.shape),
        "linear_weight_shape": list(weight.shape),
        "qweight_sha256": tensor_sha256(qweight),
        "pack_roundtrip_exact": True,
        "linear_output_close": True,
        "finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qwen official AWQ packed tensors")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=EXPECTED_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SafeTensorStore(args.checkpoint)
    qweight_names = [name for name in store.keys() if name.endswith(".qweight")]
    if len(qweight_names) != EXPECTED_PACKED_MODULES:
        raise ValueError(
            f"official AWQ packed modules {len(qweight_names)} != {EXPECTED_PACKED_MODULES}"
        )
    modules = select_distinct_shapes(store, 3)
    results = [
        validate_layer(store, module_name, seed=42 + index)
        for index, module_name in enumerate(modules)
    ]
    atomic_json(args.output, {
        "status": "complete",
        "kind": "official_qwen_awq_checkpoint_gate",
        "checkpoint": str(args.checkpoint.resolve()),
        "revision": args.revision,
        "packed_modules": len(qweight_names),
        "packing_order": [0, 2, 4, 6, 1, 3, 5, 7],
        "group_size": 128,
        "formula": "W[in,out] = (qweight_int - qzero_int) * scale",
        "validated_layers": results,
    })


if __name__ == "__main__":
    main()
