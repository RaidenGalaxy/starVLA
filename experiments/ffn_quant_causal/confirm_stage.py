from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.infinitevl_quant.config import QuantScheme
from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    file_sha256,
    json_digest,
)
from experiments.infinitevl_quant.quantization import quantized_linears
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import SCORER_ID
from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    sample_manifest_digest,
)

from .adapters import MODEL_KINDS, ModelBundle
from .budget import WallclockBudget
from .evaluation import evaluate_variant
from .statistics import stratified_paired_bootstrap


SCALE_ARTIFACT_FORMAT_VERSION = 2
SCALE_ARTIFACT_PROTOCOL = "ffn_quant_causal_scale_artifact"
SCALE_SEARCH_ALGORITHM = "matched_mbq_vvs_qig_acceptance_v1"
SCALE_METHODS = ("mbq", "qig")
SCALE_ROLES = {
    "mbq": "mbq_calibration_output",
    "qig": "post_qig_layerwise_acceptance_output",
}
SCALE_IDENTITY_SPLITS = (
    "mbq_calibration",
    "vvs",
    "qig",
    "qig_acceptance",
    "screen",
    "confirm",
)
CONFIRM_SAMPLE_COUNT = 192
CONFIRM_DATASET_COUNT = 4
CONFIRM_ROWS_PER_DATASET = 48


def _canonical_scheme(scheme: QuantScheme | Mapping[str, Any]) -> dict[str, Any]:
    value = scheme.to_dict() if isinstance(scheme, QuantScheme) else dict(scheme)
    try:
        weight_bits = value["weight_bits"]
        activation_bits = value["activation_bits"]
        group_size = value["group_size"]
    except KeyError as error:
        raise ValueError(f"incomplete quantization scheme: missing {error.args[0]}") from error
    if weight_bits not in (2, 3, 4, 8):
        raise ValueError("scale artifacts require a low-bit weight scheme")
    if activation_bits is not None:
        raise ValueError("the matched FFN confirm protocol requires A16 (no activation QDQ)")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("quantization group_size must be a positive integer")
    # JSON round-tripping provides a simple, strict rejection of tensors,
    # Paths, and other process-local objects in the identity.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclasses.dataclass(frozen=True)
class ScaleArtifactIdentity:
    """Content identity for one model/method's complete FFN scale mapping.

    Scale tensors deliberately live in a weights-only ``torch.save`` payload;
    all provenance needed to decide whether they may be applied lives in the
    adjacent JSON sidecar.  A loader must receive the expected identity and
    requires exact equality before deserializing the tensor payload.
    """

    model_kind: str
    model_digest: str
    method: str
    target_names: tuple[str, ...]
    scheme: Mapping[str, Any]
    config_digest: str
    split_digests: Mapping[str, str]
    protocol_digest: str
    implementation_digest: str
    search_identity: str
    search_audit_digest: str
    scale_role: str

    def __post_init__(self) -> None:
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(f"unsupported model kind: {self.model_kind}")
        if self.method not in SCALE_METHODS:
            raise ValueError(f"unsupported scale method: {self.method}")
        if not isinstance(self.model_digest, str) or not self.model_digest:
            raise ValueError("model_digest must be non-empty")
        if not isinstance(self.config_digest, str) or not self.config_digest:
            raise ValueError("config_digest must be non-empty")
        for field in (
            "protocol_digest", "implementation_digest", "search_identity",
            "search_audit_digest", "scale_role",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be non-empty")
        if self.scale_role != SCALE_ROLES[self.method]:
            raise ValueError(
                f"scale_role {self.scale_role!r} does not match method {self.method!r}"
            )
        names = tuple(self.target_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("target_names must contain non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("target_names contains duplicates")
        splits = dict(self.split_digests)
        missing = set(SCALE_IDENTITY_SPLITS) - set(splits)
        if missing:
            raise ValueError(f"scale identity is missing split digests: {sorted(missing)}")
        if any(not isinstance(value, str) or not value for value in splits.values()):
            raise ValueError("all split digests must be non-empty strings")
        object.__setattr__(self, "target_names", names)
        object.__setattr__(self, "scheme", _canonical_scheme(self.scheme))
        object.__setattr__(self, "split_digests", dict(sorted(splits.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": SCALE_ARTIFACT_FORMAT_VERSION,
            "protocol": SCALE_ARTIFACT_PROTOCOL,
            "model_kind": self.model_kind,
            "model_digest": self.model_digest,
            "method": self.method,
            "target_names": list(self.target_names),
            "target_names_digest": json_digest(list(self.target_names)),
            "scheme": dict(self.scheme),
            "config_digest": self.config_digest,
            "split_digests": dict(self.split_digests),
            "protocol_digest": self.protocol_digest,
            "implementation_digest": self.implementation_digest,
            "search_algorithm": SCALE_SEARCH_ALGORITHM,
            "search_identity": self.search_identity,
            "search_audit_digest": self.search_audit_digest,
            "scale_role": self.scale_role,
        }


@dataclasses.dataclass(frozen=True)
class LoadedScaleArtifact:
    path: Path
    identity: ScaleArtifactIdentity
    scales: dict[str, torch.Tensor]
    artifact_sha256: str


def build_scale_artifact_identity(
    *,
    model_kind: str,
    model_digest: str,
    method: str,
    target_names: Sequence[str],
    scheme: QuantScheme | Mapping[str, Any],
    config_digest: str,
    split_digests: Mapping[str, str],
    protocol_digest: str,
    implementation_digest: str,
    search_identity: str,
    search_audit_digest: str,
) -> ScaleArtifactIdentity:
    """Build the stable public identity shared by MBQ/QIG and confirm.

    Only protocol partitions that affect fitting, selection, or the
    within-protocol holdout are retained.  In particular, ``full256`` is
    descriptive and can never silently substitute for ``confirm``.
    """

    missing = set(SCALE_IDENTITY_SPLITS) - set(split_digests)
    if missing:
        raise ValueError(f"missing required protocol split digests: {sorted(missing)}")
    relevant_splits = {
        name: str(split_digests[name]) for name in SCALE_IDENTITY_SPLITS
    }
    return ScaleArtifactIdentity(
        model_kind=model_kind,
        model_digest=model_digest,
        method=method,
        target_names=tuple(target_names),
        scheme=_canonical_scheme(scheme),
        config_digest=config_digest,
        split_digests=relevant_splits,
        protocol_digest=protocol_digest,
        implementation_digest=implementation_digest,
        search_identity=search_identity,
        search_audit_digest=search_audit_digest,
        scale_role=SCALE_ROLES[method],
    )


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".identity.json")


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _normalized_scales(
    scales: Mapping[str, torch.Tensor], target_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    expected = set(target_names)
    if set(scales) != expected:
        missing = expected - set(scales)
        extra = set(scales) - expected
        raise ValueError(
            "scale names do not match identity targets; "
            f"missing={sorted(missing)[:4]} extra={sorted(extra)[:4]}"
        )
    result: dict[str, torch.Tensor] = {}
    for name in target_names:
        value = scales[name]
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"scale {name} must be a floating tensor")
        normalized = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if normalized.ndim != 1 or normalized.numel() == 0:
            raise ValueError(f"scale {name} must be a non-empty 1D tensor")
        if not torch.isfinite(normalized).all() or (normalized <= 0).any():
            raise ValueError(f"scale {name} contains non-positive or non-finite values")
        result[name] = normalized.clone()
    return result


def save_scale_artifact(
    path: Path,
    scales: Mapping[str, torch.Tensor],
    *,
    identity: ScaleArtifactIdentity,
) -> LoadedScaleArtifact:
    """Atomically save a weights-only scale payload and provenance sidecar.

    A fully committed existing artifact is treated as a resume point only when
    both identity and every scale tensor exactly match.  A lone payload or
    sidecar is an interrupted write and is safely replaced.
    """

    path = Path(path)
    sidecar = _sidecar_path(path)
    normalized = _normalized_scales(scales, identity.target_names)
    if path.is_file() and sidecar.is_file():
        existing = load_scale_artifact(path, expected_identity=identity)
        if any(
            not torch.equal(existing.scales[name], normalized[name])
            for name in identity.target_names
        ):
            raise RuntimeError(f"refusing to overwrite different scale artifact: {path}")
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save({
        "format_version": SCALE_ARTIFACT_FORMAT_VERSION,
        "scales": normalized,
    }, temporary)
    temporary.replace(path)
    metadata = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in normalized.items()
    }
    atomic_json(sidecar, {
        "identity": identity.to_dict(),
        "payload": {
            "artifact_sha256": file_sha256(path),
            "scales": metadata,
        },
    })
    return load_scale_artifact(path, expected_identity=identity)


def load_scale_artifact(
    path: Path,
    *,
    expected_identity: ScaleArtifactIdentity,
) -> LoadedScaleArtifact:
    """Safely load scales after exact provenance and content verification."""

    path = Path(path)
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"incomplete scale artifact: {path} / {sidecar}")
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid scale artifact sidecar: {sidecar}") from error
    if set(record) != {"identity", "payload"}:
        raise RuntimeError(f"unexpected scale sidecar schema: {sidecar}")
    if record["identity"] != expected_identity.to_dict():
        raise RuntimeError(
            "scale artifact identity mismatch for model/targets/scheme/config/"
            "splits/protocol/search provenance"
        )
    payload_metadata = record["payload"]
    if not isinstance(payload_metadata, dict) or set(payload_metadata) != {
        "artifact_sha256", "scales",
    }:
        raise RuntimeError(f"invalid scale payload metadata: {sidecar}")
    actual_sha256 = file_sha256(path)
    if payload_metadata["artifact_sha256"] != actual_sha256:
        raise RuntimeError(f"scale tensor payload checksum mismatch: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"format_version", "scales"}:
        raise RuntimeError(f"unexpected scale tensor schema: {path}")
    if payload["format_version"] != SCALE_ARTIFACT_FORMAT_VERSION:
        raise RuntimeError(f"unsupported scale artifact version: {path}")
    raw_scales = payload["scales"]
    if not isinstance(raw_scales, dict):
        raise RuntimeError(f"scale payload does not contain a mapping: {path}")
    scales = _normalized_scales(raw_scales, expected_identity.target_names)
    metadata = payload_metadata["scales"]
    if not isinstance(metadata, dict) or set(metadata) != set(expected_identity.target_names):
        raise RuntimeError(f"scale tensor metadata names mismatch: {sidecar}")
    for name in expected_identity.target_names:
        value = scales[name]
        expected = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        if metadata[name] != expected:
            raise RuntimeError(f"scale tensor metadata mismatch at {name}")
    return LoadedScaleArtifact(path, expected_identity, scales, actual_sha256)


@dataclasses.dataclass(frozen=True)
class _ModuleSignature:
    module: nn.Module
    module_type: type[nn.Module]
    weight: nn.Parameter
    weight_data_ptr: int
    weight_version: int
    bias: nn.Parameter | None
    bias_data_ptr: int | None
    bias_version: int | None


def _module_signatures(model: nn.Module, targets: Sequence[Any]) -> dict[str, _ModuleSignature]:
    result = {}
    for target in targets:
        module = model.get_submodule(target.name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"confirm target is not nn.Linear: {target.name}")
        result[target.name] = _ModuleSignature(
            module=module,
            module_type=type(module),
            weight=module.weight,
            weight_data_ptr=module.weight.data_ptr(),
            weight_version=module.weight._version,
            bias=module.bias,
            bias_data_ptr=None if module.bias is None else module.bias.data_ptr(),
            bias_version=None if module.bias is None else module.bias._version,
        )
    return result


def _assert_modules_restored(
    model: nn.Module, signatures: Mapping[str, _ModuleSignature],
) -> None:
    failures = []
    for name, before in signatures.items():
        current = model.get_submodule(name)
        restored = (
            current is before.module
            and type(current) is before.module_type
            and getattr(current, "weight", None) is before.weight
            and current.weight.data_ptr() == before.weight_data_ptr
            and current.weight._version == before.weight_version
            and getattr(current, "bias", None) is before.bias
            and (
                before.bias is None
                or (
                    current.bias.data_ptr() == before.bias_data_ptr
                    and current.bias._version == before.bias_version
                )
            )
        )
        if not restored:
            failures.append(name)
    if failures:
        raise RuntimeError(
            "quantization context did not exactly restore original FFN modules: "
            f"{failures[:8]}"
        )


def _scheme_label(scheme: QuantScheme) -> str:
    if scheme.weight_bits is None:
        raise ValueError("confirm candidate must use a low-bit scheme")
    activation = 16 if scheme.activation_bits is None else scheme.activation_bits
    return f"w{scheme.weight_bits}a{activation}"


def _validate_confirm_split(
    samples: Sequence[MultimodalSample], split_digests: Mapping[str, str],
) -> dict[str, Any]:
    if len(samples) != CONFIRM_SAMPLE_COUNT:
        raise ValueError(
            f"within-protocol holdout requires exactly {CONFIRM_SAMPLE_COUNT} samples; "
            f"got {len(samples)} (full256 is descriptive only)"
        )
    identifiers = [sample.id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("confirm contains duplicate sample IDs")
    counts = dict(sorted(Counter(sample.dataset for sample in samples).items()))
    if len(counts) != CONFIRM_DATASET_COUNT or set(counts.values()) != {
        CONFIRM_ROWS_PER_DATASET
    }:
        raise ValueError(
            "confirm must contain four datasets with 48 samples each; "
            f"got {counts}"
        )
    digest = sample_manifest_digest(list(samples))
    if split_digests.get("confirm") != digest:
        raise RuntimeError("runtime confirm samples do not match the audited confirm digest")
    return {
        "name": "confirm",
        "samples": len(samples),
        "per_dataset": counts,
        "digest": digest,
        "disjoint_from_calibration_acceptance_and_screen_within_protocol": True,
        "report_label": "within_protocol_holdout192",
        "evaluation_history": "historically_reused_evaluation",
        "historically_unseen": False,
        "full256_is_descriptive_only": True,
    }


def _validate_scale_shapes(bundle: ModelBundle, scales: Mapping[str, torch.Tensor]) -> None:
    for target in bundle.ffn_targets:
        expected = target.module.in_features
        actual = scales[target.name].numel()
        if actual != expected:
            raise ValueError(
                f"scale width mismatch at {target.name}: {actual} != {expected}"
            )


def _evaluate_quantized(
    bundle: ModelBundle,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    reference_dir: Path,
    scheme: QuantScheme,
    scales: Mapping[str, torch.Tensor] | None,
    audit_path: Path,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    signatures = _module_signatures(bundle.model, bundle.ffn_targets)
    try:
        with quantized_linears(
            bundle.model,
            bundle.ffn_targets,
            scheme,
            scales=None if scales is None else dict(scales),
        ) as quantization_audit:
            atomic_json(audit_path, quantization_audit)
            return evaluate_variant(
                bundle.model,
                bundle.processor,
                samples,
                progress=progress,
                variant=variant,
                reference_dir=reference_dir,
                write_reference=False,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                generate=generate,
                wallclock_budget=wallclock_budget,
            )
    finally:
        _assert_modules_restored(bundle.model, signatures)


def run_model_confirm(
    bundle: ModelBundle,
    confirm_samples: Sequence[MultimodalSample],
    *,
    progress: StageProgress,
    scheme: QuantScheme,
    mbq_artifact_path: Path,
    qig_artifact_path: Path,
    model_digest: str,
    config_digest: str,
    split_digests: Mapping[str, str],
    artifact_provenance: Mapping[str, str],
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    bootstrap_samples: int,
    seed: int,
    generate: bool = True,
    wallclock_budget: WallclockBudget | None = None,
) -> dict[str, Any]:
    """Run one resident model on the within-protocol holdout192 partition.

    The caller owns model loading/unloading, so InfiniteVL and Qwen are never
    resident together.  ``evaluate_variant`` supplies per-sample crash resume,
    corrected scorer-v2 generation, and FP32 reference-logit artifacts.
    """

    split = _validate_confirm_split(confirm_samples, split_digests)
    samples = list(confirm_samples)
    target_names = tuple(target.name for target in bundle.ffn_targets)
    if len(target_names) != len(set(target_names)):
        raise RuntimeError("runtime FFN target inventory contains duplicate names")
    provenance_fields = (
        "protocol_digest", "implementation_digest", "search_identity",
        "search_audit_digest",
    )
    missing_provenance = [
        field for field in provenance_fields
        if not isinstance(artifact_provenance.get(field), str)
        or not artifact_provenance.get(field)
    ]
    if missing_provenance:
        raise ValueError(
            f"scale artifact provenance is incomplete: {missing_provenance}"
        )
    identities = {
        method: build_scale_artifact_identity(
            model_kind=bundle.kind,
            model_digest=model_digest,
            method=method,
            target_names=target_names,
            scheme=scheme,
            config_digest=config_digest,
            split_digests=split_digests,
            protocol_digest=str(artifact_provenance["protocol_digest"]),
            implementation_digest=str(artifact_provenance["implementation_digest"]),
            search_identity=str(artifact_provenance["search_identity"]),
            search_audit_digest=str(artifact_provenance["search_audit_digest"]),
        )
        for method in SCALE_METHODS
    }
    artifacts = {
        "mbq": load_scale_artifact(
            mbq_artifact_path, expected_identity=identities["mbq"],
        ),
        "qig": load_scale_artifact(
            qig_artifact_path, expected_identity=identities["qig"],
        ),
    }
    for artifact in artifacts.values():
        _validate_scale_shapes(bundle, artifact.scales)

    label = _scheme_label(scheme)
    variants = {
        "bf16": f"{bundle.kind}__confirm192__bf16",
        "rtn": f"{bundle.kind}__confirm192__rtn_{label}_ffn",
        "mbq": f"{bundle.kind}__confirm192__mbq_{label}_ffn",
        "qig": f"{bundle.kind}__confirm192__qig_{label}_ffn",
    }
    reference_dir = progress.root / "reference_artifacts" / bundle.kind
    audit_dir = progress.root / "quantization_audits" / bundle.kind
    audit_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    metrics["bf16"] = evaluate_variant(
        bundle.model,
        bundle.processor,
        samples,
        progress=progress,
        variant=variants["bf16"],
        reference_dir=reference_dir,
        write_reference=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        generate=generate,
        wallclock_budget=wallclock_budget,
    )
    for method, scales in (
        ("rtn", None),
        ("mbq", artifacts["mbq"].scales),
        ("qig", artifacts["qig"].scales),
    ):
        metrics[method] = _evaluate_quantized(
            bundle,
            samples,
            progress=progress,
            variant=variants[method],
            reference_dir=reference_dir,
            scheme=scheme,
            scales=scales,
            audit_path=audit_dir / f"{method}_{label}.json",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            generate=generate,
            wallclock_budget=wallclock_budget,
        )

    expected_ids = {sample.id for sample in samples}
    rows = {name: progress.rows(variant) for name, variant in variants.items()}
    for name, values in rows.items():
        actual_ids = {row.get("id") for row in values}
        if len(values) != CONFIRM_SAMPLE_COUNT or actual_ids != expected_ids:
            raise RuntimeError(
                f"incomplete or foreign confirm rows for {name}: "
                f"rows={len(values)}, ids_match={actual_ids == expected_ids}"
            )
    comparisons = {
        method: stratified_paired_bootstrap(
            rows["bf16"], rows[method], samples=bootstrap_samples, seed=seed,
        )
        for method in ("rtn", "mbq", "qig")
    }
    result = {
        "format_version": 1,
        "model_kind": bundle.kind,
        "split": split,
        "scheme": scheme.to_dict(),
        "scope": {
            "name": "decoder_ffn_gate_up_down_only",
            "linear_count": len(bundle.ffn_targets),
            "target_names_digest": json_digest(list(target_names)),
        },
        "variants": variants,
        "metrics": metrics,
        "comparisons_to_bf16": comparisons,
        "scale_artifacts": {
            method: {
                "path": str(artifact.path),
                "sha256": artifact.artifact_sha256,
                "identity": artifact.identity.to_dict(),
            }
            for method, artifact in artifacts.items()
        },
        "scorer": SCORER_ID,
        "labels": [
            "cpu_reference_backend",
            "fake_quantized",
            "ffn_only",
            "within_protocol_holdout192",
            "historically_reused_evaluation",
            "no_speed_or_memory_claim",
        ],
    }
    atomic_json(progress.root / f"{bundle.kind}_confirm_summary.json", result)
    return result
