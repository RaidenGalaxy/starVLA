from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from experiments.vlm_quant_benchmark.data import load_manifest
from experiments.vlm_quant_benchmark.mbq import mbq_calibration_batches
from experiments.vlm_quant_benchmark.provenance import source_digest
from experiments.vlm_quant_benchmark.runner import load_reference
from experiments.vlm_quant_benchmark.selection import select_linears

from .core import (
    TOKEN_GROUPS,
    append_jsonl_fsync,
    applied_mbq,
    atomic_json,
    compare_teacher,
    decoder_layers,
    file_sha256,
    layer_set_id,
    load_existing_rows,
    load_validated_mbq,
    mbq_config,
    prepare_teacher_input,
    result_key,
    stable_json_hash,
    teacher_reference,
    temporary_token_group_freeze,
)
from .probe import DEFAULT_MBQ_ARTIFACT, DEFAULT_MBQ_AUDIT, DEFAULT_MODEL


SCHEMA_NAME = "qshortv_visual_freeze_v1"
SCHEMA_VERSION = 1
DEFAULT_MBQ_W3_ARTIFACT = Path(
    "outputs/vlm_quant_benchmark/top4_protocol_v3/vlm_screen/_calibration_cache/"
    "mbq_e9581657793fb14a0394b714c8c2f190b36ac3bfa3fde665b4a5e9614b0db136.pt"
)
DEFAULT_MBQ_W3_AUDIT = Path(
    "outputs/vlm_quant_benchmark/top4_protocol_v3/vlm_screen/"
    "mbq_w3a16_g128_asym/quantization_audit.json"
)


def _parse_layer_sets(layers: str, layer_sets: str | None) -> list[tuple[int, ...]]:
    if layer_sets:
        parsed = []
        for raw_set in layer_sets.split(";"):
            values = tuple(sorted({
                int(value.strip()) for value in raw_set.split("+") if value.strip()
            }))
            if not values:
                raise ValueError("--layer-sets cannot contain an empty set")
            parsed.append(values)
    elif layers.strip().lower() == "all":
        parsed = [(index,) for index in range(36)]
    else:
        parsed = [(int(value.strip()),) for value in layers.split(",") if value.strip()]
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("layer sets must be non-empty and unique")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable FP/MBQ-W3/W4 ShortV visual-freeze probes",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,35")
    parser.add_argument(
        "--layer-sets",
        help="Optional semicolon-separated cumulative sets, for example '1;1+4;1+4+7'",
    )
    parser.add_argument("--quant-schemes", default="fp,mbq_w4")
    parser.add_argument("--mbq-artifact", type=Path, default=DEFAULT_MBQ_ARTIFACT)
    parser.add_argument("--mbq-audit", type=Path, default=DEFAULT_MBQ_AUDIT)
    parser.add_argument(
        "--mbq-w3-artifact", type=Path, default=DEFAULT_MBQ_W3_ARTIFACT,
    )
    parser.add_argument("--mbq-w3-audit", type=Path, default=DEFAULT_MBQ_W3_AUDIT)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _baseline_metrics(reference) -> dict:
    return {
        "target_nll": reference.target_nll,
        "delta_target_nll": 0.0,
        "answer_logit_kl": 0.0,
        "final_hidden_cosine_drift": {name: 0.0 for name in TOKEN_GROUPS},
        "token_counts": dict(reference.token_counts),
    }


def _run_state(
    *,
    model,
    processor,
    samples,
    layers,
    layer_sets,
    quant_scheme: str,
    output: Path,
    completed: set[tuple],
    config_hash: str,
) -> int:
    written = 0
    for sample_position, sample in enumerate(samples, 1):
        requested = [
            (sample.id, quant_scheme, "baseline", "baseline"),
            *[
                (sample.id, quant_scheme, layer_set_id(indices), "freeze_visual")
                for indices in layer_sets
            ],
        ]
        if all(key in completed for key in requested):
            continue
        batch = next(mbq_calibration_batches(
            processor,
            [sample],
            min_pixels=mbq_config(4).min_pixels,
            max_pixels=mbq_config(4).max_pixels,
        ))
        prepared = prepare_teacher_input(model, batch)
        started = time.perf_counter()
        reference = teacher_reference(model, batch, prepared=prepared)
        baseline_key = requested[0]
        if baseline_key not in completed:
            row = {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "config_hash": config_hash,
                "sample_id": sample.id,
                "dataset": sample.dataset,
                "quant_scheme": quant_scheme,
                "layer_set_id": "baseline",
                "layer_indices": [],
                "operation": "baseline",
                "metrics": _baseline_metrics(reference),
                "forward_seconds": time.perf_counter() - started,
                "sample_position": sample_position,
            }
            append_jsonl_fsync(output, row)
            completed.add(result_key(row))
            written += 1
        for indices in layer_sets:
            key = (sample.id, quant_scheme, layer_set_id(indices), "freeze_visual")
            if key in completed:
                continue
            started = time.perf_counter()
            with temporary_token_group_freeze(
                layers, indices, prepared.masks["vision"],
            ):
                candidate = teacher_reference(model, batch, prepared=prepared)
            row = {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "config_hash": config_hash,
                "sample_id": sample.id,
                "dataset": sample.dataset,
                "quant_scheme": quant_scheme,
                "layer_set_id": layer_set_id(indices),
                "layer_indices": list(indices),
                "operation": "freeze_visual",
                "metrics": compare_teacher(candidate, reference),
                "forward_seconds": time.perf_counter() - started,
                "sample_position": sample_position,
            }
            append_jsonl_fsync(output, row)
            completed.add(result_key(row))
            written += 1
            del candidate
        del reference, prepared, batch
        gc.collect()
        print(
            f"[{quant_scheme}] {sample_position}/{len(samples)} {sample.id} "
            f"rows={written}",
            flush=True,
        )
    return written


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    layer_sets = _parse_layer_sets(args.layers, args.layer_sets)
    schemes = tuple(value.strip() for value in args.quant_schemes.split(",") if value.strip())
    if not schemes or set(schemes) - {"fp", "mbq_w3", "mbq_w4"}:
        raise ValueError("--quant-schemes supports fp, mbq_w3, and mbq_w4")
    samples = load_manifest(args.manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("probe manifest is empty")

    model_digest = source_digest(args.model_path)
    artifact_digest = file_sha256(args.mbq_artifact) if "mbq_w4" in schemes else ""
    audit_digest = file_sha256(args.mbq_audit) if "mbq_w4" in schemes else ""
    config = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "model_path": str(Path(args.model_path).resolve()),
        "model_digest": model_digest,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "sample_ids": [sample.id for sample in samples],
        "layer_sets": [list(indices) for indices in layer_sets],
        "quant_schemes": list(schemes),
        "mbq_artifact_sha256": artifact_digest,
        "mbq_audit_sha256": audit_digest,
        "threads": args.threads,
        "token_groups": list(TOKEN_GROUPS),
        "intervention": {
            "name": "freeze_visual",
            "definition": "replace selected visual-token layer outputs with layer inputs",
            "claims_real_speedup": False,
        },
        "metric_definition": {
            "visual_lc": "KL(base-state logits || visual-freeze logits) at answer predictors",
            "target_nll": "teacher-forced causal language-model loss",
            "hidden_drift": "mean per-token 1-cosine at final hidden state",
        },
        "forward_backend": "cached_vision_embedding_selective_lm_head_v2",
    }
    if "mbq_w3" in schemes:
        config["mbq_w3_artifact_sha256"] = file_sha256(args.mbq_w3_artifact)
        config["mbq_w3_audit_sha256"] = file_sha256(args.mbq_w3_audit)
    config_hash = stable_json_hash(config)
    config["config_hash"] = config_hash
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") != config_hash:
            raise ValueError(f"existing probe config does not match: {config_path}")
    else:
        atomic_json(config_path, config)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume: {args.output}")
    _rows, completed = load_existing_rows(args.output, config_hash=config_hash)
    expected_rows = len(samples) * len(schemes) * (1 + len(layer_sets))
    if len(completed) == expected_rows:
        atomic_json(args.output.with_suffix(args.output.suffix + ".complete.json"), {
            "config_hash": config_hash,
            "expected_rows": expected_rows,
            "actual_rows": len(completed),
            "new_rows": 0,
            "complete": True,
        })
        return

    def state_pending(state: str) -> bool:
        return any(
            (sample.id, state, set_id, operation) not in completed
            for sample in samples
            for set_id, operation in [
                ("baseline", "baseline"),
                *[(layer_set_id(indices), "freeze_visual") for indices in layer_sets],
            ]
        )

    model, processor, conversion = load_reference(args.model_path)
    layers = decoder_layers(model)
    if len(layers) != 36:
        raise RuntimeError(f"expected 36 decoder layers, found {len(layers)}")
    for indices in layer_sets:
        if any(index < 0 or index >= len(layers) for index in indices):
            raise IndexError(f"invalid layer set: {indices}")
    selected = select_linears(model, mbq_config(4).scope, strict_qwen_3b=True)
    provenance = {"reference_conversion": conversion}
    searches = {}
    if "mbq_w4" in schemes:
        searches["mbq_w4"], mbq_provenance = load_validated_mbq(
            args.mbq_artifact, args.mbq_audit, selected, weight_bits=4,
        )
        provenance["mbq_w4"] = mbq_provenance
    if "mbq_w3" in schemes:
        searches["mbq_w3"], mbq_provenance = load_validated_mbq(
            args.mbq_w3_artifact, args.mbq_w3_audit, selected, weight_bits=3,
        )
        provenance["mbq_w3"] = mbq_provenance
    atomic_json(args.output.with_suffix(args.output.suffix + ".provenance.json"), provenance)

    total_written = 0
    if "fp" in schemes and state_pending("fp"):
        total_written += _run_state(
            model=model,
            processor=processor,
            samples=samples,
            layers=layers,
            layer_sets=layer_sets,
            quant_scheme="fp",
            output=args.output,
            completed=completed,
            config_hash=config_hash,
        )
    for quant_scheme, weight_bits in (("mbq_w4", 4), ("mbq_w3", 3)):
        if quant_scheme not in schemes or not state_pending(quant_scheme):
            continue
        with applied_mbq(
            model, selected, searches[quant_scheme], weight_bits=weight_bits,
        ) as quant_audit:
            if quant_audit.get("weight_bits") != weight_bits:
                raise ValueError(
                    f"applied MBQ audit does not report W{weight_bits}"
                )
            total_written += _run_state(
                model=model,
                processor=processor,
                samples=samples,
                layers=layers,
                layer_sets=layer_sets,
                quant_scheme=quant_scheme,
                output=args.output,
                completed=completed,
                config_hash=config_hash,
            )
    atomic_json(args.output.with_suffix(args.output.suffix + ".complete.json"), {
        "config_hash": config_hash,
        "expected_rows": expected_rows,
        "actual_rows": len(completed),
        "new_rows": total_written,
        "complete": len(completed) == expected_rows,
    })


if __name__ == "__main__":
    main()
