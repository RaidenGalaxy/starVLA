from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

import torch

from time import perf_counter

from .config import ExperimentConfig, MethodName, QuantScope
from .data import assert_disjoint, file_digest, load_manifest, normalized_question_digest
from .dataset_protocol import (
    ACTIVE_DATASET_NAMES,
    DATASET_SPECS,
    PROTOCOL_VERSION,
    docvqa_anls_score,
    finite_unit_score,
    textvqa_consensus_score,
)
from .provenance import completed_run, file_sha256, mark_complete, sha256_json, source_digest
from .reporting import atomic_json
from .runner import environment_record


VLMEVALKIT_REVISION = "4c4540cb19610b166499d004873674d00c88680f"
DATASETS = ACTIVE_DATASET_NAMES
JUDGE_DATASETS: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pinned official VLMEvalKit evaluator")
    parser.add_argument("--vlmevalkit-root", type=Path, required=True)
    parser.add_argument("--lmu-data-root", type=Path)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--official-awq-path")
    parser.add_argument("--method", choices=sorted({item.value for item in MethodName}),
                        default="bf16_value_fp32_compute")
    parser.add_argument("--scope", choices=[item.value for item in QuantScope],
                        default="decoder_canonical")
    parser.add_argument("--weight-bits", type=int, choices=(3, 4, 8))
    parser.add_argument("--activation-bits", type=int, choices=(8,))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--weight-granularity", choices=("groupwise", "per_output_channel"))
    parser.add_argument("--scheme-id")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument(
        "--experiment-config-json", type=Path,
        help="Complete ExperimentConfig JSON for a promoted local variant.",
    )
    parser.add_argument("--screen-manifest", type=Path)
    parser.add_argument("--expected-samples-per-dataset", type=int)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--preset", choices=("smoke", "timing", "screen", "confirm", "full"),
        default="screen",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-cache-dir", type=Path)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--freeze-visual-layers", default="",
        help="Comma-separated decoder layers whose visual rows are frozen during prefill.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-predictions-from", type=Path, action="append", default=[],
    )
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def validate_vlmevalkit(root: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != VLMEVALKIT_REVISION:
        raise ValueError(f"VLMEvalKit {revision} != required {VLMEVALKIT_REVISION}")
    sys.path.insert(0, str(root))


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_index(value) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


PREDICTION_COMPARABLE_FIELDS = (
    "method", "scope", "weight_bits", "activation_bits", "group_size",
    "zero_point", "weight_granularity",
    "mbq_reweight", "mbq_gradient_aggregation",
    "mbq_objective_normalization", "mbq_median_floor",
    "mbq_reconstruction_level", "mbq_algorithm_version",
    "mbq_modality_mode", "mbq_activation_mode", "mbq_objective_mode",
    "mbq_prompt_objective_scale",
    "squeezellm_sparse_fraction", "calibration_protocol",
    "freeze_visual_layers",
)


def comparable_config_value(payload: dict, key: str):
    value = payload.get(key)
    if key == "weight_granularity" and value is None:
        return "groupwise"
    if key == "mbq_modality_mode" and value is None:
        return "vision_answer"
    if key == "mbq_prompt_objective_scale" and value is None:
        return 1.0
    if key == "zero_point" and value is None:
        return (
            payload.get("weight_bits") is not None
            and payload.get("activation_bits") is None
        )
    return value


def merge_prediction_donors(
    current_rows: list[dict],
    donor_roots: list[Path],
    *,
    config: dict,
    dataset_name: str,
    expected_rows: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Merge resumable and donor predictions with strict config/hash checks."""
    merged: dict[str, dict] = {}
    for row in current_rows:
        index = canonical_index(row.get("index"))
        if index in merged:
            raise ValueError(f"duplicate generation index: {dataset_name}:{index}")
        if index not in expected_rows:
            raise ValueError(f"unexpected generation index: {dataset_name}:{index}")
        if row.get("dataset_row_sha256") != expected_rows[index]:
            raise ValueError(f"resume row hash mismatch: {dataset_name}:{index}")
        normalized = dict(row)
        normalized["index"] = index
        merged[index] = normalized

    details = []
    for root in donor_roots:
        config_path = root / "config.json"
        donor_path = root / dataset_name / "generations.partial.jsonl"
        detail = {"root": str(root), "matched": 0, "status": "missing"}
        if not config_path.is_file() or not donor_path.is_file():
            details.append(detail)
            continue
        donor_config = json.loads(config_path.read_text())
        mismatch = [
            key for key in PREDICTION_COMPARABLE_FIELDS
            if comparable_config_value(donor_config, key)
            != comparable_config_value(config, key)
        ]
        if mismatch:
            raise ValueError(
                f"prediction donor config mismatch for {mismatch}: {root}"
            )
        donor_rows = read_jsonl(donor_path)
        seen = set()
        for row in donor_rows:
            index = canonical_index(row.get("index"))
            if index in seen:
                raise ValueError(f"duplicate donor index: {root}:{dataset_name}:{index}")
            seen.add(index)
            if (
                index not in expected_rows
                or row.get("dataset_row_sha256") != expected_rows[index]
            ):
                continue
            if index in merged:
                if merged[index].get("prediction") != row.get("prediction"):
                    raise ValueError(
                        f"prediction donor conflict: {dataset_name}:{index}"
                    )
                continue
            normalized = dict(row)
            normalized["index"] = index
            normalized.setdefault("prediction_source", f"donor:{root}")
            merged[index] = normalized
            detail["matched"] += 1
        detail["status"] = "used"
        details.append(detail)
    return list(merged.values()), details


def selected_data(dataset, name: str, args):
    if args.preset == "full":
        return dataset.data.copy(), []
    if args.screen_manifest is None:
        if args.preset == "smoke":
            return dataset.data.iloc[:1].copy(), []
        raise ValueError("--screen-manifest is required for screen/confirm")
    if args.preset == "smoke":
        manifest = [item for item in load_manifest(args.screen_manifest) if item.dataset == name]
        if not manifest:
            raise ValueError(f"{name} is absent from the smoke manifest")
        wanted = canonical_index(manifest[0].official_index)
        selected = dataset.data[[canonical_index(value) == wanted for value in dataset.data["index"]]]
        if len(selected) != 1:
            raise ValueError(f"{name} smoke index {wanted} is not unique upstream")
        return selected.copy(), manifest[:1]
    manifest = [item for item in load_manifest(args.screen_manifest) if item.dataset == name]
    expected = (
        args.expected_samples_per_dataset
        if args.expected_samples_per_dataset is not None
        else 256 if args.preset == "confirm" else 64
    )
    if len(manifest) != expected:
        raise ValueError(
            f"{name} {args.preset} manifest requires exactly {expected} rows"
        )
    if args.preset == "timing":
        manifest = manifest[:32]
    positions = {
        canonical_index(value): position
        for position, value in enumerate(dataset.data["index"])
    }
    try:
        chosen = [positions[canonical_index(item.official_index)] for item in manifest]
    except KeyError as error:
        raise ValueError(f"{name} screen index absent upstream: {error}") from error
    return dataset.data.iloc[chosen].copy(), manifest


def build_prompt(model, line, dataset_name: str):
    try:
        return model.build_prompt(line, dataset=dataset_name)
    except TypeError:
        return model.build_prompt(line, dataset_name)


def verify_manifest(messages: list[dict], sample) -> None:
    prompt = "\n".join(
        str(item["value"]) for item in messages if item["type"] == "text"
    ).strip()
    if normalized_question_digest(sample.dataset, prompt) != sample.question_sha256:
        raise ValueError(f"question hash mismatch: {sample.id}")
    actual = [file_digest(Path(item["value"])) for item in messages if item["type"] == "image"]
    expected = [file_digest(path) for path in sample.images]
    if actual != expected:
        raise ValueError(f"image hash mismatch: {sample.id}")


def score_value(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        if value.strip().lower() in {"true", "correct", "yes"}:
            return 1.0
        if value.strip().lower() in {"false", "incorrect", "no"}:
            return 0.0
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_scores(
    dataset_name: str,
    result,
    directory: Path,
    indices: list[str],
    load,
) -> dict[str, float]:
    frames = [result] if hasattr(result, "columns") else []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".xlsx", ".csv", ".tsv", ".pkl"}:
            continue
        try:
            frame = load(str(path))
        except Exception:
            continue
        if hasattr(frame, "columns"):
            frames.append(frame)
    for frame in frames:
        columns = {str(column).lower(): column for column in frame.columns}
        if "index" not in columns:
            continue
        candidate = {}
        if dataset_name == "DocVQA_VAL":
            if not {"answer", "prediction"} <= set(columns):
                continue
            for _, row in frame.iterrows():
                candidate[canonical_index(row[columns["index"]])] = docvqa_anls_score(
                    row[columns["answer"]], row[columns["prediction"]],
                )
        elif dataset_name == "TextVQA_VAL":
            if not {"answer", "prediction"} <= set(columns):
                continue
            for _, row in frame.iterrows():
                candidate[canonical_index(row[columns["index"]])] = textvqa_consensus_score(
                    row[columns["answer"]], row[columns["prediction"]],
                )
        else:
            preferred = (
                ("hit",) if dataset_name in {"MMMU_DEV_VAL", "ScienceQA_TEST"} else
                ("official_score", "hit", "eval_score", "score", "anls", "correct", "match")
            )
            score_column = next(
                (columns[name] for name in preferred if name in columns), None,
            )
            if score_column is None:
                continue
            for _, row in frame.iterrows():
                score = finite_unit_score(row[score_column])
                if score is not None:
                    candidate[canonical_index(row[columns["index"]])] = score
        if all(index in candidate for index in indices):
            return {index: candidate[index] for index in indices}
    raise RuntimeError(
        f"official {dataset_name} scorer did not expose per-sample official scores"
    )


def mathvista_auxeval_fixed(model, line):
    """Pinned MathVista extraction with exactly three temperature-zero attempts."""
    from vlmeval.dataset.utils.mathvista import (
        FAIL_MSG,
        build_mathvista_gpt4_prompt,
        post_check,
    )
    prefetched = post_check(line, prefetch=True)
    if prefetched:
        return {"log": "Prefetch succeed", "res": prefetched}
    prompt = build_mathvista_gpt4_prompt(line)
    logs = []
    for attempt in range(3):
        result = model.generate(prompt, temperature=0)
        if FAIL_MSG not in result:
            return {"log": f"Succeed on attempt {attempt + 1}", "res": result}
        logs.append(f"attempt {attempt + 1} failed")
    return {"log": "; ".join(logs), "res": ""}


def install_fixed_judge_policy(dataset_name: str) -> None:
    if dataset_name == "MathVista_MINI":
        import vlmeval.dataset.utils.mathvista as mathvista
        mathvista.MathVista_auxeval = mathvista_auxeval_fixed


def scorer_hash(dataset) -> str:
    source = Path(inspect.getfile(dataset.__class__))
    payload = {
        "revision": VLMEVALKIT_REVISION,
        "class": dataset.__class__.__qualname__,
        "source_sha256": file_sha256(source),
        "fixed_judge_policy_sha256": hashlib.sha256(
            inspect.getsource(mathvista_auxeval_fixed).encode()
        ).hexdigest(),
        "score_extraction_sha256": hashlib.sha256(
            inspect.getsource(extract_scores).encode()
        ).hexdigest(),
        "protocol_version": PROTOCOL_VERSION,
    }
    if dataset.dataset_name == "MathVista_MINI":
        import vlmeval.dataset.utils.mathvista as mathvista
        payload["mathvista_scorer_sha256"] = file_sha256(
            Path(inspect.getfile(mathvista))
        )
    return sha256_json(payload)


def cache_key(dataset: str, row: dict, judge: str, prompt_hash: str) -> str:
    return sha256_json({
        "dataset": dataset,
        "index": row["index"],
        "dataset_row_sha256": row["dataset_row_sha256"],
        "prediction": row["prediction"],
        "judge_model": judge,
        "temperature": 0,
        "nproc": 4,
        "retry": 3,
        "scoring_prompt_hash": prompt_hash,
    })


def judge_kwargs(dataset: str, model: str) -> dict:
    kwargs = {"nproc": 4, "retry": 3, "temperature": 0, "verbose": True}
    if dataset in JUDGE_DATASETS:
        kwargs["model"] = model
    return kwargs


def main() -> None:
    args = parse_args()
    if args.lmu_data_root is not None:
        args.lmu_data_root.mkdir(parents=True, exist_ok=True)
        os.environ["LMUData"] = str(args.lmu_data_root.resolve())

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    freeze_visual_layers = sorted({
        int(value.strip())
        for value in args.freeze_visual_layers.split(",")
        if value.strip()
    })
    if any(index < 0 or index >= 36 for index in freeze_visual_layers):
        raise ValueError(
            f"--freeze-visual-layers contains an invalid index: {freeze_visual_layers}"
        )
    validate_vlmevalkit(args.vlmevalkit_root)
    if args.calibration_manifest and args.screen_manifest:
        assert_disjoint(
            load_manifest(args.calibration_manifest), load_manifest(args.screen_manifest),
        )
    if args.official_awq_path and args.method != "bf16_value_fp32_compute":
        raise ValueError("official AWQ cannot be combined with a local method")

    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump, load
    from .vlmeval_adapter import Qwen25VLQuantized

    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if set(datasets) - set(DATASETS):
        raise ValueError(f"datasets must be a subset of {DATASETS}")
    experiment_config = (
        ExperimentConfig.from_dict(json.loads(args.experiment_config_json.read_text()))
        if args.experiment_config_json else None
    )
    method = experiment_config.method if experiment_config else MethodName(args.method)
    scope = experiment_config.scope if experiment_config else QuantScope(args.scope)
    if experiment_config:
        args.weight_bits = experiment_config.weight_bits
        args.activation_bits = experiment_config.activation_bits
        args.group_size = experiment_config.group_size
        args.max_new_tokens = experiment_config.max_new_tokens
        args.seed = experiment_config.seed
    weight_granularity = args.weight_granularity or (
        experiment_config.weight_granularity if experiment_config else
        "per_output_channel" if method == MethodName.MBQ and args.activation_bits == 8
        else "groupwise"
    )
    scheme = (experiment_config.scheme_id if experiment_config else args.scheme_id) or (
        "official_w4a16_g128_asym"
        if args.official_awq_path
        else "bf16_value_fp32_compute"
        if method == MethodName.BF16_VALUE_FP32_COMPUTE
        else f"{method.value}_w{args.weight_bits or 4}a{args.activation_bits or 16}_g{args.group_size}"
    )
    config = (experiment_config.to_dict() if experiment_config else {}) | {
        "method": "official_qwen_awq" if args.official_awq_path else method.value,
        "scheme_id": scheme,
        "scope": scope.value,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "group_size": args.group_size,
        "weight_granularity": weight_granularity,
        "zero_point": (
            experiment_config.zero_point if experiment_config else
            args.weight_bits is not None and args.activation_bits is None
        ),
        "preset": args.preset,
        "expected_samples_per_dataset": args.expected_samples_per_dataset,
        "datasets": datasets,
        "seed": args.seed,
        "min_pixels": experiment_config.min_pixels if experiment_config else 256 * 28 * 28,
        "max_pixels": experiment_config.max_pixels if experiment_config else 1280 * 28 * 28,
        "vlmevalkit_revision": VLMEVALKIT_REVISION,
        "judge": None,
        "protocol_version": PROTOCOL_VERSION,
        "freeze_visual_layers": freeze_visual_layers,
        "visual_freeze_definition": (
            "replace selected visual-token layer outputs with layer inputs during "
            "prefill; leave cached one-token decoding unchanged"
        ),
    }
    config_digest = sha256_json(config)
    digests = {
        "base_model": source_digest(args.base_model_path),
        "official_awq": source_digest(args.official_awq_path) if args.official_awq_path else "",
        "calibration_manifest": (
            file_sha256(args.calibration_manifest) if args.calibration_manifest else ""
        ),
        "screen_manifest": file_sha256(args.screen_manifest) if args.screen_manifest else "",
        "experiment_config": (
            file_sha256(args.experiment_config_json)
            if args.experiment_config_json else ""
        ),
        "vlmevalkit_revision": VLMEVALKIT_REVISION,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = ("official_result.json",)
    if args.resume and completed_run(
        args.output_dir, config_digest=config_digest, input_digests=digests,
        required_artifacts=required,
    ):
        return
    atomic_json(args.output_dir / "config.json", config | {
        "config_hash": config_digest,
        "input_digests": digests,
        "environment": environment_record(),
    })
    judge_cache = args.judge_cache_dir or args.output_dir.parent / "_judge_cache"
    prepared, generation_pending = {}, False
    for dataset_name in datasets:
        destination = args.output_dir / dataset_name
        destination.mkdir(parents=True, exist_ok=True)
        dataset = build_dataset(dataset_name)
        data, manifest = selected_data(dataset, dataset_name, args)
        dataset.data = data
        progress_path = destination / "generations.partial.jsonl"
        progress_meta = {
            "config_digest": config_digest,
            "input_digests": digests,
            "indices": [canonical_index(value) for value in data["index"]],
        }
        if args.resume and progress_path.is_file():
            progress_meta_path = destination / "progress.json"
            if not progress_meta_path.is_file():
                raise ValueError(f"generation progress has no metadata: {dataset_name}")
            saved_meta = json.loads(progress_meta_path.read_text())
            if saved_meta != progress_meta:
                raise ValueError(f"progress digest mismatch: {dataset_name}")
            generated = read_jsonl(progress_path)
        else:
            generated = []
        expected_rows = {
            canonical_index(line["index"]): sha256_json({
                str(key): str(value) for key, value in line.to_dict().items()
            })
            for _, line in data.iterrows()
        }
        resumed_count = len(generated)
        generated, donor_details = merge_prediction_donors(
            generated,
            args.reuse_predictions_from,
            config=config,
            dataset_name=dataset_name,
            expected_rows=expected_rows,
        )
        by_index = {row["index"]: row for row in generated}
        expected_indices = set(progress_meta["indices"])
        if set(by_index) - expected_indices:
            raise ValueError(f"unexpected generation index: {dataset_name}")
        if expected_indices - set(by_index):
            generation_pending = True
        prepared[dataset_name] = {
            "destination": destination,
            "reuse": {
                "expected": len(expected_indices),
                "resumed": resumed_count,
                "donors": donor_details,
                "reused_total": len(by_index),
                "pending_after_reuse": len(expected_indices - set(by_index)),
            },
            "dataset": dataset,
            "data": data,
            "manifest": manifest,
            "progress_path": progress_path,
            "progress_meta": progress_meta,
            "generated": generated,
            "by_index": by_index,
        }

    atomic_json(args.output_dir / "prediction_reuse.json", {
        "status": "prepared",
        "datasets": {
            name: state["reuse"] for name, state in prepared.items()
        },
    })
    audit_path = args.output_dir / "quantization_audit.json"
    if generation_pending:
        model = Qwen25VLQuantized(
            model_path=args.base_model_path,
            method=method.value,
            scope=scope.value,
            calibration_manifest=(
                str(args.calibration_manifest) if args.calibration_manifest else None
            ),
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
            seed=args.seed,
            strict_qwen_3b=not args.allow_nonstandard_model,
            official_awq_path=args.official_awq_path,
            calibration_cache_dir=str(
                args.calibration_cache_dir
                or args.output_dir.parent / "_calibration_cache"
            ),
            model_revision_digest=digests["base_model"],
            calibration_samples=2 if args.preset == "smoke" else 128,
            omniquant_steps=2 if args.preset == "smoke" else 20,
            scheme_id=scheme,
            weight_granularity=weight_granularity,
            experiment_config=(
                experiment_config.to_dict() if experiment_config else None
            ),
            freeze_visual_layers=freeze_visual_layers,
        )
        audit = (
            model.official_awq_audit if args.official_awq_path else model.applied.audit
        )
        atomic_json(audit_path, audit)
    else:
        model = None
        if not audit_path.is_file():
            raise ValueError("complete generation progress has no quantization audit")
        audit = json.loads(audit_path.read_text())

    summaries, combined, awaiting = {}, [], False
    try:
        for dataset_name in datasets:
            dataset_started = perf_counter()
            state = prepared[dataset_name]
            destination = state["destination"]
            dataset = state["dataset"]
            data = state["data"]
            manifest = state["manifest"]
            progress_path = state["progress_path"]
            progress_meta = state["progress_meta"]
            generated = state["generated"]
            by_index = state["by_index"]
            if model is not None:
                model.set_dump_image(dataset.dump_image)
            for position, (_, line) in enumerate(data.iterrows()):
                index = canonical_index(line["index"])
                if index in by_index:
                    continue
                messages = build_prompt(model, line, dataset_name)
                if manifest:
                    verify_manifest(messages, manifest[position])
                row = {
                    "id": manifest[position].id if manifest else f"{dataset_name}:{index}",
                    "dataset": dataset_name,
                    "index": index,
                    "dataset_row_sha256": sha256_json({
                        str(key): str(value) for key, value in line.to_dict().items()
                    }),
                    "prediction": "",
                    "prediction_source": "generated_v4",
                }
                generation_started = perf_counter()
                row["prediction"] = model.generate_inner(messages, dataset=dataset_name)
                row["generation_seconds"] = perf_counter() - generation_started
                generated.append(row)
                by_index[index] = row
                atomic_jsonl(progress_path, generated)
                atomic_json(destination / "progress.json", progress_meta)
            ordered = [by_index[canonical_index(value)] for value in data["index"]]
            predictions = data.copy()
            predictions["prediction"] = [row["prediction"] for row in ordered]
            prediction_file = destination / "predictions.xlsx"
            temporary_prediction = destination / "predictions.tmp.xlsx"
            dump(predictions, str(temporary_prediction))
            temporary_prediction.replace(prediction_file)

            if dataset_name in JUDGE_DATASETS and not os.environ.get("OPENAI_API_KEY"):
                atomic_json(destination / "status.json", {
                    "status": "awaiting_judge",
                    "required_environment": ["OPENAI_API_KEY"],
                    "judge": config["judge"],
                    "prediction_sha256": file_sha256(prediction_file),
                })
                awaiting = True
                continue

            prompt_hash = scorer_hash(dataset)
            judge = args.judge_model if dataset_name in JUDGE_DATASETS else "exact_matching"
            keys = [cache_key(dataset_name, row, judge, prompt_hash) for row in ordered]
            cached = {}
            for row, key in zip(ordered, keys):
                path = judge_cache / key / "score.json"
                if path.is_file():
                    cached[row["index"]] = float(json.loads(path.read_text())["official_score"])
            if len(cached) == len(ordered):
                scores, cache_hit = cached, True
            else:
                install_fixed_judge_policy(dataset_name)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = dataset.evaluate(
                        str(prediction_file), **judge_kwargs(dataset_name, args.judge_model),
                    )
                fallback = [
                    str(item.message) for item in caught
                    if "exact matching" in str(item.message).lower()
                    or "api is not working" in str(item.message).lower()
                ]
                if dataset_name in JUDGE_DATASETS and fallback:
                    raise RuntimeError(
                        f"judge API failed; refusing exact-match fallback: {fallback}"
                    )
                scores = extract_scores(
                    dataset_name,
                    result,
                    destination,
                    [row["index"] for row in ordered],
                    load,
                )
                cache_hit = False
                for row, key in zip(ordered, keys):
                    atomic_json(judge_cache / key / "score.json", {
                        "dataset": dataset_name,
                        "index": row["index"],
                        "prediction_sha256": hashlib.sha256(
                            row["prediction"].encode()
                        ).hexdigest(),
                        "official_score": scores[row["index"]],
                        "cache_key": key,
                    })
            scored = [
                row | {
                    "official_score": scores[row["index"]],
                    **(
                        {"anls_score": scores[row["index"]]}
                        if dataset_name == "DocVQA_VAL" else {}
                    ),
                    "judge_cache_key": key,
                    "scoring_prompt_hash": prompt_hash,
                }
                for row, key in zip(ordered, keys)
            ]
            sample_path = destination / "official_samples.jsonl"
            atomic_jsonl(sample_path, scored)
            score_payload = {
                "status": "complete",
                "dataset": dataset_name,
                "samples": len(scored),
                "mean_official_score": sum(scores.values()) / len(scores),
                "score": 100.0 * sum(scores.values()) / len(scores),
                "metric": DATASET_SPECS[dataset_name].metric,
                "scale": "0_100",
                "max_score": DATASET_SPECS[dataset_name].max_score,
                "higher_is_better": DATASET_SPECS[dataset_name].higher_is_better,
                "cache_hit": cache_hit,
                "elapsed_seconds": perf_counter() - dataset_started,
                "generation_seconds_p95": sorted(
                    row.get("generation_seconds", 0.0) for row in ordered
                )[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            }
            atomic_json(destination / "official_score.json", score_payload)
            atomic_json(destination / "status.json", {
                "status": "complete",
                "prediction_sha256": file_sha256(prediction_file),
                "scored_samples_sha256": file_sha256(sample_path),
            })
            summaries[dataset_name] = score_payload
            combined.extend(scored)

        reuse_payload = {
            "status": "complete",
            "datasets": {},
        }
        for name, state in prepared.items():
            source_counts = {}
            for row in state["by_index"].values():
                source = row.get("prediction_source", "resume_unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            reuse_payload["datasets"][name] = state["reuse"] | {
                "completed": len(state["by_index"]),
                "prediction_sources": source_counts,
            }
        atomic_json(args.output_dir / "prediction_reuse.json", reuse_payload)
        payload = {
            "run_id": f"{config['method']}__{scheme}__{scope.value}__{config_digest[:12]}",
            "method": config["method"],
            "scheme_id": scheme,
            "scope": scope.value,
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "weight_granularity": weight_granularity,
            "implementation_tier": (
                "official_anchor" if args.official_awq_path else
                "reference" if method == MethodName.BF16_VALUE_FP32_COMPUTE else
                "naive_baseline" if method in {MethodName.RTN, MethodName.VECTORWISE_W8} else
                "cpu_research_approximation" if method in {
                    MethodName.OMNIQUANT, MethodName.SPQR, MethodName.SQUEEZELLM
                } else "local_algorithm_simulation"
            ),
            "freeze_visual_layers": freeze_visual_layers,
            "logical_bits_per_weight": (
                4.15625 if args.official_awq_path else
                16.0 if method == MethodName.BF16_VALUE_FP32_COMPUTE else
                audit.get("logical_bits_per_weight", args.weight_bits or 4)
            ),
            "datasets": summaries,
            "samples": combined,
            "status": "awaiting_judge" if awaiting else "complete",
        }
        atomic_json(args.output_dir / "official_result.json", payload)
        if not awaiting:
            mark_complete(
                args.output_dir, config_digest=config_digest, input_digests=digests,
                required_artifacts=required,
            )
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "config_hash": config_digest,
            "input_digests": digests,
        })
        raise


if __name__ == "__main__":
    main()
