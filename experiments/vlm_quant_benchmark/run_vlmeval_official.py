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

from .config import MethodName, QuantScope
from .data import assert_disjoint, file_digest, load_manifest, normalized_question_digest
from .provenance import completed_run, file_sha256, mark_complete, sha256_json, source_digest
from .reporting import atomic_json
from .runner import environment_record


VLMEVALKIT_REVISION = "4c4540cb19610b166499d004873674d00c88680f"
DATASETS = ("DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN")
JUDGE_DATASETS = {"MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN"}


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
    parser.add_argument("--weight-bits", type=int, choices=(4, 8))
    parser.add_argument("--activation-bits", type=int, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scheme-id")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--screen-manifest", type=Path)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--preset", choices=("smoke", "timing", "screen", "full"), default="screen")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-cache-dir", type=Path)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
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


def selected_data(dataset, name: str, args):
    if args.preset == "full":
        return dataset.data.copy(), []
    if args.preset == "smoke":
        return dataset.data.iloc[:1].copy(), []
    if args.screen_manifest is None:
        raise ValueError("--screen-manifest is required for screen")
    manifest = [item for item in load_manifest(args.screen_manifest) if item.dataset == name]
    if len(manifest) != 64:
        raise ValueError(f"{name} fixed manifest requires exactly 64 rows")
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
        if dataset_name == "MathVista_MINI" and {"res", "log"} <= set(columns):
            from vlmeval.dataset.utils.mathvista import post_check
            for _, row in frame.iterrows():
                candidate[canonical_index(row[columns["index"]])] = float(
                    bool(post_check(row, prefetch=False))
                )
        else:
            preferred = (
                ("eval_score",) if dataset_name == "DocVQA_VAL" else
                ("hit",) if dataset_name in {"MMMU_DEV_VAL", "MMBench_DEV_EN"} else
                ("official_score", "hit", "eval_score", "score", "anls", "correct", "match")
            )
            score_column = next(
                (columns[name] for name in preferred if name in columns), None,
            )
            if score_column is None:
                continue
            for _, row in frame.iterrows():
                score = score_value(row[score_column])
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
    method, scope = MethodName(args.method), QuantScope(args.scope)
    scheme = args.scheme_id or (
        "official_w4a16_g128_asym"
        if args.official_awq_path
        else "bf16_value_fp32_compute"
        if method == MethodName.BF16_VALUE_FP32_COMPUTE
        else f"{method.value}_w{args.weight_bits or 4}a{args.activation_bits or 16}_g{args.group_size}"
    )
    config = {
        "method": "official_qwen_awq" if args.official_awq_path else method.value,
        "scheme_id": scheme,
        "scope": scope.value,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "group_size": args.group_size,
        "preset": args.preset,
        "datasets": datasets,
        "seed": args.seed,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "vlmevalkit_revision": VLMEVALKIT_REVISION,
        "judge": {"model": args.judge_model, "temperature": 0, "nproc": 4, "retry": 3},
    }
    config_digest = sha256_json(config)
    digests = {
        "base_model": source_digest(args.base_model_path),
        "official_awq": source_digest(args.official_awq_path) if args.official_awq_path else "",
        "calibration_manifest": (
            file_sha256(args.calibration_manifest) if args.calibration_manifest else ""
        ),
        "screen_manifest": file_sha256(args.screen_manifest) if args.screen_manifest else "",
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
        by_index = {row["index"]: row for row in generated}
        if len(by_index) != len(generated):
            raise ValueError(f"duplicate generation index: {dataset_name}")
        expected_indices = set(progress_meta["indices"])
        if set(by_index) - expected_indices:
            raise ValueError(f"unexpected generation index: {dataset_name}")
        if expected_indices - set(by_index):
            generation_pending = True
        prepared[dataset_name] = {
            "destination": destination,
            "dataset": dataset,
            "data": data,
            "manifest": manifest,
            "progress_path": progress_path,
            "progress_meta": progress_meta,
            "generated": generated,
            "by_index": by_index,
        }

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
            calibration_cache_dir=str(args.output_dir.parent / "_calibration_cache"),
            calibration_samples=2 if args.preset == "smoke" else 128,
            scheme_id=scheme,
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

        payload = {
            "run_id": f"{config['method']}__{scheme}__{scope.value}__{config_digest[:12]}",
            "method": config["method"],
            "scheme_id": scheme,
            "scope": scope.value,
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "implementation_tier": (
                "official_anchor" if args.official_awq_path else
                "reference" if method == MethodName.BF16_VALUE_FP32_COMPUTE else
                "naive_baseline" if method in {MethodName.RTN, MethodName.VECTORWISE_W8} else
                "cpu_research_approximation" if method in {
                    MethodName.OMNIQUANT, MethodName.SPQR, MethodName.SQUEEZELLM
                } else "local_algorithm_simulation"
            ),
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
