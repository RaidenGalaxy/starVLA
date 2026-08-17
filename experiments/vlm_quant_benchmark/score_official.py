from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from .provenance import file_sha256, sha256_json
from .reporting import atomic_json


VLMEVALKIT_REVISION = "4c4540cb19610b166499d004873674d00c88680f"
JUDGE_DATASETS = {"MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN"}


def _jsonable(value):
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return {
                "kind": "dataframe",
                "columns": list(value.columns),
                "records": value.to_dict(orient="records"),
            }
        if isinstance(value, pd.Series):
            return {"kind": "series", "values": value.to_dict()}
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def judge_cache_key(
    *,
    dataset: str,
    prediction_file: Path,
    judge_model: str,
    scorer_revision: str = VLMEVALKIT_REVISION,
) -> str:
    return sha256_json({
        "dataset": dataset,
        "prediction_sha256": file_sha256(prediction_file),
        "judge_model": judge_model,
        "temperature": 0,
        "nproc": 4,
        "retry": 3,
        "scorer_revision": scorer_revision,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pinned VLMEvalKit official scorer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = judge_cache_key(
        dataset=args.dataset,
        prediction_file=args.prediction_file,
        judge_model=args.judge_model,
    )
    cache_path = args.cache_dir / key / "official_score.json"
    if cache_path.is_file():
        atomic_json(args.output_dir / "official_score.json", json.loads(cache_path.read_text()))
        atomic_json(args.output_dir / "scoring_status.json", {
            "status": "complete", "cache_hit": True, "cache_key": key,
        })
        return

    needs_judge = args.dataset in JUDGE_DATASETS
    if needs_judge and not os.environ.get("OPENAI_API_KEY"):
        atomic_json(args.output_dir / "scoring_status.json", {
            "status": "awaiting_judge",
            "dataset": args.dataset,
            "judge_model": args.judge_model,
            "required_environment": ["OPENAI_API_KEY"],
            "cache_key": key,
        })
        return

    try:
        from vlmeval.dataset import build_dataset
    except ImportError as error:
        raise ImportError(
            f"VLMEvalKit at {VLMEVALKIT_REVISION} must be available on PYTHONPATH"
        ) from error

    judge_kwargs = {
        "nproc": 4,
        "verbose": True,
        "retry": 3,
        "temperature": 0,
    }
    if needs_judge:
        judge_kwargs["model"] = args.judge_model

    try:
        dataset = build_dataset(args.dataset)
        result = dataset.evaluate(str(args.prediction_file), **judge_kwargs)
        payload = {
            "status": "complete",
            "dataset": args.dataset,
            "prediction_sha256": file_sha256(args.prediction_file),
            "vlmevalkit_revision": VLMEVALKIT_REVISION,
            "judge": judge_kwargs,
            "result": _jsonable(result),
            "cache_key": key,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(cache_path, payload)
        atomic_json(args.output_dir / "official_score.json", payload)
        atomic_json(args.output_dir / "scoring_status.json", {
            "status": "complete", "cache_hit": False, "cache_key": key,
        })
    except Exception as error:
        atomic_json(args.output_dir / "scoring_status.json", {
            "status": "failed_no_fallback",
            "dataset": args.dataset,
            "judge_model": args.judge_model if needs_judge else None,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "cache_key": key,
        })
        raise


if __name__ == "__main__":
    main()
