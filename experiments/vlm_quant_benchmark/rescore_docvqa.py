"""Repair legacy DocVQA results that stored normalized edit distance as score."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .dataset_protocol import PROTOCOL_VERSION, docvqa_anls_score
from .reporting import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score existing DocVQA predictions with ANLS")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def rescore_prediction_file(path: Path, apply: bool) -> dict:
    frame = pd.read_excel(path)
    required = {"index", "answer", "prediction"}
    if not required <= set(frame.columns):
        raise ValueError(f"{path} lacks columns {sorted(required - set(frame.columns))}")
    scores = [
        docvqa_anls_score(row.answer, row.prediction)
        for row in frame.itertuples(index=False)
    ]
    score = 100.0 * sum(scores) / len(scores)
    result = {"path": str(path), "samples": len(scores), "score": score}
    if not apply:
        return result

    directory = path.parent
    score_path = directory / "official_score.json"
    legacy_path = directory / "official_score.legacy_raw_distance.json"
    old_payload = json.loads(score_path.read_text()) if score_path.is_file() else {}
    if score_path.is_file() and not legacy_path.exists():
        shutil.copy2(score_path, legacy_path)
    payload = old_payload | {
        "status": "complete",
        "dataset": "DocVQA_VAL",
        "samples": len(scores),
        "mean_official_score": sum(scores) / len(scores),
        "score": score,
        "metric": "ANLS",
        "scale": "0_100",
        "max_score": 100.0,
        "higher_is_better": True,
        "protocol_version": PROTOCOL_VERSION,
        "legacy_raw_distance_mean": old_payload.get("mean_official_score"),
        "rescored_from_predictions": True,
    }
    atomic_json(score_path, payload)

    sample_path = directory / "official_samples.jsonl"
    if sample_path.is_file():
        rows = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
        by_index = {str(row["index"]): value for row, value in zip(frame.to_dict("records"), scores)}
        repaired = []
        for row in rows:
            value = by_index[str(row["index"])]
            repaired.append(row | {"official_score": value, "anls_score": value})
        atomic_jsonl(sample_path, repaired)

    run_dir = directory.parent
    result_path = run_dir / "official_result.json"
    if result_path.is_file():
        aggregate = json.loads(result_path.read_text())
        aggregate.setdefault("datasets", {})["DocVQA_VAL"] = payload
        for row in aggregate.get("samples", []):
            if row.get("dataset") == "DocVQA_VAL":
                value = by_index.get(str(row.get("index"))) if sample_path.is_file() else None
                if value is not None:
                    row["official_score"] = value
                    row["anls_score"] = value
        aggregate["protocol_version"] = PROTOCOL_VERSION
        atomic_json(result_path, aggregate)
        atomic_json(run_dir / "legacy_protocol_notice.json", {
            "status": "legacy_old_4dataset_protocol",
            "active_ranking": False,
            "reason": "MathVista/MMBench removed; DocVQA raw-distance score repaired",
            "docvqa_protocol_version": PROTOCOL_VERSION,
        })
    return result


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.glob("**/DocVQA_VAL/predictions.xlsx"))
    results = [rescore_prediction_file(path, args.apply) for path in paths]
    if args.apply:
        notice = {
            "status": "legacy_old_4dataset_protocol",
            "active_ranking": False,
            "reason": (
                "MathVista/MMBench were retired and legacy DocVQA eval_score stored "
                "normalized edit distance instead of ANLS"
            ),
            "replacement_protocol_version": PROTOCOL_VERSION,
        }
        report_dirs = {path.parent for path in args.root.glob("**/vlm_screen.csv")}
        report_dirs.update(path.parent for path in args.root.glob("**/report*.md"))
        for directory in sorted(report_dirs):
            atomic_json(directory / "legacy_protocol_notice.json", notice)
        atomic_json(args.root / "docvqa_anls_migration.json", {
            "status": "complete",
            "prediction_files": len(results),
            "backup_name": "official_score.legacy_raw_distance.json",
            "invalidated_report_directories": [str(path) for path in sorted(report_dirs)],
            "protocol_version": PROTOCOL_VERSION,
        })
    print(json.dumps({"apply": args.apply, "files": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
