"""Create deterministic, source-disjoint CPU pilot subsets and download jobs."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from .adapters import fpsbench_trials, moment_video_trials
from .protocol import Segment, StreamTrial, atomic_json, atomic_jsonl, file_sha256


T = TypeVar("T")


def _rank(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def balanced_select(
    items: list[T],
    count: int,
    *,
    stratum: Callable[[T], str],
    identity: Callable[[T], str],
    seed: int,
    excluded: set[str] | None = None,
) -> list[T]:
    """Deterministic round-robin selection across non-empty strata."""
    excluded = set() if excluded is None else set(excluded)
    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        if identity(item) not in excluded:
            groups[stratum(item)].append(item)
    for key, values in groups.items():
        values.sort(key=lambda item: _rank(seed, f"{key}:{identity(item)}"))
    keys = sorted(groups, key=lambda key: _rank(seed, key))
    result: list[T] = []
    used = set(excluded)
    while len(result) < count:
        progressed = False
        for key in keys:
            while groups[key] and identity(groups[key][0]) in used:
                groups[key].pop(0)
            if groups[key] and len(result) < count:
                item = groups[key].pop(0)
                result.append(item)
                used.add(identity(item))
                progressed = True
        if not progressed:
            break
    if len(result) < count:
        raise ValueError(f"requested {count} source-disjoint items but only selected {len(result)}")
    return result


def _fps_duration(trial: StreamTrial) -> float:
    return sum(span.end_sec - span.start_sec for span in trial.evidence_spans)


def _with_fps_local_clip(trial: StreamTrial, media_root: Path, split: str, cohort: str) -> StreamTrial:
    duration = trial.segments[0].duration_sec
    metadata = dict(trial.metadata)
    metadata.update({"pilot_cohort": cohort, "upstream_clip_start_sec": trial.segments[0].start_sec})
    condition = dict(trial.condition)
    condition.update({"appended_delay_sec": 0.0, "distractor_profile": "clean", "virtual_hold_last": False})
    local_path = (media_root / "fpsbench" / f"{trial.sample_id}.mp4").resolve()
    return dataclasses.replace(
        trial,
        split=split,
        segments=(Segment(str(local_path), 0.0, duration, "source", trial.source_group),),
        query_time_sec=duration,
        condition=condition,
        metadata=metadata,
    )


def _summary_counts(items: list[StreamTrial], field: Callable[[StreamTrial], str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[field(item)] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps-annotations", type=Path, required=True)
    parser.add_argument("--moment-annotations", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps-total", type=int, default=96)
    parser.add_argument("--fps-hard", type=int, default=48)
    parser.add_argument("--fps-dev", type=int, default=24)
    parser.add_argument("--moment-qa", type=int, default=32)
    parser.add_argument("--moment-distractors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (0 <= args.fps_hard <= args.fps_total and 0 <= args.fps_dev < args.fps_total):
        parser.error("invalid FPS subset sizes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.media_root.mkdir(parents=True, exist_ok=True)

    fps_all = fpsbench_trials(args.fps_annotations, args.media_root / "fpsbench")
    hard_candidates = [
        trial
        for trial in fps_all
        if _fps_duration(trial) <= 2.0 and (trial.min_fps or 0.0) >= 8.0
    ]
    strict_hard_count = min(args.fps_hard, len({item.source_group for item in hard_candidates}))
    strict_hard = balanced_select(
        hard_candidates,
        strict_hard_count,
        stratum=lambda item: str(item.metadata["task_type"]),
        identity=lambda item: item.source_group,
        seed=args.seed,
    )
    strict_groups = {item.source_group for item in strict_hard}
    fallback_hard = balanced_select(
        [
            trial
            for trial in fps_all
            if _fps_duration(trial) <= 3.0 and (trial.min_fps or 0.0) >= 8.0
        ],
        args.fps_hard - len(strict_hard),
        stratum=lambda item: str(item.metadata["task_type"]),
        identity=lambda item: item.source_group,
        seed=args.seed + 17,
        excluded=strict_groups,
    ) if len(strict_hard) < args.fps_hard else []
    hard = strict_hard + fallback_hard
    used = {item.source_group for item in hard}
    balanced = balanced_select(
        fps_all,
        args.fps_total - args.fps_hard,
        stratum=lambda item: str(item.metadata["task_type"]),
        identity=lambda item: item.source_group,
        seed=args.seed + 1,
        excluded=used,
    )
    cohorts = {item.source_group: "temporal_hard" for item in hard}
    cohorts.update({item.source_group: "task_balanced" for item in balanced})
    selected_fps = hard + balanced
    dev = balanced_select(
        selected_fps,
        args.fps_dev,
        stratum=lambda item: f"{cohorts[item.source_group]}:{item.metadata['task_type']}",
        identity=lambda item: item.source_group,
        seed=args.seed + 2,
    )
    dev_groups = {item.source_group for item in dev}
    fps_trials = [
        _with_fps_local_clip(
            item,
            args.media_root,
            "dev" if item.source_group in dev_groups else "eval",
            cohorts[item.source_group],
        )
        for item in selected_fps
    ]
    fps_trials.sort(key=lambda item: (item.split, item.sample_id))
    fps_downloads = []
    upstream_by_id = {item.sample_id: item for item in selected_fps}
    for trial in fps_trials:
        upstream = upstream_by_id[trial.sample_id]
        fps_downloads.append(
            {
                "backend": "yt_dlp",
                "sample_id": trial.sample_id,
                "source_group": trial.source_group,
                "split": trial.split,
                "url": upstream.metadata["source_url"],
                "start_sec": upstream.segments[0].start_sec,
                "end_sec": upstream.segments[0].end_sec,
                "output_path": trial.segments[0].path,
            }
        )

    moment_all = moment_video_trials(args.moment_annotations, args.media_root / "moment" / "data" / "videos")
    with args.moment_annotations.open(encoding="utf-8") as handle:
        raw_moment = json.load(handle)
    raw_by_key = {
        (str(row["Category"]), str(row["Subclass"]), int(row["Index"])): row for row in raw_moment
    }
    closed = [trial for trial in moment_all if trial.question.answer_type == "multiple_choice"]
    moment_qa = balanced_select(
        closed,
        args.moment_qa,
        stratum=lambda item: f"{item.metadata['category']}:{item.metadata['task_type']}",
        identity=lambda item: item.source_group,
        seed=args.seed + 3,
    )
    qa_groups = {item.source_group for item in moment_qa}
    moment_distractors = balanced_select(
        moment_all,
        args.moment_distractors,
        stratum=lambda item: str(item.metadata["category"]),
        identity=lambda item: item.source_group,
        seed=args.seed + 4,
        excluded=qa_groups,
    )

    def raw_row(trial: StreamTrial) -> dict[str, Any]:
        return raw_by_key[
            (
                str(trial.metadata["category"]),
                str(trial.metadata["subclass"]),
                int(Path(trial.segments[0].path).stem),
            )
        ]

    qa_raw = [raw_row(item) for item in moment_qa]
    distractor_raw = [raw_row(item) for item in moment_distractors]
    moment_downloads = []
    for purpose, trials in (("qa", moment_qa), ("distractor", moment_distractors)):
        for trial in trials:
            row = raw_row(trial)
            filename = f"data/videos/{row['Category']}/{row['Subclass']}/{int(row['Index'])}.mp4"
            moment_downloads.append(
                {
                    "backend": "huggingface",
                    "repo_id": "VisionXLab/Moment-Video",
                    "repo_type": "dataset",
                    "filename": filename,
                    "sample_id": trial.sample_id,
                    "source_group": trial.source_group,
                    "purpose": purpose,
                    "output_path": str((args.media_root / "moment" / filename).resolve()),
                }
            )

    fps_manifest = args.output_dir / "fpsbench_pilot_base.jsonl"
    atomic_jsonl(fps_manifest, (item.to_dict(relative_to=args.output_dir) for item in fps_trials))
    atomic_jsonl(args.output_dir / "fpsbench_downloads.jsonl", fps_downloads)
    atomic_jsonl(args.output_dir / "moment_downloads.jsonl", moment_downloads)
    atomic_jsonl(args.output_dir / "moment_qa_unprobed.jsonl", (item.to_dict(relative_to=args.output_dir) for item in moment_qa))
    atomic_jsonl(
        args.output_dir / "moment_distractors_unprobed.jsonl",
        (item.to_dict(relative_to=args.output_dir) for item in moment_distractors),
    )
    atomic_json(args.output_dir / "moment_qa_annotations.json", qa_raw)
    atomic_json(args.output_dir / "moment_distractor_annotations.json", distractor_raw)
    report = {
        "format_version": 1,
        "seed": args.seed,
        "inputs": {
            "fps_annotations": str(args.fps_annotations.resolve()),
            "fps_annotations_sha256": file_sha256(args.fps_annotations),
            "moment_annotations": str(args.moment_annotations.resolve()),
            "moment_annotations_sha256": file_sha256(args.moment_annotations),
        },
        "fpsbench": {
            "total": len(fps_trials),
            "hard": len(hard),
            "strict_hard_certificate_le_2s_min_fps_ge_8": len(strict_hard),
            "relaxed_hard_certificate_le_3s_min_fps_ge_8": len(fallback_hard),
            "dev": sum(item.split == "dev" for item in fps_trials),
            "eval": sum(item.split == "eval" for item in fps_trials),
            "unique_source_groups": len({item.source_group for item in fps_trials}),
            "task_counts": _summary_counts(fps_trials, lambda item: str(item.metadata["task_type"])),
            "domain_counts": _summary_counts(fps_trials, lambda item: str(item.metadata["visual_domain"])),
            "dev_eval_overlap": sorted(
                {item.source_group for item in fps_trials if item.split == "dev"}
                & {item.source_group for item in fps_trials if item.split == "eval"}
            ),
            "manifest_sha256": file_sha256(fps_manifest),
        },
        "moment_video": {
            "qa": len(moment_qa),
            "distractors": len(moment_distractors),
            "closed_question_categories": sorted({str(item.metadata["category"]) for item in closed}),
            "upstream_categories_without_closed_questions": sorted(
                {str(item.metadata["category"]) for item in moment_all}
                - {str(item.metadata["category"]) for item in closed}
            ),
            "qa_distractor_overlap": sorted(qa_groups & {item.source_group for item in moment_distractors}),
            "qa_category_counts": _summary_counts(moment_qa, lambda item: str(item.metadata["category"])),
            "qa_task_counts": _summary_counts(moment_qa, lambda item: str(item.metadata["task_type"])),
        },
        "download_jobs": len(fps_downloads) + len(moment_downloads),
    }
    atomic_json(args.output_dir / "sampling_report.json", report)
    print(
        f"prepared {len(fps_trials)} FPS-Bench clips, {len(moment_qa)} Moment QA, "
        f"and {len(moment_distractors)} Moment distractors in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
