"""Adapters for FPS-Bench, Moment-Video, and delayed distractor trials."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .protocol import EvidenceSpan, Question, Segment, StreamTrial


_MOMENT_OPTION = re.compile(
    r"(?is)\(([a-e])\)\s*(.*?)(?=\s*\([a-e]\)\s*|\Z)"
)


def _moment_question(row: dict[str, Any]) -> Question:
    raw_question = str(row["Question"])
    raw_answer = str(row["Answer"]).strip()
    answer_match = re.fullmatch(r"\(?\s*([a-eA-E])\s*\)?", raw_answer)
    option_matches = list(_MOMENT_OPTION.finditer(raw_question))
    if answer_match and len(option_matches) >= 2:
        choices = {
            match.group(1).upper(): " ".join(match.group(2).strip().split())
            for match in option_matches
        }
        answer = answer_match.group(1).upper()
        if answer not in choices:
            raise ValueError(f"Moment-Video answer {raw_answer!r} does not name a parsed choice")
        stem = raw_question[: option_matches[0].start()].strip()
        return Question(stem, answer, "multiple_choice", choices[answer], choices)
    if str(row.get("AnswerType", "open")).lower() == "closed":
        raise ValueError("could not parse a closed Moment-Video question")
    return Question(raw_question, raw_answer, "open", raw_answer)


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list: {path}")
    return value


def fpsbench_trials(
    annotation_path: Path,
    video_root: Path,
    *,
    path_template: str = "{id}.mp4",
) -> list[StreamTrial]:
    trials: list[StreamTrial] = []
    for row in _read_json_or_jsonl(annotation_path):
        sample_id = str(row["id"])
        source = dict(row["source"])
        time = dict(row["time"])
        question = dict(row["question"])
        requirements = dict(row["temporal_requirements"])
        categories = dict(row["categories"])
        source_group = f"{source['platform']}:{source['video_id']}"
        relative_path = path_template.format(
            id=sample_id,
            video_id=source["video_id"],
            dataset=source["dataset"],
        )
        clip_start = float(time["clip_start_sec"])
        clip_end = float(time["clip_end_sec"])
        duration = clip_end - clip_start
        certificate_start = float(time["temporal_certificate_start_sec"])
        certificate_end = float(time["temporal_certificate_end_sec"])
        if certificate_end > clip_start and certificate_start < clip_end:
            clipped_start = max(certificate_start, clip_start)
            clipped_end = min(certificate_end, clip_end)
            evidence_start = clipped_start - clip_start
            evidence_end = clipped_end - clip_start
            certificate_timebase = (
                "source_absolute"
                if clipped_start == certificate_start and clipped_end == certificate_end
                else "source_absolute_clipped"
            )
        elif certificate_end > 0 and certificate_start < duration:
            evidence_start = max(0.0, certificate_start)
            evidence_end = min(duration, certificate_end)
            certificate_timebase = (
                "clip_relative"
                if evidence_start == certificate_start and evidence_end == certificate_end
                else "clip_relative_clipped"
            )
        else:
            raise ValueError(f"invalid temporal certificate for {sample_id}")
        segment = Segment(
            path=str((video_root / relative_path).resolve()),
            start_sec=clip_start,
            end_sec=clip_end,
            role="source",
            source_group=source_group,
        )
        duration = segment.duration_sec
        trials.append(
            StreamTrial(
                trial_id=f"{sample_id}__delay0__clean",
                sample_id=sample_id,
                dataset="fpsbench_v1",
                split=str(row.get("split", "test")),
                source_group=source_group,
                question=Question(
                    text=str(question["text"]),
                    answer=str(question["answer"]),
                    answer_type="multiple_choice",
                    answer_text=str(question.get("answer_text", "")),
                    choices={str(key): str(value) for key, value in question["choices"].items()},
                ),
                segments=(segment,),
                evidence_spans=(EvidenceSpan(evidence_start, evidence_end, "fpsbench_temporal_certificate"),),
                query_time_sec=duration,
                min_fps=float(requirements["min_fps"]),
                native_fps=(
                    float(requirements["native_fps"])
                    if requirements.get("native_fps") is not None
                    else None
                ),
                condition={"delay_sec": round(max(0.0, duration - evidence_end), 6), "distractor_profile": "clean"},
                metadata={
                    "task_type": question["type"],
                    "visual_domain": categories["visual_domain"],
                    "visual_subdomain": categories["visual_subdomain"],
                    "min_required_frames": requirements.get("min_required_frames_for_certificate"),
                    "source_url": source["url"],
                    "annotation_version": row.get("version"),
                    "certificate_timebase": certificate_timebase,
                    "certificate_clipped": certificate_timebase.endswith("_clipped"),
                },
            )
        )
    return trials


def moment_video_trials(annotation_path: Path, video_root: Path) -> list[StreamTrial]:
    rows: list[dict[str, Any]]
    if annotation_path.suffix.lower() == ".csv":
        with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = _read_json_or_jsonl(annotation_path)
    trials: list[StreamTrial] = []
    for row in rows:
        category = str(row["Category"])
        subclass = str(row["Subclass"])
        index = str(int(row["Index"]))
        sample_id = f"moment_video_{category.lower()}_{subclass.lower()}_{int(index):04d}"
        # Hugging Face stores these components with the annotation's original
        # case (notably ``AIGC`` and ``GUI``), so keep them byte-for-byte.
        path = (video_root / category / subclass / f"{index}.mp4").resolve()
        # Moment-Video does not publish temporal certificates. The end time is
        # filled by probe_media_lengths before the trial can be evaluated.
        trials.append(
            StreamTrial(
                trial_id=f"{sample_id}__unprobed",
                sample_id=sample_id,
                dataset="moment_video",
                split="test",
                source_group=sample_id,
                question=_moment_question(row),
                segments=(Segment(str(path), 0.0, 1e-6, "source", sample_id),),
                evidence_spans=(),
                query_time_sec=1e-6,
                condition={"delay_sec": None, "distractor_profile": "clean"},
                metadata={
                    "category": category,
                    "subclass": subclass,
                    "task_type": str(row["QuestionType"]),
                    "answer_type_upstream": str(row["AnswerType"]),
                    "needs_media_probe": True,
                    "temporal_certificate_available": False,
                },
            )
        )
    return trials


def probe_media_lengths(trials: list[StreamTrial]) -> list[StreamTrial]:
    """Replace placeholder segment lengths with durations reported by OpenCV."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to probe video durations") from error

    result: list[StreamTrial] = []
    for trial in trials:
        if not trial.metadata.get("needs_media_probe"):
            result.append(trial)
            continue
        segment = trial.segments[0]
        capture = cv2.VideoCapture(segment.path)
        try:
            if not capture.isOpened():
                raise FileNotFoundError(f"cannot open video: {segment.path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if fps <= 0 or frames <= 0:
            raise ValueError(f"invalid video metadata: {segment.path}")
        duration = frames / fps
        metadata = dict(trial.metadata)
        metadata["needs_media_probe"] = False
        result.append(
            StreamTrial(
                trial_id=trial.trial_id.replace("__unprobed", "__delay0__clean"),
                sample_id=trial.sample_id,
                dataset=trial.dataset,
                split=trial.split,
                source_group=trial.source_group,
                question=trial.question,
                segments=(Segment(segment.path, 0.0, duration, "source", segment.source_group),),
                evidence_spans=(),
                query_time_sec=duration,
                native_fps=fps,
                condition=trial.condition,
                metadata=metadata,
            )
        )
    return result


def load_distractor_bank(path: Path) -> list[Segment]:
    rows = _read_json_or_jsonl(path)
    result: list[Segment] = []
    for row in rows:
        media_path = Path(row["path"]).expanduser()
        if not media_path.is_absolute():
            media_path = path.resolve().parent / media_path
        result.append(
            Segment(
                path=str(media_path.resolve()),
                start_sec=float(row.get("start_sec", 0.0)),
                end_sec=float(row["end_sec"]),
                role=str(row["profile"]),
                source_group=str(row.get("source_group", row["path"])),
                semantic_domain=str(row.get("semantic_domain", "")),
            )
        )
    return result


def add_delayed_distractors(
    trials: list[StreamTrial],
    bank: list[Segment],
    *,
    delays_sec: list[float],
    profiles: list[str],
    seed: int = 42,
) -> list[StreamTrial]:
    if any(delay < 0 for delay in delays_sec):
        raise ValueError("delays must be non-negative")
    if any(profile not in {"clean", "redundancy", "novelty"} for profile in profiles):
        raise ValueError("unknown distractor profile")
    result: list[StreamTrial] = []
    for trial in trials:
        for delay in delays_sec:
            for profile in profiles:
                if delay == 0 and profile != "clean":
                    continue
                if profile == "clean":
                    query_time = trial.stream_duration_sec + delay
                    condition = dict(trial.condition)
                    total_query_delay = (
                        query_time - max(span.end_sec for span in trial.evidence_spans)
                        if trial.evidence_spans
                        else None
                    )
                    condition.update(
                        {
                            "delay_sec": total_query_delay,
                            "appended_delay_sec": delay,
                            "distractor_profile": "clean",
                            "virtual_hold_last": delay > 0,
                        }
                    )
                    delay_slug = str(delay).replace(".", "p")
                    result.append(
                        StreamTrial(
                            trial_id=f"{trial.sample_id}__delay{delay_slug}__clean",
                            sample_id=trial.sample_id,
                            dataset=trial.dataset,
                            split=trial.split,
                            source_group=trial.source_group,
                            question=trial.question,
                            segments=trial.segments,
                            evidence_spans=trial.evidence_spans,
                            query_time_sec=query_time,
                            min_fps=trial.min_fps,
                            native_fps=trial.native_fps,
                            condition=condition,
                            metadata=trial.metadata,
                        )
                    )
                    continue
                candidates = [
                    item
                    for item in bank
                    if item.role == profile and item.source_group != trial.source_group
                ]
                if profile == "redundancy":
                    domain = str(trial.metadata.get("visual_domain", ""))
                    matched = [item for item in candidates if item.semantic_domain and item.semantic_domain == domain]
                    if matched:
                        candidates = matched
                if delay > 0 and not candidates:
                    raise ValueError(f"no {profile} distractors available for {trial.trial_id}")
                digest = hashlib.sha256(f"{seed}:{trial.trial_id}:{delay}:{profile}".encode()).digest()
                rng = random.Random(int.from_bytes(digest[:8], "big"))
                rng.shuffle(candidates)
                segments = list(trial.segments)
                remaining = delay
                cursor = 0
                while remaining > 1e-9:
                    template = candidates[cursor % len(candidates)]
                    duration = min(remaining, template.duration_sec)
                    segments.append(
                        Segment(
                            template.path,
                            template.start_sec,
                            template.start_sec + duration,
                            profile,
                            template.source_group,
                            template.semantic_domain,
                        )
                    )
                    remaining -= duration
                    cursor += 1
                stream_duration = sum(item.duration_sec for item in segments)
                condition = dict(trial.condition)
                total_query_delay = (
                    stream_duration - max(span.end_sec for span in trial.evidence_spans)
                    if trial.evidence_spans
                    else None
                )
                condition.update(
                    {
                        "delay_sec": total_query_delay,
                        "appended_delay_sec": delay,
                        "distractor_profile": profile,
                    }
                )
                delay_slug = str(delay).replace(".", "p")
                result.append(
                    StreamTrial(
                        trial_id=f"{trial.sample_id}__delay{delay_slug}__{profile}",
                        sample_id=trial.sample_id,
                        dataset=trial.dataset,
                        split=trial.split,
                        source_group=trial.source_group,
                        question=trial.question,
                        segments=tuple(segments),
                        evidence_spans=trial.evidence_spans,
                        query_time_sec=stream_duration,
                        min_fps=trial.min_fps,
                        native_fps=trial.native_fps,
                        condition=condition,
                        metadata=trial.metadata,
                    )
                )
    return result
