"""Manifest schema and validation for the streaming moment benchmark."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1


def _finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclasses.dataclass(frozen=True)
class Question:
    text: str
    answer: str
    answer_type: str = "open"
    answer_text: str = ""
    choices: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.answer.strip():
            raise ValueError("question text and answer must be non-empty")
        if self.answer_type not in {"multiple_choice", "open"}:
            raise ValueError("answer_type must be multiple_choice or open")
        if self.answer_type == "multiple_choice":
            if len(self.choices) < 2 or self.answer not in self.choices:
                raise ValueError("multiple-choice answer must name one of at least two choices")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Question:
        return cls(
            text=str(row["text"]),
            answer=str(row["answer"]),
            answer_type=str(row.get("answer_type", "open")),
            answer_text=str(row.get("answer_text", "")),
            choices={str(key): str(value) for key, value in row.get("choices", {}).items()},
        )


@dataclasses.dataclass(frozen=True)
class EvidenceSpan:
    start_sec: float
    end_sec: float
    source: str = "annotation"

    def __post_init__(self) -> None:
        _finite_non_negative("evidence start", self.start_sec)
        _finite_non_negative("evidence end", self.end_sec)
        if self.end_sec < self.start_sec:
            raise ValueError("evidence span ends before it starts")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> EvidenceSpan:
        return cls(float(row["start_sec"]), float(row["end_sec"]), str(row.get("source", "annotation")))


@dataclasses.dataclass(frozen=True)
class Segment:
    path: str
    start_sec: float
    end_sec: float
    role: str = "source"
    source_group: str = ""
    semantic_domain: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("segment path must be non-empty")
        _finite_non_negative("segment start", self.start_sec)
        _finite_non_negative("segment end", self.end_sec)
        if self.end_sec <= self.start_sec:
            raise ValueError("segment end must be greater than start")
        if self.role not in {"source", "clean", "redundancy", "novelty"}:
            raise ValueError(f"unsupported segment role: {self.role}")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @classmethod
    def from_dict(cls, row: dict[str, Any], *, base_dir: Path | None = None) -> Segment:
        path = Path(str(row["path"])).expanduser()
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return cls(
            path=str(path.resolve()),
            start_sec=float(row.get("start_sec", 0.0)),
            end_sec=float(row["end_sec"]),
            role=str(row.get("role", "source")),
            source_group=str(row.get("source_group", "")),
            semantic_domain=str(row.get("semantic_domain", "")),
        )


@dataclasses.dataclass(frozen=True)
class StreamTrial:
    trial_id: str
    sample_id: str
    dataset: str
    split: str
    source_group: str
    question: Question
    segments: tuple[Segment, ...]
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    query_time_sec: float = 0.0
    min_fps: float | None = None
    native_fps: float | None = None
    condition: dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"expected protocol_version={PROTOCOL_VERSION}")
        for name, value in (
            ("trial_id", self.trial_id),
            ("sample_id", self.sample_id),
            ("dataset", self.dataset),
            ("split", self.split),
            ("source_group", self.source_group),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.segments:
            raise ValueError("a trial needs at least one segment")
        _finite_non_negative("query time", self.query_time_sec)
        if self.query_time_sec + 1e-6 < self.stream_duration_sec:
            raise ValueError("query time cannot precede the end of the stream")
        for span in self.evidence_spans:
            if span.end_sec > self.stream_duration_sec + 1e-6:
                raise ValueError("evidence span lies outside the stream")
        for name, value in (("min_fps", self.min_fps), ("native_fps", self.native_fps)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

    @property
    def stream_duration_sec(self) -> float:
        return sum(segment.duration_sec for segment in self.segments)

    @property
    def query_delay_sec(self) -> float | None:
        if not self.evidence_spans:
            return None
        return self.query_time_sec - max(span.end_sec for span in self.evidence_spans)

    @classmethod
    def from_dict(cls, row: dict[str, Any], *, base_dir: Path | None = None) -> StreamTrial:
        return cls(
            trial_id=str(row["trial_id"]),
            sample_id=str(row.get("sample_id", row["trial_id"])),
            dataset=str(row["dataset"]),
            split=str(row.get("split", "test")),
            source_group=str(row.get("source_group", row.get("sample_id", row["trial_id"]))),
            question=Question.from_dict(dict(row["question"])),
            segments=tuple(Segment.from_dict(item, base_dir=base_dir) for item in row["segments"]),
            evidence_spans=tuple(EvidenceSpan.from_dict(item) for item in row.get("evidence_spans", [])),
            query_time_sec=float(row.get("query_time_sec", 0.0)),
            min_fps=float(row["min_fps"]) if row.get("min_fps") is not None else None,
            native_fps=float(row["native_fps"]) if row.get("native_fps") is not None else None,
            condition=dict(row.get("condition", {})),
            metadata=dict(row.get("metadata", {})),
            protocol_version=int(row.get("protocol_version", PROTOCOL_VERSION)),
        )

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["segments"] = []
        for segment in self.segments:
            item = dataclasses.asdict(segment)
            if relative_to is not None:
                try:
                    item["path"] = os.path.relpath(segment.path, relative_to)
                except ValueError:
                    pass
            row["segments"].append(item)
        return row


def load_trials(path: Path, *, require_files: bool = False) -> list[StreamTrial]:
    base_dir = path.resolve().parent
    trials: list[StreamTrial] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                trial = StreamTrial.from_dict(json.loads(line), base_dir=base_dir)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid trial at {path}:{line_number}: {error}") from error
            if trial.trial_id in seen:
                raise ValueError(f"duplicate trial_id: {trial.trial_id}")
            seen.add(trial.trial_id)
            if require_files:
                missing = [segment.path for segment in trial.segments if not Path(segment.path).exists()]
                if missing:
                    raise FileNotFoundError(f"missing media for {trial.trial_id}: {missing}")
            trials.append(trial)
    if not trials:
        raise ValueError(f"manifest is empty: {path}")
    return trials


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
