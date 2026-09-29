from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.streaming_moment_benchmark.adapters import (
    _moment_question,
    add_delayed_distractors,
    fpsbench_trials,
    moment_video_trials,
)
from experiments.streaming_moment_benchmark.protocol import Segment, atomic_jsonl, load_trials


def _fps_row() -> dict:
    return {
        "id": "fpsbench_000000",
        "version": "1.0.0",
        "split": "test",
        "source": {"dataset": "youtube8m", "platform": "youtube", "video_id": "abcdefghijk", "url": "x"},
        "time": {
            "clip_start_sec": 100,
            "clip_end_sec": 106,
            "temporal_certificate_start_sec": 101,
            "temporal_certificate_end_sec": 101.2,
        },
        "question": {
            "text": "What happened?",
            "type": "blink_and_miss",
            "choices": {"A": "red", "B": "blue", "C": "green", "D": "white"},
            "answer": "A",
            "answer_text": "red",
        },
        "temporal_requirements": {"min_fps": 10, "min_required_frames_for_certificate": 2, "native_fps": 30},
        "categories": {"visual_domain": "GUI", "visual_subdomain": "indicator"},
    }


def test_fpsbench_adapter_uses_stream_relative_certificate(tmp_path: Path) -> None:
    annotations = tmp_path / "fps.jsonl"
    annotations.write_text(json.dumps(_fps_row()) + "\n", encoding="utf-8")
    trial = fpsbench_trials(annotations, tmp_path / "videos")[0]
    assert trial.evidence_spans[0].start_sec == pytest.approx(1.0)
    assert trial.evidence_spans[0].end_sec == pytest.approx(1.2)
    assert trial.segments[0].start_sec == 100
    assert trial.query_delay_sec == pytest.approx(4.8)


def test_fpsbench_adapter_detects_clip_relative_certificate(tmp_path: Path) -> None:
    row = _fps_row()
    row["time"]["temporal_certificate_start_sec"] = 1
    row["time"]["temporal_certificate_end_sec"] = 1.2
    annotations = tmp_path / "fps.jsonl"
    annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    trial = fpsbench_trials(annotations, tmp_path / "videos")[0]
    assert trial.evidence_spans[0].start_sec == pytest.approx(1.0)
    assert trial.metadata["certificate_timebase"] == "clip_relative"


def test_fpsbench_adapter_clips_certificate_to_stream(tmp_path: Path) -> None:
    row = _fps_row()
    row["time"]["temporal_certificate_start_sec"] = 105
    row["time"]["temporal_certificate_end_sec"] = 107
    annotations = tmp_path / "fps.jsonl"
    annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    trial = fpsbench_trials(annotations, tmp_path / "videos")[0]
    assert trial.evidence_spans[0].start_sec == pytest.approx(5.0)
    assert trial.evidence_spans[0].end_sec == pytest.approx(6.0)
    assert trial.metadata["certificate_timebase"] == "source_absolute_clipped"


def test_manifest_roundtrip_and_duplicate_rejection(tmp_path: Path) -> None:
    annotations = tmp_path / "fps.jsonl"
    annotations.write_text(json.dumps(_fps_row()) + "\n", encoding="utf-8")
    trial = fpsbench_trials(annotations, tmp_path / "videos")[0]
    manifest = tmp_path / "trials.jsonl"
    atomic_jsonl(manifest, [trial.to_dict(relative_to=tmp_path)])
    loaded = load_trials(manifest)
    assert loaded[0].trial_id == trial.trial_id
    atomic_jsonl(manifest, [trial.to_dict(relative_to=tmp_path), trial.to_dict(relative_to=tmp_path)])
    with pytest.raises(ValueError, match="duplicate trial_id"):
        load_trials(manifest)


def test_delay_builder_excludes_same_source_group(tmp_path: Path) -> None:
    annotations = tmp_path / "fps.jsonl"
    annotations.write_text(json.dumps(_fps_row()) + "\n", encoding="utf-8")
    trial = fpsbench_trials(annotations, tmp_path / "videos")[0]
    bank = [
        Segment(str(tmp_path / "same.mp4"), 0, 10, "novelty", trial.source_group),
        Segment(str(tmp_path / "other.mp4"), 0, 10, "novelty", "other"),
    ]
    delayed = add_delayed_distractors([trial], bank, delays_sec=[2], profiles=["novelty"])[0]
    assert delayed.segments[-1].source_group == "other"
    assert delayed.stream_duration_sec == pytest.approx(8.0)
    assert delayed.query_delay_sec == pytest.approx(6.8)


def test_clean_delay_is_virtual_hold_last_without_extra_media(tmp_path: Path) -> None:
    annotations = tmp_path / "fps.jsonl"
    annotations.write_text(json.dumps(_fps_row()) + "\n", encoding="utf-8")
    trial = fpsbench_trials(annotations, tmp_path / "videos")[0]
    delayed = add_delayed_distractors([trial], [], delays_sec=[120], profiles=["clean"])[0]
    assert delayed.segments == trial.segments
    assert delayed.query_time_sec == pytest.approx(trial.stream_duration_sec + 120)
    assert delayed.condition["virtual_hold_last"] is True


def test_moment_closed_question_is_locally_scorable() -> None:
    question = _moment_question(
        {
            "Question": "What changed?\n(a) Nothing.\n(b) The light turned red.\n(c) It moved.",
            "Answer": "(b)",
            "AnswerType": "closed",
        }
    )
    assert question.text == "What changed?"
    assert question.answer_type == "multiple_choice"
    assert question.answer == "B"
    assert question.choices["B"] == "The light turned red."


def test_moment_inline_choices_are_parsed() -> None:
    question = _moment_question(
        {
            "Question": "Why did they separate? (a) Left moved. (b) Right moved. (c) A skill pushed them.",
            "Answer": "(c)",
            "AnswerType": "closed",
        }
    )
    assert question.text == "Why did they separate?"
    assert question.answer == "C"


def test_moment_media_path_preserves_official_case(tmp_path: Path) -> None:
    annotations = tmp_path / "moment.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "Category": "GUI",
                    "Subclass": "WebNavigation",
                    "Index": 7,
                    "Question": "What happened? (a) A (b) B",
                    "Answer": "(a)",
                    "AnswerType": "closed",
                    "QuestionType": "temporal",
                }
            ]
        ),
        encoding="utf-8",
    )
    trial = moment_video_trials(annotations, tmp_path / "videos")[0]
    assert Path(trial.segments[0].path) == tmp_path / "videos" / "GUI" / "WebNavigation" / "7.mp4"
