"""Strict-budget causal frame admission and retention policies."""
from __future__ import annotations

import dataclasses
import math
import random

import numpy as np

from .frames import FrameObservation, FrameRef
from .utility import UtilityModel


METHODS = (
    "uniform_fifo",
    "recent_fifo",
    "reservoir",
    "motion_topk",
    "novelty_topk",
    "future_utility",
)


@dataclasses.dataclass
class MemoryEntry:
    observation: FrameObservation
    score: float
    admission_index: int


@dataclasses.dataclass(frozen=True)
class SelectionResult:
    method: str
    memory_frames: int
    encode_fps: float
    window_sec: float
    frames_seen: int
    frames_admitted: int
    captured: tuple[FrameRef, ...]
    evicted: tuple[FrameRef, ...]
    final_observations: tuple[FrameObservation, ...]

    @property
    def final_refs(self) -> tuple[FrameRef, ...]:
        return tuple(item.ref for item in self.final_observations)


class OnlineMemorySelector:
    def __init__(
        self,
        method: str,
        *,
        memory_frames: int,
        encode_fps: float,
        window_sec: float = 1.0,
        seed: int = 42,
        utility_model: UtilityModel | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method}; choose from {METHODS}")
        if memory_frames <= 0 or encode_fps <= 0 or window_sec <= 0:
            raise ValueError("memory_frames, encode_fps, and window_sec must be positive")
        if method == "future_utility" and utility_model is None:
            raise ValueError("future_utility requires a trained utility model")
        self.method = method
        self.memory_frames = memory_frames
        self.encode_fps = encode_fps
        self.window_sec = window_sec
        self.utility_model = utility_model
        self._rng = random.Random(seed)
        self._entries: list[MemoryEntry] = []
        self._protected: list[MemoryEntry] = []
        self._recent: list[MemoryEntry] = []
        self._captured: list[FrameRef] = []
        self._evicted: list[FrameRef] = []
        self._frames_seen = 0
        self._admissions = 0
        self._windows_processed = 0

    def _quota(self) -> int:
        previous = math.floor(self._windows_processed * self.window_sec * self.encode_fps + 1e-9)
        current = math.floor((self._windows_processed + 1) * self.window_sec * self.encode_fps + 1e-9)
        return max(0, current - previous)

    @staticmethod
    def _evenly_spaced(frames: list[FrameObservation], count: int) -> list[FrameObservation]:
        if count >= len(frames):
            return list(frames)
        indices = np.linspace(0, len(frames) - 1, count, dtype=int)
        return [frames[int(index)] for index in indices]

    def _novelty(self, frame: FrameObservation, anchors: list[MemoryEntry]) -> float:
        if not anchors:
            return 1.0
        visual_size = len(frame.features) - 2
        return min(
            float(np.linalg.norm(frame.features[:visual_size] - item.observation.features[:visual_size]))
            for item in anchors
        )

    def _rank_window(self, frames: list[FrameObservation], count: int) -> list[tuple[FrameObservation, float]]:
        count = min(count, len(frames))
        if count <= 0:
            return []
        if self.method in {"uniform_fifo", "reservoir"}:
            return [(item, 0.0) for item in self._evenly_spaced(frames, count)]
        if self.method == "recent_fifo":
            return [(item, item.ref.timestamp_sec) for item in frames[-count:]]
        if self.method == "motion_topk":
            ranked = sorted(frames, key=lambda item: (float(item.features[-2]), item.ref.frame_index), reverse=True)
            return [(item, float(item.features[-2])) for item in ranked[:count]]
        if self.method == "novelty_topk":
            anchors = list(self._entries)
            selected: list[tuple[FrameObservation, float]] = []
            remaining = list(frames)
            while remaining and len(selected) < count:
                best_index, score = max(
                    enumerate(self._novelty(item, anchors) for item in remaining),
                    key=lambda pair: (pair[1], -remaining[pair[0]].ref.frame_index),
                )
                item = remaining.pop(best_index)
                selected.append((item, score))
                anchor = MemoryEntry(item, score, self._admissions + len(selected))
                anchors.append(anchor)
            return selected
        assert self.utility_model is not None
        scores = self.utility_model.predict(np.stack([item.features for item in frames]))
        ranked = sorted(zip(frames, scores.tolist()), key=lambda pair: (pair[1], -pair[0].ref.frame_index), reverse=True)
        return [(item, float(score)) for item, score in ranked[:count]]

    def _fifo(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.memory_frames:
            self._evicted.append(self._entries.pop(0).observation.ref)

    def _reservoir(self, entry: MemoryEntry) -> None:
        if len(self._entries) < self.memory_frames:
            self._entries.append(entry)
            return
        position = self._rng.randrange(self._admissions)
        if position < self.memory_frames:
            self._evicted.append(self._entries[position].observation.ref)
            self._entries[position] = entry
        else:
            self._evicted.append(entry.observation.ref)

    def _priority(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.memory_frames:
            lowest = min(
                range(len(self._entries)),
                key=lambda index: (self._entries[index].score, self._entries[index].admission_index),
            )
            self._evicted.append(self._entries.pop(lowest).observation.ref)

    def _utility_memory(self, entry: MemoryEntry) -> None:
        protected_capacity = max(1, math.ceil(self.memory_frames * 0.75))
        recent_capacity = self.memory_frames - protected_capacity
        if len(self._protected) < protected_capacity:
            self._protected.append(entry)
        else:
            lowest = min(range(len(self._protected)), key=lambda index: self._protected[index].score)
            if entry.score > self._protected[lowest].score:
                displaced = self._protected[lowest]
                self._protected[lowest] = entry
                if recent_capacity:
                    self._recent.append(displaced)
                else:
                    self._evicted.append(displaced.observation.ref)
            elif recent_capacity:
                self._recent.append(entry)
            else:
                self._evicted.append(entry.observation.ref)
        while len(self._recent) > recent_capacity:
            self._evicted.append(self._recent.pop(0).observation.ref)

    def ingest_window(self, frames: list[FrameObservation]) -> None:
        if not frames:
            return
        self._frames_seen += len(frames)
        selected = self._rank_window(frames, self._quota())
        self._windows_processed += 1
        for observation, score in sorted(selected, key=lambda pair: pair[0].ref.timestamp_sec):
            self._admissions += 1
            self._captured.append(observation.ref)
            entry = MemoryEntry(observation, score, self._admissions)
            if self.method in {"uniform_fifo", "recent_fifo"}:
                self._fifo(entry)
            elif self.method == "reservoir":
                self._reservoir(entry)
            elif self.method in {"motion_topk", "novelty_topk"}:
                self._priority(entry)
            else:
                self._utility_memory(entry)

    def result(self) -> SelectionResult:
        entries = self._protected + self._recent if self.method == "future_utility" else self._entries
        observations = tuple(
            item.observation
            for item in sorted(entries, key=lambda item: item.observation.ref.timestamp_sec)
        )
        if len(observations) > self.memory_frames:
            raise AssertionError("selector exceeded its memory budget")
        return SelectionResult(
            method=self.method,
            memory_frames=self.memory_frames,
            encode_fps=self.encode_fps,
            window_sec=self.window_sec,
            frames_seen=self._frames_seen,
            frames_admitted=self._admissions,
            captured=tuple(self._captured),
            evicted=tuple(self._evicted),
            final_observations=observations,
        )
