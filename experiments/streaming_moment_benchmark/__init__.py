"""Future-query-agnostic streaming benchmark for momentary visual evidence."""

from .protocol import PROTOCOL_VERSION, EvidenceSpan, Question, Segment, StreamTrial

__all__ = [
    "PROTOCOL_VERSION",
    "EvidenceSpan",
    "Question",
    "Segment",
    "StreamTrial",
]
