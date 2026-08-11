"""Recommended import path for the VLMEvalKit model adapter."""

from . import safe_methods as _safe_methods  # patches bounded RTN audit path
from .vlmeval_adapter import Qwen25VLQuantized

__all__ = ["Qwen25VLQuantized"]
