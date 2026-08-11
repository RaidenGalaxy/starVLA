"""Corpus-PPL entry point enabling the second method wave."""

from . import all_methods as _all_methods
from . import run_corpus_ppl as _entry

_entry.CALIBRATION_METHODS.update(_all_methods.SECOND_WAVE)
_entry.apply_method = _all_methods.apply_method


if __name__ == "__main__":
    _entry.main()
