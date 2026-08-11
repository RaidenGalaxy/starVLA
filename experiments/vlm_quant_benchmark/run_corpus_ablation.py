"""Memory-safe alias for corpus perplexity evaluation."""

from . import safe_methods as _safe_methods
from . import run_corpus_ppl as _entry

_entry.apply_method = _safe_methods.apply_method


if __name__ == "__main__":
    _entry.main()
