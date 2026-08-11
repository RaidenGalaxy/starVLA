"""Recommended safe alias for the unified matrix runner."""

# Importing safe_methods first installs the bounded RTN audit serializer before
# runner.py imports apply_method.
from . import safe_methods as _safe_methods  # noqa: F401
from . import runner as _runner

_runner.apply_method = _safe_methods.apply_method

from .run_matrix import main  # noqa: E402


if __name__ == "__main__":
    main()
