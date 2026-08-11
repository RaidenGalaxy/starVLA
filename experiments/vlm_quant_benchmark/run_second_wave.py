"""Matrix entry point enabling OmniQuant, SpQR, and SqueezeLLM CPU research paths."""

from . import all_methods as _all_methods
from . import runner as _runner

_runner.CALIBRATION_METHODS.update(_all_methods.SECOND_WAVE)
_runner.apply_method = _all_methods.apply_method

from .run_matrix import main  # noqa: E402


if __name__ == "__main__":
    main()
