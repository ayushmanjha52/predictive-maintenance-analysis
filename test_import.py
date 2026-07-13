"""
Smoke test: every module in the project must import cleanly.

FIX vs. previous version: the old test_import.py only checked ONE import
(MASTER_EVENTS_PATH from config) and had no assertions -- it was a
print-based script, not a real test, and pytest can't discover any real
test in it (no test_ function inside, despite the pytest-matching
filename). It also lived at the project root instead of tests/.

This version checks EVERY module in src/ imports without error -- cheap
(runs in well under a second) and genuinely useful: after any refactor
(like the primary_device/align_features/CausePredictor changes made
across this project), a typo or circular import gets caught here
immediately, rather than surfacing later as a confusing runtime error
in whichever script happens to import the broken module first.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

# Every module expected to exist in src/. Add new ones here as the
# project grows -- this list is the single place that defines "what
# must always import cleanly."
SRC_MODULES = [
    "src.config",
    "src.data_loader",
    "src.feature_engineering",
    "src.fmea_risk",
    "src.forecasting",
    "src.predict",
    "src.train",
    "src.feature_importance",
    "src.hyperparameter_tuning",
]


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_module_imports_cleanly(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"{module_name} failed to import: {e}")


def test_config_has_required_paths():
    """The one thing the old test actually checked -- kept, but as a
    real assertion instead of a print statement."""
    from src.config import MASTER_EVENTS_PATH, MODEL_PATH, VECTORIZER_PATH, FEATURE_NAMES_PATH
    for path_var, name in [(MASTER_EVENTS_PATH, "MASTER_EVENTS_PATH"),
                            (MODEL_PATH, "MODEL_PATH"),
                            (VECTORIZER_PATH, "VECTORIZER_PATH"),
                            (FEATURE_NAMES_PATH, "FEATURE_NAMES_PATH")]:
        assert path_var is not None, f"config.{name} is not set"


def test_app_module_imports_cleanly():
    """app.py lives at the project root, not in src/ -- checked
    separately since it has its own import path and pulls in FastAPI."""
    try:
        import app  # noqa: F401
    except ImportError as e:
        pytest.fail(f"app.py failed to import: {e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))