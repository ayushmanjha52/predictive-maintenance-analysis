"""
Critical Test: Ensures features created at prediction time match what the
DEPLOYED model on disk actually expects.

FIXES vs. previous version:

1. `create_domain_features` was only imported inside `if __name__ ==
   "__main__":` -- meaning this test crashes with NameError the moment
   it's run the normal way (`pytest test_feature_consistency.py`), which
   is presumably how a file living in tests/ is meant to run. Confirmed
   directly: running the exact same logic under non-__main__ execution
   raises `NameError: name 'create_domain_features' is not defined`.
   Fix: import at module level like everything else.

2. CRITICAL GAP: the old test only compared two freshly-recomputed
   in-memory objects (a vectorizer fit moments ago, against itself) --
   it never touched the actual model.pkl / feature_names.pkl saved on
   disk. This means it would NOT have caught the real bug hit earlier
   in this project ("model has 164 importances but feature_names.pkl
   has 168 names") -- that bug was a mismatch between the DEPLOYED
   artifacts, which this test structurally can't see since it never
   loads them.
   Fix: loads the actual saved model/vectorizer/feature_names from
   MODELS_DIR and verifies THOSE are consistent with each other and
   with a live prediction -- this is the test that would have actually
   caught the earlier bug.

3. Only checked for MISSING features, never asserted on EXTRA features
   (computed `extra_in_pred` but never asserted on it). Both are
   real risks: missing features silently zero-fill via reindex (usually
   fine), but the test should still confirm reindex handles it, and
   verify ORDER matches too, since some estimators are sensitive to
   column order even with matching names.

4. Single sample text only -- added a few more, including short/vague
   text and text with only unknown vocabulary, to catch edge cases.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODEL_PATH, VECTORIZER_PATH, FEATURE_NAMES_PATH
from src.feature_engineering import clean_text, create_domain_features, align_features

SAMPLE_TEXTS = [
    "encoder feedback missing at stand 3",
    "photocell lens dirty cleaning required",
    "delay occurred",  # deliberately vague -- edge case
    "xyzabc unknown vocabulary term qwerty",  # no real vocabulary matches
    "",  # empty string edge case
]


@pytest.fixture(scope="module")
def deployed_artifacts():
    """Loads the ACTUAL artifacts currently deployed on disk -- this is
    the fix. Testing anything else can't catch a stale-file mismatch."""
    for path, label in [(MODEL_PATH, "model"), (VECTORIZER_PATH, "vectorizer"),
                         (FEATURE_NAMES_PATH, "feature_names")]:
        assert path.exists(), f"{label} not found at {path} -- train the model first."
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, vectorizer, feature_names


def _build_prediction_features(text, vectorizer):
    """Mirrors the real prediction pipeline (same functions predict.py
    uses) -- NOT a separately reimplemented copy, so this test tracks
    the actual deployed pipeline instead of a parallel version that
    can silently drift from it."""
    df = pd.DataFrame({"reason_text": [text]})
    df["clean_text"] = df["reason_text"].apply(clean_text)
    tfidf_matrix = vectorizer.transform(df["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
    )
    df = pd.concat([df, tfidf_df], axis=1)
    df = create_domain_features(df)
    return df.select_dtypes(include=[np.number])


def test_model_and_feature_names_agree_on_count(deployed_artifacts):
    """This is the exact check that would have caught the earlier real
    bug: model.pkl and feature_names.pkl must describe the same number
    of features. If they don't, they're from different training runs."""
    model, vectorizer, feature_names = deployed_artifacts
    n_model_features = (model.n_features_in_ if hasattr(model, "n_features_in_")
                         else len(feature_names))
    assert n_model_features == len(feature_names), (
        f"Model expects {n_model_features} features but feature_names.pkl "
        f"has {len(feature_names)} -- these artifacts are from different "
        f"training runs. Retrain to regenerate both together."
    )


@pytest.mark.parametrize("sample_text", SAMPLE_TEXTS)
def test_prediction_features_align_to_deployed_model(deployed_artifacts, sample_text):
    """For each sample input, build features the same way the real
    prediction pipeline does, align to feature_names.pkl, and confirm
    BOTH the column set AND order match exactly, then confirm the
    deployed model can actually score it without error."""
    model, vectorizer, feature_names = deployed_artifacts
    X_new = _build_prediction_features(sample_text, vectorizer)
    X_aligned = align_features(X_new, feature_names)

    assert list(X_aligned.columns) == feature_names, (
        "align_features() output does not match feature_names order/set exactly."
    )
    assert X_aligned.shape[1] == len(feature_names)

    # The real test: does the deployed model actually accept this input?
    try:
        proba = model.predict_proba(X_aligned)
    except Exception as e:
        pytest.fail(f"Deployed model rejected aligned features for "
                     f"input '{sample_text}': {e}")
    assert proba.shape[1] == len(model.classes_)


def test_extra_features_are_dropped_not_passed_through(deployed_artifacts):
    """Simulates the scenario where prediction-time feature building
    produces an extra column the model never saw in training (e.g. a
    new domain feature added without retraining) -- confirms
    align_features() drops it rather than passing it through and
    breaking the model."""
    model, vectorizer, feature_names = deployed_artifacts
    X_new = _build_prediction_features("test input", vectorizer)
    X_new["totally_new_unseen_feature"] = 1  # simulate drift
    X_aligned = align_features(X_new, feature_names)
    assert "totally_new_unseen_feature" not in X_aligned.columns
    assert list(X_aligned.columns) == feature_names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))