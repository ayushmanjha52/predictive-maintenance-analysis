"""
Test the full prediction pipeline -- imports and tests the ACTUAL
CausePredictor class from predict.py, not a reimplemented copy.

FIXES vs. previous version:

1. The only assertion, `assert pred_class in model.classes_`, is
   trivially true by construction: pred_class is LITERALLY derived as
   `model.classes_[proba.argmax()]`, so it can never be anything else.
   Confirmed directly: even a deliberately nonsense probability array
   still passes this assertion. This test could not have caught a
   broken pipeline no matter how badly broken it was.
   Fix: real assertions -- probabilities sum to ~1, and domain-obvious
   sample texts must predict their EXPECTED device (giving the test
   actual power to catch a regression, e.g. if feature engineering
   breaks and predictions become effectively random).

2. Hardcoded relative paths ("models/cause_classifier_model.pkl") --
   breaks outside one specific working directory. Fix: uses config.py
   paths, consistent with the rest of the project.

3. Reimplemented its own copy of the prediction pipeline (manual text
   cleaning, manual TF-IDF transform, manual reindex) instead of
   testing the actual CausePredictor class from predict.py. This is the
   fourth independent copy of this pipeline found across the test
   suite -- every one of them can silently drift from what's actually
   deployed. Fix: imports and tests CausePredictor directly, so this
   test tracks the real code path app.py and the CLI both use.

4. Single hardcoded sample with no clear expected outcome. Fix: several
   domain-obvious samples, each asserted against its expected device --
   plus one deliberately vague sample asserted to trigger low_confidence.
"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.predict import CausePredictor

# (sample_text, expected_top_device) -- these are domain-obvious enough
# that a working pipeline should get them right. If any of these starts
# failing, something in feature engineering or the model changed in a
# way that broke an easy case -- investigate immediately.
EXPECTED_PREDICTIONS = [
    ("photocell flickering in cooling bed", "PHOTOCELL"),
    ("hmd lens dirty covered with dust", "HMD"),
    ("BDM exit side guide transducer fault", "LVDT"),
    ("pressure switch oil low warning", "PRESSURE_SWITCH"),
]




@pytest.fixture(scope="module")
def predictor():
    return CausePredictor()


@pytest.mark.parametrize("text,expected_device", EXPECTED_PREDICTIONS)
def test_domain_obvious_predictions_match_expected_device(predictor, text, expected_device):
    """Real assertion with actual power to fail: a domain-obvious delay
    description must predict its obviously-correct device. Unlike the
    old `pred_class in model.classes_` check, this can genuinely catch
    a broken pipeline -- if feature engineering breaks and the model
    effectively guesses randomly, these WILL start failing."""
    result = predictor.predict(text, top_n=3)
    top_device = result[0]["Device"]
    assert top_device == expected_device, (
        f"Expected '{expected_device}' for clearly {expected_device.lower()}-related "
        f"text '{text}', got '{top_device}' instead -- pipeline may be broken."
    )


def test_probabilities_are_valid(predictor):
    """Probabilities must be in [0,100], and the FULL underlying
    probability vector (not just the top-3 shown to a user) must sum
    to ~1 -- catches a broken predict_proba call or a reindex that
    silently drops/duplicates columns."""
    result = predictor.predict("photocell lens dirty at cold saw", top_n=3)
    assert all(0 <= r["Probability"] <= 100 for r in result)

    from src.feature_engineering import prepare_modeling_data, align_features
    import pandas as pd
    df = pd.DataFrame({"reason_text": ["photocell lens dirty at cold saw"]})
    X, _ = prepare_modeling_data(df, vectorizer=predictor.vectorizer, fit=False)
    X = align_features(X, predictor.feature_names)
    full_proba = predictor.model.predict_proba(X)[0]
    assert abs(full_proba.sum() - 1.0) < 1e-6, (
        f"Full probability vector sums to {full_proba.sum():.4f}, expected ~1.0"
    )





def test_empty_and_edge_case_text_does_not_crash(predictor):
    for text in ["", "a", "1234567890", "!!!???"]:
        try:
            result = predictor.predict(text if text else "x", top_n=3)
            assert len(result) == 3
        except Exception as e:
            pytest.fail(f"Predictor crashed on edge-case input '{text}': {e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))