"""
Prediction Script — CLI + shared CausePredictor class.

FIXES vs. previous version:

1. Model filename ambiguity (real risk, confirmed): this script hardcoded
   "final_random_forest_model.pkl" while app.py (reviewed earlier)
   hardcoded "cause_classifier_model.pkl" -- two different scripts each
   independently deciding which model file is "the" deployed model, with
   nothing forcing them to agree. If you retrain and only one script's
   target file gets updated, predict.py and the API would silently give
   different answers to the identical input text.
   Fix: single MODEL_PATH from config.py, used everywhere. If your
   actual best model is saved under a different name, either rename the
   file to match config.MODEL_PATH or update config.py once -- not each
   script separately.

2. clean_text() was redefined here, duplicating the one already in
   feature_engineering.py. Same class of bug flagged in app.py's review:
   if cleaning logic is ever tweaked, this copy silently goes stale.
   Fix: import the single shared implementation.

3. No confidence threshold -- a 12%-confidence top guess was presented
   exactly the same as an 85%-confidence one. Fix: reuses
   config.CONFIDENCE_THRESHOLD and flags low_confidence explicitly, same
   behavior as the API, via one shared CausePredictor class.

4. No error handling for missing model files -- crashed with a raw
   FileNotFoundError. Fix: clear message pointing at train_model.py.

5. CLI-only (interactive input() loop) -- no way to call this from a
   script/test without spawning stdin interaction. Fix: interactive mode
   is preserved, but predict_device() is also usable as a plain function
   / CausePredictor as an importable class (this is what app.py should
   import FROM here, instead of maintaining its own separate copy of the
   same pipeline).
"""
import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from config import MODEL_PATH, VECTORIZER_PATH, FEATURE_NAMES_PATH, CONFIDENCE_THRESHOLD
from feature_engineering import clean_text, create_domain_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CausePredictor:
    """Single shared implementation of the prediction pipeline.
    app.py should import and use THIS class rather than re-implementing
    its own copy -- that duplication is exactly what let app.py and
    predict.py load two different model files without either script
    knowing the other existed."""

    def __init__(self):
        for path, label in [(MODEL_PATH, "model"), (VECTORIZER_PATH, "vectorizer"),
                             (FEATURE_NAMES_PATH, "feature_names")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} not found at {path}. Run train_model.py first, or "
                    f"if your trained model is saved under a different filename, "
                    f"update config.MODEL_PATH (etc.) to point at it -- don't "
                    f"hardcode a different path in this script."
                )
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info(f"Loaded model={MODEL_PATH.name}, "
                    f"vectorizer={VECTORIZER_PATH.name}, "
                    f"{len(self.feature_names)} features")

    def predict(self, delay_text: str, top_n: int = 3) -> list:
        df = pd.DataFrame({"reason_text": [delay_text]})
        df["clean_text"] = df["reason_text"].apply(clean_text)

        tfidf_matrix = self.vectorizer.transform(df["clean_text"])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{feat}" for feat in self.vectorizer.get_feature_names_out()]
        )
        df = pd.concat([df, tfidf_df], axis=1)
        df = create_domain_features(df)

        X = df.select_dtypes(include=[np.number])
        X = X.reindex(columns=self.feature_names, fill_value=0)

        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        top_indices = np.argsort(proba)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            results.append({"Device": classes[idx], "Probability": round(float(proba[idx]) * 100, 2)})

        low_confidence = results[0]["Probability"] / 100 < CONFIDENCE_THRESHOLD
        for r in results:
            r["low_confidence"] = low_confidence
        return results


def predict_device(delay_text: str, predictor: CausePredictor = None, top_n: int = 3) -> list:
    """Convenience function for scripting/tests: predict_device(text, predictor)."""
    if predictor is None:
        predictor = CausePredictor()
    return predictor.predict(delay_text, top_n=top_n)


def interactive_loop():
    print("Loading model...")
    try:
        predictor = CausePredictor()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    print("Model loaded successfully!\n")

    print("=== Combi Mill Delay Cause Predictor ===")
    print("Type a delay description (or type 'exit' to quit)\n")

    while True:
        text = input("Enter delay description: ").strip()
        if text.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break
        if not text:
            continue

        predictions = predictor.predict(text, top_n=3)
        print("\nTop Predictions:")
        for i, pred in enumerate(predictions, 1):
            flag = "  [LOW CONFIDENCE]" if pred["low_confidence"] and i == 1 else ""
            print(f"  {i}. {pred['Device']:<25} -> {pred['Probability']}%{flag}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Non-interactive single-shot mode: python predict.py "delay text here"
        text = " ".join(sys.argv[1:])
        predictor = CausePredictor()
        for i, pred in enumerate(predictor.predict(text), 1):
            print(f"{i}. {pred['Device']:<25} -> {pred['Probability']}%")
    else:
        interactive_loop()