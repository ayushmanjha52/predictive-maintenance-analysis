"""
Combi Mill Cause Classifier — Single Canonical Training Script

This REPLACES train.py, train_final_model.py, train_hist_gradient_boosting.py,
and train_catboost.py (if you have it). Delete those after confirming this
works -- having multiple training scripts that each independently overwrite
the shared tfidf_vectorizer.pkl / feature_names.pkl files is what caused the
"model has 164 importances but feature_names.pkl has 168 names" error hit
earlier: whichever script ran LAST silently won, with no way to tell which
model.pkl file actually matched the feature_names.pkl on disk.

KEY DESIGN DECISION: model selection is done on TIME-BASED HOLDOUT accuracy
(train on all months except the latest, test only on that unseen month),
NOT cross-validation accuracy. This project already proved CV can overstate
real-world performance by 25 points on this exact data (76.5% CV vs 51.4%
true holdout for an early Random Forest). Picking the "best" model by CV
alone risks picking the model that's best at fitting folds of similar
months, not the one that generalizes to a genuinely new month -- which is
the actual production use case every month going forward.

FIXES carried over from prior review of the individual scripts:
  - Compound labels (e.g. "HMD_/_LVDT") collapsed to primary device BEFORE
    anything else -- proven to recover real training examples and improve
    holdout accuracy by ~11 points on this dataset.
  - Rare classes (<MIN_CLASS_COUNT) dropped explicitly -- the old
    hist_gradient script (like train_final_model.py before it) didn't do
    this, letting a 1-example class ("TT") silently corrupt StratifiedKFold.
  - ONE shared output filename set -- no more scripts fighting over the
    same tfidf_vectorizer.pkl / feature_names.pkl with different models.
  - training_manifest.json records EVERY candidate's CV and holdout
    accuracy, plus which one won and why -- full audit trail in one file.
  - Hard consistency assertion before saving (feature count must match).
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, MIN_CLASS_COUNT, MONTH_ORDER
from data_loader import load_master_events, primary_device
from feature_engineering import prepare_modeling_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All candidate models to compare. Add/remove here -- this list is the
# single place that defines "what we consider training," replacing what
# used to be 3-4 separate scripts.
CANDIDATE_MODELS = {
    "random_forest_tuned": RandomForestClassifier(
        n_estimators=430, max_depth=None, max_features="log2",
        min_samples_split=8, min_samples_leaf=1,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=10,
        min_samples_leaf=20, random_state=RANDOM_STATE, class_weight="balanced",
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced",
    ),
}

try:
    from catboost import CatBoostClassifier
    CANDIDATE_MODELS["catboost"] = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=False, auto_class_weights="Balanced",
    )
except ImportError:
    logger.info("catboost not installed -- skipping (pip install catboost to include it)")


def get_labeled_data(min_samples_per_class=MIN_CLASS_COUNT):
    df = load_master_events()
    df["field_device"] = df["field_device"].apply(primary_device)
    df = df[df["field_device"].notna()].copy()

    class_counts = df["field_device"].value_counts()
    rare_classes = class_counts[class_counts < min_samples_per_class].index.tolist()
    if rare_classes:
        logger.info(f"Dropping classes with <{min_samples_per_class} examples: {rare_classes}")
    df = df[~df["field_device"].isin(rare_classes)].reset_index(drop=True)
    return df


def get_latest_month(months_present):
    ordered = [m for m in MONTH_ORDER if m in months_present]
    return ordered[-1] if ordered else sorted(months_present)[-1]


def evaluate_holdout(model, df, latest_month):
    """Train fresh on all-but-latest-month, test only on the unseen month."""
    train_df = df[df["month"] != latest_month].reset_index(drop=True)
    test_df = df[df["month"] == latest_month].reset_index(drop=True)
    if len(test_df) < 5:
        return None

    X_train, vec = prepare_modeling_data(train_df, fit=True)
    y_train = X_train["field_device"]
    X_train = X_train.drop(columns=["field_device"], errors="ignore").select_dtypes(include=[np.number])

    X_test, _ = prepare_modeling_data(test_df, vectorizer=vec, fit=False)
    y_test = test_df["field_device"]
    X_test = X_test.drop(columns=["field_device"], errors="ignore").select_dtypes(include=[np.number])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    valid_mask = y_test.isin(y_train.unique())
    if valid_mask.sum() == 0:
        return None

    model_clone = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
    model_clone.fit(X_train, y_train)
    preds = model_clone.predict(X_test[valid_mask])
    report = classification_report(y_test[valid_mask], preds, zero_division=0, output_dict=True)
    return {"accuracy": report["accuracy"], "n_test": int(valid_mask.sum()), "report": report}


def compare_all_models(df, X, y):
    """Runs CV + honest holdout for every candidate, logs both, and
    selects the winner by HOLDOUT accuracy (not CV -- see module docstring)."""
    months_present = df["month"].dropna().unique().tolist()
    latest_month = get_latest_month(months_present) if "month" in df.columns else None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    logger.info("\n=== Comparing candidate models (CV accuracy AND honest holdout) ===")
    for name, model in CANDIDATE_MODELS.items():
        try:
            y_pred_cv = cross_val_predict(model, X, y, cv=skf)
            cv_report = classification_report(y, y_pred_cv, zero_division=0, output_dict=True)
            cv_acc = cv_report["accuracy"]
        except Exception as e:
            logger.warning(f"  {name}: CV failed ({e}) -- skipping this candidate")
            continue

        holdout = evaluate_holdout(model, df, latest_month) if latest_month else None
        holdout_acc = holdout["accuracy"] if holdout else None

        results[name] = {"cv_accuracy": cv_acc, "holdout_accuracy": holdout_acc,
                          "cv_report": cv_report, "holdout_report": holdout["report"] if holdout else None}
        logger.info(f"  {name:24s} CV={cv_acc:.3f}   Holdout({latest_month})={holdout_acc}")

    if not results:
        raise RuntimeError("No candidate model could be evaluated -- check data/config.")

    # Select by holdout accuracy when available (the honest metric);
    # fall back to CV only if holdout couldn't be computed for anyone.
    have_holdout = {k: v for k, v in results.items() if v["holdout_accuracy"] is not None}
    if have_holdout:
        winner_name = max(have_holdout, key=lambda k: have_holdout[k]["holdout_accuracy"])
        logger.info(f"\nWinner selected by HOLDOUT accuracy (the honest metric): {winner_name}")
    else:
        winner_name = max(results, key=lambda k: results[k]["cv_accuracy"])
        logger.warning(f"\nNo holdout available for any candidate -- winner selected by CV only "
                        f"(less reliable): {winner_name}")

    return winner_name, CANDIDATE_MODELS[winner_name], results, latest_month


def train_and_evaluate(min_samples_per_class=MIN_CLASS_COUNT):
    logger.info("Loading data...")
    df = get_labeled_data(min_samples_per_class)

    logger.info("Creating features...")
    df_model, vectorizer = prepare_modeling_data(df, fit=True)
    y = df_model["field_device"].astype(str)
    X = df_model.drop(columns=["field_device"], errors="ignore").select_dtypes(include=[np.number])
    logger.info(f"Training on {X.shape[1]} features with {len(X)} samples, {y.nunique()} classes.")

    winner_name, winner_model, all_results, latest_month = compare_all_models(df, X, y)
    winner_result = all_results[winner_name]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(winner_result["cv_report"]).transpose().to_csv(REPORTS_DIR / "classification_report.csv")
    if winner_result["holdout_report"]:
        pd.DataFrame(winner_result["holdout_report"]).transpose().to_csv(REPORTS_DIR / "holdout_report.csv")

    logger.info(f"\n=== Winning model ({winner_name}) CV report ===")
    print(classification_report(y, cross_val_predict(winner_model, X, y,
          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)), zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y, cross_val_predict(winner_model, X, y,
         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)), labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (5-Fold CV) — {winner_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    logger.info(f"Training final '{winner_name}' model on full data...")
    winner_model.fit(X, y)

    feature_names = X.columns.tolist()
    n_model_features = getattr(winner_model, "n_features_in_", len(feature_names))
    if n_model_features != len(feature_names):
        raise RuntimeError(
            f"Refusing to save: model trained on {n_model_features} features but "
            f"feature_names has {len(feature_names)} entries."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_model, MODELS_DIR / "cause_classifier_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    logger.info(f"Saved winning model ({winner_name}) + vectorizer + "
                f"{len(feature_names)} feature names to {MODELS_DIR}")

    manifest = {
        "trained_at": datetime.now().isoformat(),
        "winner": winner_name,
        "n_features": len(feature_names),
        "n_samples": len(X),
        "n_classes": int(y.nunique()),
        "classes": sorted(y.unique().tolist()),
        "holdout_month": latest_month,
        "all_candidates": {
            name: {"cv_accuracy": round(r["cv_accuracy"], 4),
                   "holdout_accuracy": round(r["holdout_accuracy"], 4) if r["holdout_accuracy"] else None}
            for name, r in all_results.items()
        },
    }
    with open(MODELS_DIR / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote training_manifest.json with full comparison audit trail.")
    logger.info(f"\nFINAL SUMMARY: winner={winner_name}  "
                f"CV={winner_result['cv_accuracy']:.3f}  "
                f"Holdout={winner_result['holdout_accuracy']}")

    return winner_model


if __name__ == "__main__":
    train_and_evaluate()