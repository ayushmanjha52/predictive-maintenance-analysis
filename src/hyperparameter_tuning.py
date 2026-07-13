"""
Hyperparameter Search for Combi Mill PdM Model

STATUS: this is a PERIODIC EXPLORATION TOOL, not part of the automatic
train.py pipeline. Run it manually when you want to re-search for better
hyperparameters (e.g. after adding a few more months of data), review
its output, then manually update the RandomForest config inside
CANDIDATE_MODELS in train.py if the new params genuinely improve the
HONEST holdout accuracy -- don't copy them in on CV score alone (see
fix #1 below for why).

FIXES vs. previous version:

1. CRITICAL, confirmed with real numbers: the printed "Classification
   Report" evaluated best_estimator_.predict(X) against the SAME X, y
   the model was just refit on (RandomizedSearchCV refits on the full
   data by default). This reports training-set performance dressed up
   as a real evaluation. Confirmed directly on this data:
     CV best_score_ (honest):        0.5935
     Training-set "accuracy" (misleading): 0.7977
   A ~20-point gap, in the wrong direction (looks BETTER than reality).
   Fix: reports the honest CV score as the headline number, and adds a
   genuine time-based holdout check (train on all-but-latest-month,
   test only on that month) -- never predict-on-training-data again.

2. No compound-label collapsing or rare-class filtering -- same class of
   bug fixed in train.py, and confirmed here too (StratifiedKFold warns
   about the 1-example "TT" class). Fixed the same way: primary_device()
   collapsing + dropping classes below MIN_CLASS_COUNT before anything else.

3. Saved to yet ANOTHER model filename ("best_random_forest_model.pkl")
   -- a fifth artifact competing with cause_classifier_model.pkl,
   final_random_forest_model.pkl, and hist_gradient_model.pkl from
   earlier in this project, none of which app.py/predict.py actually
   load. Fix: this script no longer saves a competing model.pkl at all.
   It saves ONLY the discovered best_hyperparameters as a JSON file --
   a human (or a future automated step) then decides whether to merge
   them into train.py's CANDIDATE_MODELS. This ends the "which .pkl is
   actually deployed" ambiguity for good.
"""
import json
import logging
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from scipy.stats import randint

from config import MODELS_DIR, RANDOM_STATE, MIN_CLASS_COUNT, MONTH_ORDER
from data_loader import load_master_events, primary_device
from feature_engineering import prepare_modeling_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_labeled_data(min_samples_per_class=MIN_CLASS_COUNT):
    df = load_master_events()
    df["field_device"] = df["field_device"].apply(primary_device)
    df = df[df["field_device"].notna()].copy()
    counts = df["field_device"].value_counts()
    rare = counts[counts < min_samples_per_class].index.tolist()
    if rare:
        logger.info(f"Dropping classes with <{min_samples_per_class} examples: {rare}")
    return df[~df["field_device"].isin(rare)].reset_index(drop=True)


def honest_holdout_check(best_params, df):
    """Train fresh with the discovered params on all-but-latest-month,
    test ONLY on that unseen month -- this is the real question:
    do these hyperparameters generalize, not just fit CV folds well."""
    months_present = df["month"].dropna().unique().tolist()
    ordered = [m for m in MONTH_ORDER if m in months_present]
    latest_month = ordered[-1] if ordered else sorted(months_present)[-1]

    train_df = df[df["month"] != latest_month].reset_index(drop=True)
    test_df = df[df["month"] == latest_month].reset_index(drop=True)
    if len(test_df) < 5:
        logger.warning("Too few holdout-month events -- skipping honest holdout check.")
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

    model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test[valid_mask])
    report = classification_report(y_test[valid_mask], preds, zero_division=0, output_dict=True)
    return {"holdout_month": latest_month, "accuracy": report["accuracy"], "n_test": int(valid_mask.sum())}


def tune_hyperparameters(n_iter=30):
    logger.info("Loading data and preparing features...")
    df = get_labeled_data()
    df_model, vectorizer = prepare_modeling_data(df, fit=True)

    y = df_model["field_device"].astype(str)
    X = df_model.drop(columns=["field_device"], errors="ignore").select_dtypes(include=[np.number])
    logger.info(f"Data shape: {X.shape}, {y.nunique()} classes")

    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    logger.info(f"Starting RandomizedSearchCV ({n_iter} iterations, this may take a while)...")
    random_search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_dist, n_iter=n_iter,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted", verbose=1, random_state=RANDOM_STATE, n_jobs=-1,
    )
    random_search.fit(X, y)

    logger.info("=" * 60)
    logger.info(f"BEST HYPERPARAMETERS: {random_search.best_params_}")
    # FIX: this CV score IS honest (unlike the old predict-on-training-data
    # report) -- it's the average across held-out folds, never seen during
    # that fold's fit.
    logger.info(f"Best CV-weighted-F1 (honest, held-out folds): {random_search.best_score_:.4f}")

    # FIX: additional genuine holdout check, the real question that matters
    # for production trust -- do these params generalize to a truly unseen month.
    holdout = honest_holdout_check(random_search.best_params_, df)
    if holdout:
        logger.info(f"Honest holdout accuracy on unseen '{holdout['holdout_month']}' "
                     f"(n={holdout['n_test']}): {holdout['accuracy']:.4f}")
        logger.info("Compare this to train.py's training_manifest.json holdout_accuracy "
                     "for the currently-deployed model before deciding whether to merge "
                     "these new params in -- a higher CV score with a LOWER holdout score "
                     "is not actually an improvement.")

    # FIX: save ONLY the discovered parameters + both scores as JSON --
    # no competing model.pkl file. A human reviews this and decides
    # whether to merge the params into train.py's CANDIDATE_MODELS.
    output = {
        "searched_at": datetime.now().isoformat(),
        "best_params": random_search.best_params_,
        "cv_weighted_f1": round(random_search.best_score_, 4),
        "holdout_accuracy": round(holdout["accuracy"], 4) if holdout else None,
        "holdout_month": holdout["holdout_month"] if holdout else None,
        "note": ("These are candidate parameters, NOT a deployed model. Compare "
                 "holdout_accuracy against train.py's training_manifest.json before "
                 "merging into CANDIDATE_MODELS in train.py."),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "hyperparameter_search_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved candidate params + both honest scores to {out_path}")
    logger.info("No model.pkl was saved by this script -- update train.py's "
                 "CANDIDATE_MODELS manually if these params genuinely improve "
                 "on the current holdout_accuracy, then re-run train.py.")

    return output


if __name__ == "__main__":
    tune_hyperparameters()