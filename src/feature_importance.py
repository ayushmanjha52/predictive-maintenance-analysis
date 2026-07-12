"""
Feature Importance Analysis
Shows actual feature names (TF-IDF words + domain features)

FIX vs. previous version: the old version called prepare_modeling_data(df)
to "recreate features to get correct column names" -- but this REFITS a
new TF-IDF vectorizer from scratch, and filters rare classes AFTER fitting
rather than before (the reverse order training actually uses). Confirmed
by direct comparison: the saved model was trained on 1,183 features, the
recomputed version produced only 168 -- a totally different feature set,
not just reordered. Pairing model.feature_importances_ (indexed to the
real 1,183 training features) with 168 recomputed names would either
crash or silently mislabel every single importance value.

Fix: load the model, vectorizer, AND feature_names.pkl that were all
saved together at training time -- never recompute. This guarantees the
names line up with the importances index-for-index, because they're the
literal artifact from that training run.
"""

import joblib
import pandas as pd

from config import MODELS_DIR, REPORTS_DIR, MODEL_PATH, FEATURE_NAMES_PATH


def show_feature_importance(top_n=25):
    print("Loading saved model artifacts (not recomputing features)...")

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return None
    if not FEATURE_NAMES_PATH.exists():
        print(f"feature_names.pkl not found at {FEATURE_NAMES_PATH}. "
              f"This file is required -- it's saved by train_model.py alongside "
              f"the model and guarantees names match importances correctly. "
              f"Retrain first if it's missing.")
        return None

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    # Tree-based models (RandomForest, HistGradientBoosting) expose
    # feature_importances_. Linear models (LogisticRegression) expose
    # coef_ instead, with one row per class -- handle both rather than
    # assuming the model type.
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_kind = "feature_importances_ (tree-based, global)"
    elif hasattr(model, "coef_"):
        # Average absolute coefficient magnitude across classes as a
        # single importance proxy, since coef_ is (n_classes, n_features).
        importances = abs(model.coef_).mean(axis=0)
        importance_kind = "mean(|coef_|) across classes (linear model)"
    else:
        print(f"Model type {type(model).__name__} exposes neither "
              f"feature_importances_ nor coef_ -- cannot show importance.")
        return None

    # Hard check: if these lengths ever disagree, something upstream
    # changed (retrained with different features but didn't resave
    # feature_names.pkl, or loaded mismatched files) -- fail loudly
    # instead of silently mispairing names to values.
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Mismatch: model has {len(importances)} importances but "
            f"feature_names.pkl has {len(feature_names)} names. These MUST "
            f"be from the same training run -- retrain to regenerate both "
            f"files together, don't mix an old feature_names.pkl with a "
            f"newly retrained model."
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Type": ["text (TF-IDF)" if f.startswith("tfidf_") else "domain/engineered"
                 for f in feature_names],
    }).sort_values("Importance", ascending=False)

    print(f"\nImportance metric used: {importance_kind}")
    print("\n" + "=" * 70)
    print(f"           TOP {top_n} MOST IMPORTANT FEATURES")
    print("=" * 70)
    print(importance_df.head(top_n).to_string(index=False))

    # Quick summary: how much signal comes from text vs. hand-built
    # domain features -- useful for the report ("is this model mostly
    # reading raw text, or leaning on the domain expertise we encoded?")
    type_share = importance_df.groupby("Type")["Importance"].sum()
    type_share = (type_share / type_share.sum() * 100).round(1)
    print("\nImportance share by feature type:")
    for t, pct in type_share.items():
        print(f"  {t:20s} {pct}%")

    out_dir = REPORTS_DIR if REPORTS_DIR.exists() else MODELS_DIR
    out_path = out_dir / "feature_importance.csv"
    importance_df.to_csv(out_path, index=False)
    print(f"\nFeature importance saved to: {out_path}")

    return importance_df


if __name__ == "__main__":
    show_feature_importance(top_n=25)