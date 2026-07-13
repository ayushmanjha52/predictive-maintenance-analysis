"""
True Forward-in-Time Validation
Train on all months except the most recent, test on that unseen month.

FIXES vs. previous version -- both confirmed with real data:

1. CRITICAL: split relied on the 'date' column. Confirmed Feb-26's date
   column is 100% NaN in master_events.csv, so `df['date'] < '2026-05-01'`
   silently excluded all 92 Feb-26 rows from training (464 vs the correct
   561+ training rows). Fix: split on the existing 'month' label column,
   which doesn't depend on date parsing at all.

2. CRITICAL: prepare_modeling_data() was called separately on train_df
   and test_df, each with fit defaulting to True -- meaning the "test"
   features were built with a BRAND NEW vectorizer fit on May's 37 rows
   alone (57-word vocabulary) instead of reusing the training vectorizer
   (150-word vocabulary). Confirmed these are literally different
   objects with different vocabularies. Even though reindex() makes the
   column names line up, the underlying TF-IDF VALUES for any matching
   word were computed against completely different corpus statistics --
   this makes any F1 score this script has ever reported unreliable,
   not just "a bit off." Fix: fit the vectorizer ONCE on training data,
   then reuse that exact object (fit=False) for the test set.

3. No compound-label collapsing or rare-class handling -- same class of
   bug fixed in train.py. Added here too, for a fair comparison.

4. Test set restricted to classes actually seen in training. A model
   cannot possibly predict a class it's never seen; including such rows
   in the accuracy calculation penalizes the model for something no
   classifier could do, and understates genuine generalization ability.
   Rows with unseen classes are now reported separately, not silently
   averaged in.

5. Dynamically picks "the latest month" from config.MONTH_ORDER instead
   of hardcoding '2026-05-01'/'2026-06-01' -- this script would have
   silently gone stale the moment a June file was added, still only
   ever testing on May forever.

NOTE: this script's logic now duplicates what train.py's own
evaluate_holdout() does internally for every candidate model. Consider
whether you need this as a separate script at all, or whether checking
training_manifest.json (written by train.py) is sufficient -- keeping
both means two places that must stay in sync.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from src.config import REPORTS_DIR, RANDOM_STATE, MIN_CLASS_COUNT, MONTH_ORDER
from src.data_loader import load_master_events, primary_device
from src.feature_engineering import prepare_modeling_data


def get_labeled_data(min_samples_per_class=MIN_CLASS_COUNT):
    df = load_master_events()
    df["field_device"] = df["field_device"].apply(primary_device)
    df = df[df["field_device"].notna()].copy()
    counts = df["field_device"].value_counts()
    rare = counts[counts < min_samples_per_class].index.tolist()
    if rare:
        print(f"Dropping classes with <{min_samples_per_class} examples: {rare}")
    return df[~df["field_device"].isin(rare)].reset_index(drop=True)


def run_forward_time_validation():
    print("=" * 60)
    print("FORWARD-IN-TIME VALIDATION")
    print("=" * 60)

    df = get_labeled_data()
    months_present = df["month"].dropna().unique().tolist()
    ordered = [m for m in MONTH_ORDER if m in months_present]
    latest_month = ordered[-1] if ordered else sorted(months_present)[-1]
    print(f"Train: all months except {latest_month}  |  Test: {latest_month} only")

    # FIX: split on 'month' label, not the unreliable 'date' column.
    train_df = df[df["month"] != latest_month].reset_index(drop=True)
    test_df = df[df["month"] == latest_month].reset_index(drop=True)

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Testing samples:  {len(test_df)}")

    if len(test_df) == 0:
        print(f"No data found for {latest_month}. Check month labels in master_events.csv.")
        return None

    # FIX: fit vectorizer ONCE on training data only.
    train_model_df, vectorizer = prepare_modeling_data(train_df, fit=True)
    y_train = train_model_df["field_device"].astype(str)
    X_train = train_model_df.drop(columns=["field_device"], errors="ignore").select_dtypes(include=[np.number])

    model = RandomForestClassifier(
        n_estimators=430, max_depth=None, max_features="log2",
        min_samples_split=8, min_samples_leaf=1,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # FIX: reuse the SAME fitted vectorizer for the test set -- fit=False.
    test_model_df, _ = prepare_modeling_data(test_df, vectorizer=vectorizer, fit=False)
    y_test_full = test_model_df["field_device"].astype(str)
    X_test = test_model_df.drop(columns=["field_device"], errors="ignore").select_dtypes(include=[np.number])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # FIX: restrict to classes seen during training; report the rest separately.
    valid_mask = y_test_full.isin(y_train.unique())
    n_unseen = (~valid_mask).sum()
    if n_unseen > 0:
        unseen_classes = sorted(y_test_full[~valid_mask].unique())
        print(f"\n{n_unseen} test rows belong to classes never seen in training "
              f"({unseen_classes}) -- excluded from the accuracy calculation below "
              f"since no classifier could get these right. Reported separately.")

    y_test = y_test_full[valid_mask]
    X_test_valid = X_test[valid_mask]

    y_pred = model.predict(X_test_valid)

    print("\n" + "=" * 60)
    print(f"FORWARD-IN-TIME TEST RESULTS ({latest_month}, known classes only, n={len(y_test)})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = (y_pred == y_test.values).mean()
    print(f"Weighted F1 Score:  {weighted_f1:.4f}")
    print(f"Accuracy:           {accuracy:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(REPORTS_DIR / "forward_time_validation_report.csv")
    print(f"\nSaved to {REPORTS_DIR / 'forward_time_validation_report.csv'}")

    return {"weighted_f1": weighted_f1, "accuracy": accuracy, "holdout_month": latest_month,
            "n_test": len(y_test), "n_unseen_class_excluded": int(n_unseen)}


if __name__ == "__main__":
    run_forward_time_validation()