"""
Feature Engineering for Combi Mill Field Device PdM
Creates both text features and domain-specific features.

FIX vs. previous version: create_text_features() used to call
vectorizer.fit_transform() unconditionally -- meaning every call, even
on a single new row at prediction time, tried to FIT a brand new
vocabulary from scratch. With min_df=3 this doesn't just give wrong
features, it crashes outright (< 3 documents available to satisfy
min_df). Confirmed by reproducing the crash directly:

    ValueError: max_df corresponds to < documents than min_df

Fix: fit=True/False now controls fit_transform vs. transform explicitly.
Train once with fit=True and save the vectorizer; every later call
(prediction, retraining eval, tests) passes fit=False and reuses it.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_text_features(df, text_col="reason_text", vectorizer=None, fit=True,
                          max_features=150, min_df=2):
    """Create TF-IDF features from delay descriptions.

    fit=True  (training): fits a NEW vectorizer on df[text_col] and
              returns (df_with_features, fitted_vectorizer).
    fit=False (inference/eval): REUSES the vectorizer you pass in via
              `vectorizer` -- must be the one saved from training.
              Raises if vectorizer is None, so this can't silently
              fall back to fitting-on-one-row like the previous bug.

    min_df default lowered from 3 -> 2: with several classes at n=6-11
    events (Proximity_Switch, Flow_Switch, Laser), min_df=3 risks
    filtering out terms that are informative specifically because
    they're rare and class-specific. Revisit if vocabulary noise
    becomes a problem as more months are added.
    """
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            min_df=min_df, stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    else:
        if vectorizer is None:
            raise ValueError(
                "fit=False requires a trained `vectorizer` to be passed in "
                "(load it with joblib.load(VECTORIZER_PATH)). Refusing to "
                "silently fit a new one on this data -- that was the bug "
                "that crashed single-row prediction before."
            )
        tfidf_matrix = vectorizer.transform(df["clean_text"])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()],
        index=df.index
    )
    return pd.concat([df, tfidf_df], axis=1), vectorizer


def create_domain_features(df):
    """
    Create domain-specific features based on known failure patterns.

    FIX: now computes clean_text itself if it isn't already present,
    so this function is safe to call standalone (e.g. directly on a
    single prediction row) instead of crashing on df.get("clean_text", "")
    -- the original fallback returned a plain string "" which has no
    .str accessor and would raise AttributeError the moment this ran
    without create_text_features having been called first.
    """
    df = df.copy()
    if "clean_text" not in df.columns:
        df["clean_text"] = df["reason_text"].apply(clean_text) if "reason_text" in df.columns else ""
    text = df["clean_text"].astype(str)

    # === PHOTOCELL ===
    df["photocell_flicker"] = text.str.contains("flicker|flickering|high|not sensing", regex=True).astype(int)
    df["photocell_contamination"] = text.str.contains("cleaning|dust|scale|dirty|lens", regex=True).astype(int)
    df["photocell_sequence_break"] = text.str.contains("sequence break|seq break|auto sequence", regex=True).astype(int)

    # === ENCODER ===
    df["encoder_feedback"] = text.str.contains("feedback|encoder feedback|position feedback", regex=True).astype(int)
    df["encoder_overtravel"] = text.str.contains("overtravel|over travelled|overtravelled", regex=True).astype(int)
    df["encoder_coupling"] = text.str.contains("coupling|loose|slip", regex=True).astype(int)
    df["encoder_stand"] = text.str.contains("stand", regex=True).astype(int)
    df["encoder_fault"] = text.str.contains("encoder fault|encoder error|encoder problem", regex=True).astype(int)
    df["encoder_missing"] = text.str.contains("missing|not found|no feedback", regex=True).astype(int)

    # === LVDT ===
    df["lvdt_transducer"] = text.str.contains("transducer|lvdt|position|guide|exit side", regex=True).astype(int)
    df["lvdt_stuck"] = text.str.contains("stuck|not moving|feedback missing|transducer fault", regex=True).astype(int)

    # === HMD ===
    df["hmd_cleaning"] = text.str.contains("hmd cleaning|od1 cleaning|cleaning in bdm", regex=True).astype(int)
    df["hmd_continuous"] = text.str.contains("continuous sensing|continous sensing|high after", regex=True).astype(int)
    df["hmd_flicker"] = text.str.contains("flickering|od.*flicker", regex=True).astype(int)

    # === General Context ===
    df["is_cobble"] = text.str.contains("cobble|autochopped|chopped", regex=True).astype(int)
    df["is_hot_out"] = text.str.contains("hot out|hotout", regex=True).astype(int)
    df["is_rejected"] = text.str.contains("rejected|billet rejected", regex=True).astype(int)
    df["involves_bdm"] = text.str.contains("bdm|billet|stand", regex=True).astype(int)

    return df


def align_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reindex X to the exact training-time feature set/order, filling
    any missing column with 0. Call this right before model.predict at
    inference time -- guarantees train/predict parity regardless of
    what a single new row happens to produce."""
    return X.reindex(columns=feature_names, fill_value=0)


def prepare_modeling_data(df, target_col="field_device", vectorizer=None, fit=True):
    """
    Full feature engineering pipeline.

    TRAINING call:   prepare_modeling_data(train_df, fit=True)
                     -> (df_with_features_and_target, fitted_vectorizer)
                     Save the returned vectorizer with joblib.

    INFERENCE call:  prepare_modeling_data(single_row_df, vectorizer=saved_vec, fit=False)
                     -> (df_with_features, same_vectorizer)
                     `field_device` need not be present for inference input.
    """
    if fit:
        df_labeled = df[df[target_col].notna()].copy() if target_col in df.columns else df.copy()
        print(f"Using {len(df_labeled)} labeled events for modeling.")
    else:
        df_labeled = df.copy()

    df_labeled, vectorizer = create_text_features(df_labeled, vectorizer=vectorizer, fit=fit)
    df_labeled = create_domain_features(df_labeled)

    cols_to_drop = ["reason_text", "clean_text", "source_file", "tag_source"]
    df_labeled = df_labeled.drop(columns=[c for c in cols_to_drop if c in df_labeled.columns], errors="ignore")

    return df_labeled, vectorizer


if __name__ == "__main__":
    print("Feature engineering module is ready.")
    print("Training:  prepare_modeling_data(df, fit=True)")
    print("Inference: prepare_modeling_data(single_row_df, vectorizer=saved_vec, fit=False)")