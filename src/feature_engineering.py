"""
Feature Engineering for Combi Mill Field Device PdM
Creates both text features and domain-specific features
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import re


def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_text_features(df, text_col="reason_text", max_features=150):
    """Create TF-IDF features from delay descriptions."""
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()],
        index=df.index
    )
    
    return pd.concat([df, tfidf_df], axis=1), vectorizer


def create_domain_features(df):
    """
    Create domain-specific features based on known failure patterns
    from FMEA and operational knowledge.
    """
    df = df.copy()
    text = df.get("reason_text", "").astype(str).str.lower()
    
    # === PHOTOCELL ===
    df["photocell_flicker"] = text.str.contains("flicker|flickering|high|not sensing|detection fault", regex=True).astype(int)
    df["photocell_contamination"] = text.str.contains("cleaning|dust|scale|dirty|lens", regex=True).astype(int)
    df["photocell_sequence_break"] = text.str.contains("sequence break|seq break|auto sequence", regex=True).astype(int)
    
    # === ENCODER ===
    df["encoder_coupling"] = text.str.contains("coupling|loose|slip|encoder fault|position", regex=True).astype(int)
    df["encoder_overtravel"] = text.str.contains("overtravel|over travelled|overtravelled", regex=True).astype(int)
    
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


def prepare_modeling_data(df, target_col="field_device"):
    """
    Full feature engineering pipeline.
    Returns dataframe ready for modeling + the TF-IDF vectorizer.
    """
    # Keep only rows with known field_device (labeled data)
    df_labeled = df[df[target_col].notna()].copy()
    
    print(f"Using {len(df_labeled)} labeled events for modeling.")
    
    # Create text features
    df_labeled, vectorizer = create_text_features(df_labeled)
    
    # Create domain features
    df_labeled = create_domain_features(df_labeled)
    
    # Drop unnecessary columns
    cols_to_drop = ["reason_text", "clean_text", "source_file", "tag_source"]
    df_labeled = df_labeled.drop(columns=[c for c in cols_to_drop if c in df_labeled.columns], errors="ignore")
    
    return df_labeled, vectorizer


if __name__ == "__main__":
    print("Feature engineering module is ready.")
    print("Use prepare_modeling_data(df) to generate features.")