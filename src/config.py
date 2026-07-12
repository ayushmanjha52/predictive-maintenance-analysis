"""
Configuration file for Field Device Predictive Maintenance Project
"""

import os
from pathlib import Path

# ====================== PROJECT PATHS ======================
# Fixed absolute path breaks the moment this repo moves, gets shared with
# your guide, or runs on a different machine/CI runner. Resolve relative
# to this file instead -- config.py lives in src/, so parent.parent is
# the project root. An env var override is kept for the rare case you
# really do need to point elsewhere (e.g. a shared drive at the plant).
PROJECT_ROOT = Path(os.environ.get("PDM_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR      = PROJECT_ROOT / "data"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Main unified data file
MASTER_EVENTS_PATH = DATA_DIR / "master_events.csv"

# Trained artifacts
MODEL_PATH          = MODELS_DIR / "cause_classifier_model.pkl"
VECTORIZER_PATH     = MODELS_DIR / "tfidf_vectorizer.pkl"
FEATURE_NAMES_PATH  = MODELS_DIR / "feature_names.pkl"

# Evaluation reports
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH      = REPORTS_DIR / "confusion_matrix.csv"
HOLDOUT_REPORT_PATH        = REPORTS_DIR / "holdout_report.csv"

# ====================== KEY DEVICES ======================
# Curated priority list for reporting/dashboard emphasis (Pareto leaders +
# FMEA-flagged risk). This is NOT the same as "classes the model trains
# on" -- that's derived automatically from whatever has >= MIN_CLASS_COUNT
# labeled examples, which also includes FLOW_SWITCH, LASER, PROXIMITY_SWITCH.
# Keep this list for report/dashboard framing only; don't use it to filter
# training data or you'll silently drop valid classes.
CORE_DEVICES = [
    "PHOTOCELL",
    "ENCODER",
    "LVDT",
    "HMD",
    "PROXIMITY",
    "PRESSURE_SWITCH",
]

# Devices with fewer than this many labeled examples are dropped from
# training -- not enough signal for k-fold CV or a stable classifier.
# (At last count this drops only TT, n=1.)
MIN_CLASS_COUNT = 5

# ====================== MODELING PARAMETERS ======================
RANDOM_STATE = 42          # For reproducibility
TEST_SIZE = 0.20           # Only relevant if you add a random train/test
                           # split somewhere. IMPORTANT: for the actual
                           # generalization check on this data, prefer the
                           # time-based holdout (train on early months,
                           # test on the most recent unseen month) over a
                           # random TEST_SIZE split -- a random split lets
                           # near-duplicate phrasing from the same month
                           # leak between train and test and will overstate
                           # accuracy the same way k-fold CV did (76.5% CV
                           # vs 51.4% true holdout on this dataset so far).

# Below this predicted probability, a prediction is flagged low-confidence
# rather than presented as a normal top result.
CONFIDENCE_THRESHOLD = 0.35

# ====================== PREDICTION HORIZONS ======================
# Hours-ahead windows for a future "will this device cause a delay in the
# next N hours" model (separate from the cause classifier above).
# NOT YET IMPLEMENTED: this needs each event's timestamp plus a defined
# per-device "at risk" window to build forward-looking labels -- current
# master_events.csv has date + duration but no reliable time-of-day for
# every row (some source months lack a parseable start time), so horizon
# labels can't be built reliably yet for all 7 months. Fix at the data
# layer (ensure every row has a real datetime) before using this list.
HORIZONS = [4, 8, 24]

# ====================== MONTHS COVERED ======================
MONTH_ORDER = ["Nov-25", "Dec-25", "Jan-26", "Feb-26", "Mar-26", "Apr-26", "May-26"]

# Domain keyword groups used as extra binary features on top of TF-IDF.
# Sourced from FMEA vocabulary + recurring delay-text phrasing. Extend as
# new failure phrasing is observed in future months.
DOMAIN_KEYWORDS = {
    "kw_cleaning": ["clean", "dust", "dirty", "contamina"],
    "kw_alignment": ["alignment", "misalign", "position error", "positioning"],
    "kw_electrical": ["voltage", "power", "cable", "psu", "signal"],
    "kw_mechanical": ["bolt", "vibration", "loose", "coupling", "bracket"],
    "kw_pressure": ["pressure", "oil", "lubrica", "hydraulic"],
    "kw_missing_feedback": ["feedback missing", "not sensing", "no signal", "missed detection"],
    "kw_low_high": ["low", "high", "not ok", "malfunction"],
}