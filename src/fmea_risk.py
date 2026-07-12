"""
FMEA Risk Scoring Module (Loads from CSV)

FIX vs. previous version: devices with no FMEA sheet (confirmed:
PHOTOCELL and PROXIMITY as of this data -- your #1 and #5 empirical
delay contributors) used to silently fall back to a default of
{"rpn": 100, "risk_level": "Low"}. That's not a neutral placeholder --
it actively tells a reader these devices are LOW risk, when the truth
is simply "we don't know yet, no FMEA exists." Confirmed directly:

    PHOTOCELL (original logic): {'rpn': 100, 'risk_level': 'Low'}
    PROXIMITY (original logic): {'rpn': 100, 'risk_level': 'Low'}

Fix: fallback now returns risk_level="UNKNOWN" and is_estimated=True,
so any downstream table/dashboard can visually distinguish "genuinely
low risk per FMEA" from "no FMEA data exists for this device" -- these
must never look the same to a reader.
"""

import logging
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

FMEA_FILE = DATA_DIR / "fmea_risk_scores.csv"

# RPN theoretical max is 10*10*10=1000 (Severity*Occurrence*Detection,
# each 1-10). Named here instead of a bare "500" so the normalization
# assumption is visible and adjustable.
RPN_THEORETICAL_MAX = 1000
RPN_NORMALIZATION_CAP = 500  # values above this are treated as maximally risky

# Weight given to the ML model's own probability vs. the FMEA-derived
# risk score when blending the two. Kept as named constants so they're
# easy to find and re-tune, rather than magic numbers buried in a formula.
ML_WEIGHT = 0.6
FMEA_WEIGHT = 0.4

DEFAULT_RISK = {
    "rpn": None,
    "severity": None,
    "occurrence": None,
    "detection": None,
    "risk_level": "UNKNOWN",
    "is_estimated": True,
}


def _load_fmea_data():
    """Lazy-loaded (not at import time) so a bad/missing file doesn't
    crash the whole module import, and so tests can point this at a
    different file without monkeypatching module globals."""
    if not FMEA_FILE.exists():
        logger.warning(f"FMEA file not found at {FMEA_FILE} -- all devices will "
                        f"return UNKNOWN risk until this is added.")
        return None
    try:
        df = pd.read_csv(FMEA_FILE)
        if "device" not in df.columns:
            logger.error(f"{FMEA_FILE} is missing a 'device' column -- cannot use it.")
            return None
        df["device"] = df["device"].astype(str).str.upper().str.strip()
        df.set_index("device", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load {FMEA_FILE}: {e}")
        return None


_fmea_df = _load_fmea_data()


def get_fmea_risk(device: str) -> dict:
    """
    Returns FMEA risk information for a given device.

    If the device has no FMEA entry, returns risk_level="UNKNOWN" with
    is_estimated=True -- NEVER silently reports "Low", since that would
    misrepresent devices like Photocell/Proximity that are simply
    missing an FMEA sheet, not confirmed low-risk.
    """
    device_key = device.upper().strip()

    if _fmea_df is not None and device_key in _fmea_df.index:
        row = _fmea_df.loc[device_key]
        return {
            "rpn": int(row["rpn"]),
            "severity": float(row["severity"]),
            "occurrence": float(row["occurrence"]),
            "detection": float(row["detection"]),
            "risk_level": row["risk_level"],
            "is_estimated": False,
        }

    logger.warning(f"No FMEA entry for device='{device_key}' -- returning UNKNOWN, "
                    f"not a guessed risk level. Add this device to {FMEA_FILE.name} "
                    f"once its FMEA is available.")
    return dict(DEFAULT_RISK)


def calculate_combined_risk(ml_probability: float, fmea_rpn, is_estimated: bool = False) -> dict:
    """
    Combines ML probability with FMEA RPN into one score.

    fmea_rpn may be None (device has no FMEA yet) -- in that case the
    combined score falls back to ML-only and is flagged, rather than
    silently treating a missing RPN as if it were a real low value.
    """
    if not (0 <= ml_probability <= 100):
        raise ValueError(f"ml_probability must be 0-100, got {ml_probability}")

    ml_score = ml_probability / 100

    if fmea_rpn is None:
        # No FMEA data -- score on ML probability alone, but say so.
        combined_score = ml_score
        result_is_estimated = True
    else:
        fmea_score = min(fmea_rpn / RPN_NORMALIZATION_CAP, 1.0)
        combined_score = (ml_score * ML_WEIGHT) + (fmea_score * FMEA_WEIGHT)
        result_is_estimated = is_estimated

    if combined_score >= 0.7:
        final_risk = "High"
    elif combined_score >= 0.4:
        final_risk = "Medium"
    else:
        final_risk = "Low"

    return {
        "combined_risk_score": round(combined_score * 100, 1),
        "final_risk_level": final_risk,
        "is_estimated": result_is_estimated,
        "note": ("FMEA RPN unavailable for this device -- score is ML-only, "
                 "treat with caution") if fmea_rpn is None else None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for device in ["LVDT", "ENCODER", "HMD", "PHOTOCELL", "PROXIMITY"]:
        risk = get_fmea_risk(device)
        print(f"{device:12s} -> {risk}")