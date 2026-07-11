"""
FMEA Risk Scoring Module (Loads from CSV)
"""

import pandas as pd
from pathlib import Path

FMEA_FILE = Path("data/fmea_risk_scores.csv")

# Load FMEA data once when module is imported
try:
    fmea_df = pd.read_csv(FMEA_FILE)
    fmea_df.set_index("device", inplace=True)
except FileNotFoundError:
    print("Warning: FMEA file not found. Using default values.")
    fmea_df = None


def get_fmea_risk(device: str) -> dict:
    """
    Returns FMEA risk information for a given device.
    Falls back to default values if device is not found.
    """
    device = device.upper().strip()

    if fmea_df is not None and device in fmea_df.index:
        row = fmea_df.loc[device]
        return {
            "rpn": int(row["rpn"]),
            "severity": float(row["severity"]),
            "occurrence": float(row["occurrence"]),
            "detection": float(row["detection"]),
            "risk_level": row["risk_level"]
        }
    else:
        # Default fallback
        return {
            "rpn": 100,
            "severity": 4,
            "occurrence": 3,
            "detection": 8,
            "risk_level": "Low"
        }


def calculate_combined_risk(ml_probability: float, fmea_rpn: int) -> dict:
    """
    Combines ML probability with FMEA RPN.
    """
    ml_score = ml_probability / 100
    fmea_score = min(fmea_rpn / 500, 1.0)

    combined_score = (ml_score * 0.6) + (fmea_score * 0.4)

    if combined_score >= 0.7:
        final_risk = "High"
    elif combined_score >= 0.4:
        final_risk = "Medium"
    else:
        final_risk = "Low"

    return {
        "combined_risk_score": round(combined_score * 100, 1),
        "final_risk_level": final_risk
    } 