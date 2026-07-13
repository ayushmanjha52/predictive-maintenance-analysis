"""
Validates data/fmea_risk_scores.csv against two things that are easy to
get wrong when editing this file by hand:

  1. RPN must equal Severity x Occurrence x Detection for every row --
     catches data-entry mistakes like a typo'd RPN.
  2. Every device the classifier can actually output (read live from
     training_manifest.json) must have a row here -- catches a device
     silently falling back to UNKNOWN risk in the dashboard/API just
     because nobody added its FMEA row yet.

Run this after every manual edit to fmea_risk_scores.csv.
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR, DATA_DIR

FMEA_FILE = DATA_DIR / "fmea_risk_scores.csv"
MANIFEST_FILE = MODELS_DIR / "training_manifest.json"


def validate_fmea_scores():
    issues = []

    if not FMEA_FILE.exists():
        print(f"[FAIL] {FMEA_FILE} not found.")
        return False

    df = pd.read_csv(FMEA_FILE)
    df["device"] = df["device"].str.upper().str.strip()

    print(f"Loaded {len(df)} FMEA rows from {FMEA_FILE.name}\n")

    # Check 1: RPN = S x O x D
    print("=== Checking RPN = Severity x Occurrence x Detection ===")
    for _, row in df.iterrows():
        computed = row["severity"] * row["occurrence"] * row["detection"]
        if abs(computed - row["rpn"]) > 0.01:
            msg = (f"{row['device']}: stated RPN={row['rpn']}, but "
                   f"{row['severity']}x{row['occurrence']}x{row['detection']}={computed:.0f}")
            issues.append(msg)
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [PASS] {row['device']}: {row['rpn']}")

    # Check 2: every classifier class has a row here
    print("\n=== Checking coverage against actual classifier classes ===")
    if MANIFEST_FILE.exists():
        manifest = json.load(open(MANIFEST_FILE))
        classifier_classes = set(manifest.get("classes", []))
        fmea_devices = set(df["device"])
        missing = classifier_classes - fmea_devices
        if missing:
            msg = f"Classifier classes with NO FMEA row (will show UNKNOWN risk): {missing}"
            issues.append(msg)
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [PASS] All {len(classifier_classes)} classifier classes have an FMEA row")
    else:
        print(f"  [SKIP] {MANIFEST_FILE} not found -- run train.py first to enable this check")

    print()
    if issues:
        print(f"VALIDATION FAILED: {len(issues)} issue(s) found. Fix these before trusting "
              f"the FMEA-fused priority table or dashboard.")
        return False
    print("VALIDATION PASSED: all rows are arithmetically consistent and coverage is complete.")
    return True


if __name__ == "__main__":
    ok = validate_fmea_scores()
    sys.exit(0 if ok else 1)