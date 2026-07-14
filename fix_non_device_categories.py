"""
Removes SEQUENCE_BREAK, OVERTRAVEL, and ROLLER_TABLE from field_device --
these are process outcomes/symptoms, not physical field devices, and don't
have real FMEA data (Severity/Occurrence/Detection ratings don't apply to
a symptom the way they apply to a component's failure mode).

Run this ONCE against your actual data/master_events.csv, then re-run
train.py to retrain the classifier without these phantom classes.

Usage: python fix_non_device_categories.py
"""
import pandas as pd
from pathlib import Path

MASTER_EVENTS_PATH = Path("data/master_events.csv")
NON_DEVICE_CATEGORIES = {"SEQUENCE_BREAK", "OVERTRAVEL", "ROLLER_TABLE"}


def main():
    df = pd.read_csv(MASTER_EVENTS_PATH)

    affected = df["field_device"].isin(NON_DEVICE_CATEGORIES)
    print(f"Rows currently tagged with a non-device category: {affected.sum()}")
    for cat in NON_DEVICE_CATEGORIES:
        count = (df["field_device"] == cat).sum()
        print(f"  {cat}: {count}")

    if affected.sum() == 0:
        print("Nothing to fix -- these categories aren't present in this file.")
        return

    # Reclassify these rows as unresolved/non-field-device, matching how
    # any other non-device delay (furnace, motor, crane) is already
    # represented -- field_device = blank, tag_source noted for audit.
    df.loc[affected, "field_device"] = ""
    df.loc[affected, "tag_source"] = "reclassified_non_device"

    df.to_csv(MASTER_EVENTS_PATH, index=False)
    print(f"\nFixed {affected.sum()} rows and saved back to {MASTER_EVENTS_PATH}")
    print("Next step: re-run train.py to retrain without these phantom classes.")


if __name__ == "__main__":
    main()