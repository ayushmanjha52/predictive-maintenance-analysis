"""
Data Loader & Validation Script for Combi Mill PdM Project
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import pandas as pd

from config import MASTER_EVENTS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "month", "mins", "reason_text", "field_device", "tag_source", "source_file"}


def primary_device(fd):
    """Collapse compound labels like 'HMD_/_LVDT' to the first-listed
    device. Without this, value_counts() on raw field_device silently
    splits real device counts across compound-label variants and every
    downstream Pareto/count number becomes wrong."""
    if pd.isna(fd) or not fd:
        return None
    return str(fd).split("_/_")[0]


def load_master_events() -> pd.DataFrame:
    """Load the unified master events CSV file."""
    if not MASTER_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"master_events.csv not found at: {MASTER_EVENTS_PATH}\n"
            "Please make sure the file is placed inside the 'data' folder."
        )

    df = pd.read_csv(MASTER_EVENTS_PATH)
    logger.info(f"Loaded {len(df)} events from {MASTER_EVENTS_PATH.name}")
    return df


def validate_data(df: pd.DataFrame, reference_month_totals: dict | None = None) -> dict:
    """Run data quality checks, print a human-readable report, and return
    a dict of findings so this can be asserted on in tests/CI rather than
    just eyeballed on a terminal.

    reference_month_totals: optional {month: expected_total_minutes} to
    cross-check against a known-good source (e.g. Final_data.xlsx) --
    this is the check that actually caught the April/May column-offset
    bugs earlier. Pass it whenever you have a trusted reference.
    """
    issues = []
    stats = {}

    print("\n" + "=" * 60)
    print("           DATA VALIDATION REPORT")
    print("=" * 60)

    # ---- Schema check ----
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        issues.append(f"Missing expected columns: {missing_cols}")
        print(f"\n[FAIL] Missing expected columns: {missing_cols}")
    stats["n_rows"] = len(df)
    print(f"\nTotal Events in Dataset     : {len(df)}")

    # ---- Duplicate rows ----
    dup_count = df.duplicated().sum()
    stats["duplicate_rows"] = int(dup_count)
    if dup_count > 0:
        issues.append(f"{dup_count} exact duplicate rows found")
        print(f"[WARN] {dup_count} exact duplicate rows found "
              f"(same bug as MAY2026 vs MAY2026-1 being identical files earlier)")

    # ---- Date range + missing dates ----
    if "date" in df.columns:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        n_bad_dates = parsed_dates.isna().sum()
        stats["unparseable_dates"] = int(n_bad_dates)
        print(f"Date Range                  : {parsed_dates.min()} to {parsed_dates.max()}")
        if n_bad_dates > 0:
            issues.append(f"{n_bad_dates} rows have an unparseable/missing date")
            print(f"[WARN] {n_bad_dates} rows have an unparseable or missing date")

    # ---- Field device resolution (using collapsed primary device) ----
    if "field_device" in df.columns:
        df = df.copy()
        df["_primary_device"] = df["field_device"].apply(primary_device)
        resolved = df["_primary_device"].notna().sum()
        resolved_pct = round(resolved / len(df) * 100, 1)
        stats["resolved_count"] = int(resolved)
        stats["resolved_pct"] = resolved_pct
        print(f"\nResolved Field Device       : {resolved} ({resolved_pct}%)")
        print(f"Unresolved Events           : {len(df) - resolved} ({100 - resolved_pct}%)")

        print("\nTop Devices by Count (compound labels collapsed):")
        print(df["_primary_device"].value_counts().head(8))

    # ---- Mins column: coerce + report what was lost ----
    if "mins" in df.columns:
        original_non_null = df["mins"].notna().sum()
        df["mins"] = pd.to_numeric(df["mins"], errors="coerce")
        coerced_to_nan = original_non_null - df["mins"].notna().sum()
        stats["mins_coerced_to_nan"] = int(coerced_to_nan)
        if coerced_to_nan > 0:
            issues.append(f"{coerced_to_nan} 'mins' values could not be converted to numeric and were dropped")
            print(f"\n[WARN] {coerced_to_nan} 'mins' values were non-numeric and silently became NaN "
                  f"(e.g. stray header/footer rows) -- these are EXCLUDED from totals below")

        total_mins = df["mins"].sum()
        avg_mins = round(df["mins"].mean(), 1)
        stats["total_mins"] = float(total_mins)
        stats["avg_mins"] = float(avg_mins)
        print(f"\nTotal Delay Minutes         : {total_mins:,.0f}")
        print(f"Average Delay per Event     : {avg_mins} minutes")

        # ---- Per-month breakdown + reference cross-check ----
        if "month" in df.columns:
            month_totals = df.groupby("month")["mins"].sum().round(0).to_dict()
            stats["month_totals"] = month_totals
            print("\nPer-Month Totals:")
            for m, v in month_totals.items():
                line = f"  {m:10s} {v:8.0f} min"
                if reference_month_totals and m in reference_month_totals:
                    expected = reference_month_totals[m]
                    diff = v - expected
                    if abs(diff) > 5:  # tolerance for rounding
                        issues.append(f"{m}: total {v:.0f} min differs from reference {expected:.0f} min "
                                      f"(diff {diff:+.0f})")
                        line += f"   [MISMATCH vs reference {expected:.0f}, diff {diff:+.0f}]"
                    else:
                        line += "   [OK vs reference]"
                print(line)

    print("\n" + "=" * 60)
    if issues:
        print(f"Validation found {len(issues)} issue(s) -- review before using this data:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("Validation Complete. No issues found. Data is ready for feature engineering.")
    print("=" * 60 + "\n")

    stats["issues"] = issues
    return stats


if __name__ == "__main__":
    df = load_master_events()

    # Known-good monthly totals from Final_data.xlsx, used as a cross-check.
    # Update/remove this if the reference source changes.
    reference_totals = {
        "Nov-25": 2995, "Dec-25": 3779, "Jan-26": 3782, "Feb-26": 3316,
        "Mar-26": 1103, "Apr-26": 3414, "May-26": 2058,
    }

    results = validate_data(df, reference_month_totals=reference_totals)

    if results["issues"]:
        sys.exit(1)  # non-zero exit lets this be used as a CI/pipeline gate