"""
Data Validation Tests for Combi Mill PdM Project

FIX vs. previous version: the old file only checked that master_events.csv
exists, isn't empty, and has the right column NAMES. It didn't test any of
the actual data-quality bugs this project has found along the way:
  - 8 exact duplicate rows (found via data_loader.py's dedup check)
  - Feb-26's ENTIRE date column being NaN (98 total unparseable dates)
  - Compound field_device labels ("HMD_/_LVDT") not collapsing correctly
  - Coverage silently regressing if the auto-tagger breaks
  - Reference-total mismatches vs. Final_data.xlsx (the exact check that
    caught the April/May column-offset bugs earlier in this project)

These tests encode each of those as an explicit regression check, so if
any of them recurs after a future data refresh, it's caught immediately
instead of silently degrading the model again.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.config import MASTER_EVENTS_PATH, MONTH_ORDER
from src.data_loader import primary_device

REQUIRED_COLUMNS = {"date", "month", "mins", "reason_text", "field_device", "tag_source", "source_file"}

# Known-good monthly totals from Final_data.xlsx -- the reference that
# caught the original April/May column-offset bugs. Update this if the
# reference source changes; keep it, don't delete it, even if it feels
# redundant with data_loader.py's own check -- this is the test-suite
# copy that runs in CI regardless of whether anyone remembers to run
# data_loader.py manually.
REFERENCE_MONTH_TOTALS = {
    "Nov-25": 2995, "Dec-25": 3779, "Jan-26": 3782, "Feb-26": 3316,
    "Mar-26": 1103, "Apr-26": 3414, "May-26": 2058,
}
REFERENCE_TOLERANCE = 40  # minutes -- small rounding/edge-row differences are OK


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(MASTER_EVENTS_PATH)


def test_master_events_exists():
    assert MASTER_EVENTS_PATH.exists(), f"File not found: {MASTER_EVENTS_PATH}"


def test_master_events_not_empty(df):
    assert len(df) > 0, "master_events.csv is empty"


def test_required_columns(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_exact_duplicate_rows(df):
    """Regression test for the 8 exact duplicate rows found earlier.
    A small number may be legitimate (two genuinely identical short
    delays), so this warns rather than hard-fails above a low threshold
    -- but a large jump signals a real problem (e.g. a source file
    being loaded twice, like the MAY2026/MAY2026-1 duplicate file
    issue caught earlier)."""
    dup_count = df.duplicated().sum()
    assert dup_count <= 15, (
        f"{dup_count} exact duplicate rows found (baseline was 8) -- "
        f"check whether a source file is being loaded/unified twice."
    )


def test_dates_mostly_parseable(df):
    """Regression test for the Feb-26 bug: ALL 92 of that month's dates
    were NaN due to an upstream unification issue. This doesn't require
    100% parseable (some genuine per-row ambiguity is tolerated) but
    catches another WHOLE MONTH silently going to 0% parseable dates."""
    parsed = pd.to_datetime(df["date"], errors="coerce")
    overall_rate = parsed.notna().mean()
    assert overall_rate > 0.70, (
        f"Only {overall_rate:.1%} of dates are parseable overall -- "
        f"investigate before trusting any date-based analysis."
    )


def test_no_entire_month_missing_dates(df):
    """Stronger, more specific version of the above: fails if ANY month
    present in the data has 0% parseable dates -- this is exactly what
    happened to Feb-26 and would otherwise pass a lenient overall-rate
    check if other months compensate for it."""
    parsed = pd.to_datetime(df["date"], errors="coerce")
    tmp = df.assign(_date_ok=parsed.notna())
    per_month_rate = tmp.groupby("month")["_date_ok"].mean()
    zero_rate_months = per_month_rate[per_month_rate == 0].index.tolist()
    assert not zero_rate_months, (
        f"These months have ZERO parseable dates: {zero_rate_months} -- "
        f"this is the exact Feb-26 bug recurring. Check the loader for "
        f"that month's raw source file."
    )


def test_mins_are_non_negative_and_bounded(df):
    """Sanity check on delay duration: negative values indicate a parsing
    bug (e.g. end-time-before-start-time); absurdly large values (>24h)
    likely indicate a units or column-offset error, like the ones found
    earlier in the April/May loaders."""
    mins = pd.to_numeric(df["mins"], errors="coerce").dropna()
    assert (mins >= 0).all(), f"{(mins < 0).sum()} rows have a negative 'mins' value."
    absurd = (mins > 1440).sum()  # more than 24 hours for a single delay
    assert absurd == 0, (
        f"{absurd} rows have 'mins' > 1440 (24 hours) for a single delay event -- "
        f"likely a units or column-offset bug, verify against the raw source file."
    )


def test_field_device_coverage_above_threshold(df):
    """Regression test for auto-tagger coverage. Current baseline is
    ~53%; this fails if coverage drops meaningfully below that, which
    would indicate the keyword tagger or explicit sensor-column mapping
    broke on a newly added month's data."""
    resolved_pct = df["field_device"].notna().mean()
    assert resolved_pct >= 0.45, (
        f"Only {resolved_pct:.1%} of events resolved to a field device "
        f"(baseline ~53%) -- check auto_tag.py's keyword list still "
        f"matches the newest month's phrasing."
    )


def test_compound_labels_collapse_correctly(df):
    """primary_device() must correctly collapse every compound label
    (e.g. 'HMD_/_LVDT' -> 'HMD') -- this is the exact bug that silently
    discarded real HMD/LVDT training examples in train.py before it was
    fixed. This test guarantees the collapsing function itself works,
    independent of whether every training script remembers to call it."""
    devices = df["field_device"].dropna().unique()
    for raw in devices:
        collapsed = primary_device(raw)
        assert collapsed is not None and "_/_" not in collapsed, (
            f"primary_device('{raw}') returned '{collapsed}' -- compound "
            f"labels must collapse to a single clean device name."
        )


def test_month_values_are_recognized(df):
    """Every value in the 'month' column should be one of the known
    labels in config.MONTH_ORDER. An unrecognized value means a new
    month was added to the raw data but the loader/config wasn't
    updated -- it would otherwise silently sort incorrectly or be
    excluded from month-ordered charts/forecasts."""
    unknown = set(df["month"].dropna().unique()) - set(MONTH_ORDER)
    assert not unknown, (
        f"Unrecognized month label(s): {unknown} -- add these to "
        f"config.MONTH_ORDER, in the correct chronological position."
    )


@pytest.mark.parametrize("month,expected_total", REFERENCE_MONTH_TOTALS.items())
def test_monthly_totals_match_reference(df, month, expected_total):
    """Cross-check against Final_data.xlsx -- the exact validation that
    caught the original April/May column-offset bugs in this project.
    Keep this even though it duplicates data_loader.py's own check --
    this version runs automatically under pytest/CI."""
    mins = pd.to_numeric(df["mins"], errors="coerce")
    actual_total = mins[df["month"] == month].sum()
    assert abs(actual_total - expected_total) <= REFERENCE_TOLERANCE, (
        f"{month}: total is {actual_total:.0f} min, expected ~{expected_total} "
        f"(tolerance {REFERENCE_TOLERANCE}) -- a loader for this month may "
        f"have a column-offset or date-format bug."
    )


if __name__ == "__main__":
    # Allows `python test_data_validation.py` for a quick manual run,
    # but `pytest test_data_validation.py -v` is the recommended way to
    # run this file -- it gives per-test pass/fail output and integrates
    # with CI, rather than stopping at the first failed assertion.
    sys.exit(pytest.main([__file__, "-v"]))