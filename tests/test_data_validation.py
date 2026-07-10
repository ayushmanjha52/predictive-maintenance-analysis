"""
Basic Data Validation Tests
"""

import pandas as pd
import sys

# ==================== TEMPORARY FIX ====================
# Add your PDM folder path manually
sys.path.insert(0, r"C:\Users\ayush\OneDrive\Attachments\Desktop\PDM")
# =====================================================

from config import MASTER_EVENTS_PATH


def test_master_events_exists():
    assert MASTER_EVENTS_PATH.exists(), "master_events.csv not found"
    print("✓ master_events.csv exists")


def test_master_events_not_empty():
    df = pd.read_csv(MASTER_EVENTS_PATH)
    assert len(df) > 0, "File is empty"
    print(f"✓ master_events.csv has {len(df)} rows")


def test_required_columns():
    df = pd.read_csv(MASTER_EVENTS_PATH)
    required = ['date', 'mins', 'reason_text', 'field_device']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print("✓ All required columns present")


if __name__ == "__main__":
    print("Running Data Validation Tests...\n")
    test_master_events_exists()
    test_master_events_not_empty()
    test_required_columns()
    print("\n✅ All tests passed!")