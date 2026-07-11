"""
Basic Data Validation Tests for Combi Mill PdM Project
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MASTER_EVENTS_PATH


def test_master_events_exists():
    assert MASTER_EVENTS_PATH.exists(), f"File not found: {MASTER_EVENTS_PATH}"
    print("✓ master_events.csv exists")


def test_master_events_not_empty():
    df = pd.read_csv(MASTER_EVENTS_PATH)
    assert len(df) > 0, "master_events.csv is empty"
    print(f"✓ master_events.csv has {len(df)} rows")


def test_required_columns():
    df = pd.read_csv(MASTER_EVENTS_PATH)
    required_cols = ['date', 'mins', 'reason_text', 'field_device']
    
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print("✓ All required columns present")


if __name__ == "__main__":
    print("Running Data Validation Tests...\n")
    
    test_master_events_exists()
    test_master_events_not_empty()
    test_required_columns()
    
    print("\n All tests passed!")