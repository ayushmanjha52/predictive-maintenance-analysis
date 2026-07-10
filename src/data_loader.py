"""
Data Loader & Validation Script for Combi Mill PdM Project
"""

import pandas as pd
from pathlib import Path
import sys

# Add current folder to Python path so it can find config.py
sys.path.append(str(Path(__file__).parent))

from config import MASTER_EVENTS_PATH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_master_events():
    """Load the unified master events CSV file."""
    if not MASTER_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"master_events.csv not found at: {MASTER_EVENTS_PATH}\n"
            "Please make sure the file is placed inside the 'data' folder."
        )
    
    df = pd.read_csv(MASTER_EVENTS_PATH)
    logger.info(f"Successfully loaded {len(df)} events from master_events.csv")
    return df


def validate_data(df: pd.DataFrame):
    """Perform basic data quality checks and print summary."""
    print("\n" + "="*60)
    print("           DATA VALIDATION REPORT")
    print("="*60)
    
    print(f"\nTotal Events in Dataset     : {len(df)}")
    
    if 'date' in df.columns:
        print(f"Date Range                  : {df['date'].min()} to {df['date'].max()}")
    
    if 'field_device' in df.columns:
        resolved = df['field_device'].notna().sum()
        resolved_pct = round(resolved / len(df) * 100, 1)
        print(f"\nResolved Field Device       : {resolved} ({resolved_pct}%)")
        print(f"Unresolved Events           : {len(df) - resolved} ({100 - resolved_pct}%)")
        
        print("\nTop Devices by Count:")
        print(df['field_device'].value_counts().head(8))
    
    # ====================== FIXED PART ======================
    if 'mins' in df.columns:
        # Convert 'mins' column to numeric (fixes the error)
        df['mins'] = pd.to_numeric(df['mins'], errors='coerce')
        
        total_mins = int(df['mins'].sum())
        avg_mins = round(df['mins'].mean(), 1)
        print(f"\nTotal Delay Minutes         : {total_mins:,}")
        print(f"Average Delay per Event     : {avg_mins} minutes")
    # ========================================================
    
    print("\n" + "="*60)
    print("Validation Complete. Data is ready for feature engineering.")
    print("="*60 + "\n")

if __name__ == "__main__":
    df = load_master_events()
    validate_data(df)