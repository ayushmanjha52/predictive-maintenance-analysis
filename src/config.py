"""
Configuration file for Field Device Predictive Maintenance Project
"""

from pathlib import Path

# ====================== PROJECT PATHS ======================
# Change this path to where you want to keep your project on your laptop
PROJECT_ROOT = Path(r"C:\Users\ayush\OneDrive\Attachments\Desktop\PDM")

DATA_DIR      = PROJECT_ROOT / "data"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Main unified data file
MASTER_EVENTS_PATH = DATA_DIR / "master_events.csv"

# ====================== KEY DEVICES ======================
CORE_DEVICES = [
    "PHOTOCELL", 
    "ENCODER", 
    "LVDT", 
    "HMD", 
    "PROXIMITY", 
    "PRESSURE_SWITCH"
]

# ====================== MODELING PARAMETERS ======================
RANDOM_STATE = 42          # For reproducibility
TEST_SIZE = 0.20           # Train-test split ratio

# Prediction horizons (in hours)
HORIZONS = [4, 8, 24]