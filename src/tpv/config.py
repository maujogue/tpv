"""Configuration values for local EEG data paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

data_path = DEFAULT_DATA_DIR
