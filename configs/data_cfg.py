"""
Dataset configuration.

This project uses:
    Databricks Dolly 15K

Dataset Split:
    Train      80%
    Validation 10%
    Test       10%
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

DATASET_NAME = "databricks/databricks-dolly-15k"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


# -----------------------------------------------------------------------------
# Output Directories
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

TRAIN_TOKENIZED_DIR = DATA_DIR / "train_tokenized"
VAL_TOKENIZED_DIR = DATA_DIR / "validation_tokenized"
TEST_TOKENIZED_DIR = DATA_DIR / "test_tokenized"

JSON_DIR = DATA_DIR / "json"


# -----------------------------------------------------------------------------
# JSON Export
# -----------------------------------------------------------------------------

TRAIN_JSON = JSON_DIR / "train.jsonl"
VAL_JSON = JSON_DIR / "validation.jsonl"
TEST_JSON = JSON_DIR / "test.jsonl"
