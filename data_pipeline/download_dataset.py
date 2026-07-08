"""
Download Databricks Dolly 15K.

Usage:
    python data_pipeline/download_dataset.py
"""

from datasets import load_dataset

from configs.data_cfg import (
    DATASET_NAME,
    RAW_DATA_DIR,
)


def main():

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_NAME} ...")

    dataset = load_dataset(DATASET_NAME)

    dataset.save_to_disk(RAW_DATA_DIR)

    print(f"Saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
