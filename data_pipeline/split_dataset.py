"""
Split Dolly dataset into:

80% Train
10% Validation
10% Test

Usage:
    python data_pipeline/split_dataset.py
"""

from datasets import load_from_disk

from configs.data_cfg import (
    RAW_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_JSON,
    VAL_JSON,
    TEST_JSON,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    JSON_DIR,
    SEED,
)


def main():

    dataset = load_from_disk(RAW_DATA_DIR)["train"]

    # Add sample id
    dataset = dataset.add_column(
        "sample_id",
        list(range(len(dataset)))
    )

    # -------------------------
    # Train / Temp
    # -------------------------

    split = dataset.train_test_split(
        test_size=(1.0 - TRAIN_RATIO),
        seed=SEED,
        shuffle=True,
    )

    train_dataset = split["train"]
    temp_dataset = split["test"]

    # -------------------------
    # Validation / Test
    # -------------------------

    val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    split = temp_dataset.train_test_split(
        train_size=val_ratio,
        seed=SEED,
        shuffle=True,
    )

    val_dataset = split["train"]
    test_dataset = split["test"]

    # -------------------------
    # Save HF Dataset
    # -------------------------

    train_dataset.save_to_disk(TRAIN_DIR)
    val_dataset.save_to_disk(VAL_DIR)
    test_dataset.save_to_disk(TEST_DIR)

    # -------------------------
    # Save JSONL
    # -------------------------

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset.to_json(TRAIN_JSON, lines=True)
    val_dataset.to_json(VAL_JSON, lines=True)
    test_dataset.to_json(TEST_JSON, lines=True)

    print()

    print(f"Train      : {len(train_dataset):5d}")
    print(f"Validation : {len(val_dataset):5d}")
    print(f"Test       : {len(test_dataset):5d}")

    print()

    print("Saved to:")
    print(TRAIN_DIR)
    print(VAL_DIR)
    print(TEST_DIR)


if __name__ == "__main__":
    main()
