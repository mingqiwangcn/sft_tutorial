"""
Tokenize dataset for Supervised Fine-Tuning (SFT).

Pipeline

sample
    ↓
messages
    ↓
Qwen Chat Template
    ↓
Tokenization
    ↓
Prompt / Full Diff
    ↓
Labels

This version does not pack. Padding is handled dynamically by the training
collator.
"""

from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from configs.data_cfg import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_TOKENIZED_DIR,
    VAL_TOKENIZED_DIR,
    TEST_TOKENIZED_DIR,
)

from configs.model_cfg import (
    MODEL_NAME,
    MAX_LENGTH,
)

from data_pipeline.chat_messages import (
    build_messages,
)

IGNORE_INDEX = -100


def build_labels(
    input_ids: list[int],
    prompt_length: int,
) -> list[int]:
    labels = [IGNORE_INDEX] * prompt_length
    labels.extend(input_ids[prompt_length:])

    return labels


def tokenize_sample(sample: dict, tokenizer: AutoTokenizer) -> dict:
    messages = build_messages(sample)

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_length = len(prompt_ids)
    has_generation_tokens = prompt_length < len(input_ids)

    if not has_generation_tokens:
        print(
            "[WARN] Dropping sample without generation tokens. "
            f"sample_id={sample.get('sample_id', 'unknown')} "
            f"prompt_length={prompt_length} "
            f"input_length={len(input_ids)}"
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [IGNORE_INDEX] * len(input_ids),
            "has_generation_tokens": False,
        }

    labels = build_labels(
        input_ids=input_ids,
        prompt_length=prompt_length,
    )

    if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
        raise ValueError("Tokenized fields must have the same sequence length.")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "has_generation_tokens": True,
    }


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    tokenizer: AutoTokenizer,
) -> None:
    dataset = load_from_disk(dataset_dir)

    tokenized_dataset = dataset.map(
        lambda sample: tokenize_sample(sample, tokenizer),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {dataset_dir.name}",
        load_from_cache_file=False
    )

    before_count = len(tokenized_dataset)

    tokenized_dataset = tokenized_dataset.filter(
        lambda sample: sample["has_generation_tokens"],
        desc=f"Filtering samples without generation tokens from {dataset_dir.name}",
    )

    filtered_count = before_count - len(tokenized_dataset)

    tokenized_dataset = tokenized_dataset.remove_columns(
        ["has_generation_tokens"]
    )

    if filtered_count > 0: 
        print(
            f"Filtered {filtered_count} samples "
            "without generation tokens."
        )

    tokenized_dataset.save_to_disk(output_dir)

    print(f"Saved -> {output_dir}")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    process_dataset(TRAIN_DIR, TRAIN_TOKENIZED_DIR, tokenizer)
    process_dataset(VAL_DIR, VAL_TOKENIZED_DIR, tokenizer)
    process_dataset(TEST_DIR, TEST_TOKENIZED_DIR, tokenizer)


if __name__ == "__main__":
    main()
