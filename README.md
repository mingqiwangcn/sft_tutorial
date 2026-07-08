# Enterprise SFT Tutorial

Production-oriented supervised fine-tuning tutorial for Qwen2.5-7B-Instruct on
Databricks Dolly 15K.

This project uses:

- PyTorch
- Transformers
- Accelerate with FSDP2
- FlashAttention2
- Qwen/Qwen2.5-7B-Instruct
- databricks/databricks-dolly-15k

The training path is full-parameter continued SFT. It does not use LoRA,
Transformers Trainer, or TRL.

## Pipeline

```text
Download Dolly 15K
    -> split train / validation / test
    -> convert each row to chat messages
    -> apply Qwen chat template
    -> tokenize
    -> dynamic padding collator
    -> DataLoader
    -> Qwen2.5-7B-Instruct + FlashAttention2
    -> Accelerate FSDP2
    -> full fine-tuning
```

This version intentionally does not use sequence packing. Padding is handled
dynamically in the collator, so each batch is padded only to the longest sample
in that batch.

## Setup

Create an environment and install base dependencies:

```bash
pip install -r requirements.txt
```

On the CUDA GPU training machine, install the training extras:

```bash
pip install -r requirements-train.txt
```

`flash-attn` is intentionally optional for local development because it is a
CUDA extension and usually does not install on macOS. The default training
configuration still uses FlashAttention2:

```python
ATTN_IMPLEMENTATION = "flash_attention_2"
```

If you want to run without `flash-attn`, set `ATTN_IMPLEMENTATION = "sdpa"` in
`configs/model_cfg.py`.

## Configuration

Main configuration files:

```text
configs/data_cfg.py    Dataset paths and split ratios
configs/model_cfg.py   Model name, max length, dtype, attention backend
configs/train_cfg.py   Training hyperparameters and output paths
configs/fsdp2.yaml     Accelerate FSDP2 launch configuration
```

The default model is:

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
```

This means the tutorial performs continued full-parameter SFT on an existing
instruction-tuned model.

The default sequence length is:

```python
MAX_LENGTH = 4096
```

This is the training truncation length, not the model's theoretical maximum
context length.

## Data Pipeline

Run the data pipeline in order.

### 1. Download

```bash
python data_pipeline/download_dataset.py
```

This downloads:

```text
databricks/databricks-dolly-15k
```

and saves it to:

```text
data/raw
```

### 2. Split

```bash
python data_pipeline/split_dataset.py
```

The split is fixed by `SEED = 42`:

```text
Train       80%
Validation 10%
Test       10%
```

Outputs:

```text
data/train
data/validation
data/test

data/json/train.jsonl
data/json/validation.jsonl
data/json/test.jsonl
```

### 3. Tokenize

```bash
python data_pipeline/tokenization.py
```

For each sample, tokenization:

- builds Dolly chat messages in `data_pipeline/chat_messages.py`
- calls `tokenizer.apply_chat_template(...)`
- requests `return_assistant_tokens_mask=True`
- creates assistant-only labels

The label rule is:

```text
assistant token -> token id
non-assistant token -> -100
```

Outputs:

```text
data/train_tokenized
data/validation_tokenized
data/test_tokenized
```

## Training

Before training on a real GPU machine, generate or review the Accelerate config:

```bash
accelerate config --config_file configs/fsdp2.yaml
```

Then launch training:

```bash
accelerate launch --config_file configs/fsdp2.yaml trainer/training.py
```

The training loop:

- loads tokenized train and validation datasets
- uses a custom dynamic padding collator
- loads Qwen2.5-7B-Instruct with FlashAttention2
- uses AdamW
- uses cosine learning-rate decay with warmup
- runs for 3 epochs
- logs to TensorBoard
- saves multiple checkpoints

Checkpoints are written to:

```text
checkpoints/
```

Logs are written to:

```text
logs/
```

## TensorBoard

To inspect training logs:

```bash
tensorboard --logdir logs
```

## FSDP2 Notes

The project includes an example FSDP2 config:

```text
configs/fsdp2.yaml
```

Important fields:

```yaml
distributed_type: FSDP
mixed_precision: bf16
fsdp_config:
  fsdp_version: 2
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
```

`num_processes` should match the number of GPUs on the training machine.

For example, 4 GPUs:

```yaml
num_processes: 4
```

## Padding Strategy

This tutorial uses dynamic padding, not packing.

Tokenization stores variable-length examples:

```text
input_ids
attention_mask
labels
```

The collator pads each batch to the longest sequence in that batch. Label
padding uses `-100`, so padded tokens are ignored by the loss.

This route is simpler and avoids cross-sample attention issues that can appear
with naive sequence packing.

## Expected Command Order

```bash
python data_pipeline/download_dataset.py
python data_pipeline/split_dataset.py
python data_pipeline/tokenization.py
accelerate launch --config_file configs/fsdp2.yaml trainer/training.py
```

## Evaluation And Checkpoint Selection

The current training loop logs validation loss during training, but validation
loss should not be the final checkpoint selector for this tutorial's intended
workflow.

The intended later evaluation flow is:

```text
Generate responses
    -> LLM judge
    -> tournament
    -> best checkpoint
```

Those judge modules are not implemented yet.
