"""
Full fine-tuning training loop for Qwen SFT.

Usage:
    accelerate launch --config_file configs/fsdp2.yaml trainer/training.py
"""
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from configs.data_cfg import TRAIN_TOKENIZED_DIR, VAL_TOKENIZED_DIR
from configs.model_cfg import ATTN_IMPLEMENTATION, DTYPE, MODEL_NAME
from configs.train_cfg import (
    EVAL_EVERY_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOG_DIR,
    LOG_EVERY_STEPS,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from trainer.collator import SFTDataCollator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype() -> torch.dtype:
    if DTYPE == "bfloat16":
        return torch.bfloat16
    if DTYPE == "float16":
        return torch.float16
    if DTYPE == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {DTYPE}")


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model(accelerator):
    # 仅 rank0 下载
    if accelerator.is_main_process:
        _ = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        del _

    # 等待下载完成
    accelerator.wait_for_everyone()

    # 所有 rank 从本地缓存加载
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=get_dtype(),
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,
        low_cpu_mem_usage=False,
    )

    return model

def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )


def build_dataloader(
    dataset_dir: Path,
    batch_size: int,
    collator: SFTDataCollator,
    shuffle: bool,
) -> DataLoader:
    dataset = load_from_disk(dataset_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
) -> float:
    model.eval()

    losses = []

    for batch in eval_dataloader:
        outputs = model(**batch)
        loss = outputs.loss

        gathered_loss = accelerator.gather_for_metrics(
            loss.detach()
        )
        losses.append(gathered_loss)

    model.train()

    if not losses:
        return float("nan")

    losses = torch.cat(losses)
    return losses.mean().item()


def save_checkpoint(
    accelerator: Accelerator,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    step: int,
) -> None:
    checkpoint_dir = output_dir / f"step_{step}"
    accelerator.save_state(checkpoint_dir)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    accelerator.wait_for_everyone()


def main() -> None:
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with="tensorboard",
        project_dir=LOG_DIR,
    )
    model = load_model(accelerator)

    tokenizer = load_tokenizer()
    collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataloader = build_dataloader(
        TRAIN_TOKENIZED_DIR,
        PER_DEVICE_TRAIN_BATCH_SIZE,
        collator,
        shuffle=True,
    )
    eval_dataloader = build_dataloader(
        VAL_TOKENIZED_DIR,
        PER_DEVICE_EVAL_BATCH_SIZE,
        collator,
        shuffle=False,
    )

    
    optimizer = build_optimizer(model)

    update_steps_per_epoch = math.ceil(
        len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS
    )
    total_training_steps = NUM_EPOCHS * update_steps_per_epoch
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    )

    accelerator.init_trackers("enterprise_sft")

    global_step = 0
    model.train()

    for epoch in range(NUM_EPOCHS):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        MAX_GRAD_NORM,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0:
                train_loss = loss.detach().float().item()
                accelerator.print(
                    f"epoch={epoch} step={global_step} loss={train_loss:.4f}"
                )
                accelerator.log(
                    {
                        "train/loss": train_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            if global_step % EVAL_EVERY_STEPS == 0:
                eval_loss = evaluate(model, eval_dataloader, accelerator)
                accelerator.print(
                    f"epoch={epoch} step={global_step} eval_loss={eval_loss:.4f}"
                )
                accelerator.log(
                    {
                        "eval/loss": eval_loss,
                    },
                    step=global_step,
                )

                save_checkpoint(
                    accelerator=accelerator,
                    tokenizer=tokenizer,
                    output_dir=OUTPUT_DIR,
                    step=global_step,
                )

    accelerator.end_training()


if __name__ == "__main__":
    main()
