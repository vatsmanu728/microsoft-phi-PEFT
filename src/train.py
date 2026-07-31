##### src/train.py

"""
Trains the LoRA-wrapped Phi-4 model using TrainingArguments/Trainer, with all hyperparameters pulled from config.py.
Saves the resulting LoRA adapters (not the full model) to config['output_dir'].
- fp16 (not bf16) is used for training precision —
  T4 is a Turing-architecture GPU with genuine fp16 Tensor Core support but only
  weak/emulated bf16 support.

Resilience note:
save_strategy/save_steps are deliberately overridden here to "steps"/50, rather than using config.py's "epoch" setting.
With num_train_epochs=1, "epoch"-based saving only checkpoints once, at the very end
— meaning an interrupted Colab runtime (GPU quota, disconnect) before completion would lose 100% of progress.
Saving every ~50 steps means at most ~50 steps are ever at risk, and
train_model() automatically detects and resumes from the latest checkpoint if one exists.
"""

import os

from transformers import TrainingArguments, Trainer, default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from src.config import get_config
from src.model import load_lora_model

import datasets.config as ds_config
ds_config.TORCHVISION_AVAILABLE = False  # Starting with Torchvision 0.26, VideoReader has been removed from torchvision.io


def _build_training_args(cfg: dict) -> TrainingArguments:
    # Builds TrainingArguments (but not entirely from config.py settings)
    # save_strategy overridden to step-based checkpointing for interruption resilience.

    return TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        # save_strategy=cfg["save_strategy"],# overridden — to account for quota limit
        save_strategy="steps",               # added steps
        save_steps=50,                       # 50 steps
        eval_strategy=cfg["eval_strategy"],
        save_total_limit=cfg["save_total_limit"],
        fp16=cfg["fp16"],           # new — T4-appropriate precision
        bf16=cfg["bf16"],           # new — explicitly False on T4
        gradient_checkpointing=cfg["gradient_checkpointing"],
        optim=cfg["optim"],
        seed=cfg["seed"],
        report_to="none",
    )


def train_model(train_dataset, eval_dataset, tokenizer, model=None):
    # Trains the LoRA adapters on train_dataset, evaluating on eval_dataset.
    # If model is not provided, loads a fresh one via load_lora_model().
    # Automatically detects and resumes from the latest checkpoint in
    # output_dir if one exists (e.g. after an interrupted prior run).
    # Saves final adapters + tokenizer to config['output_dir'] when done.

    cfg = get_config()

    if model is None:
        model = load_lora_model()


    # Tokenized datasets come in as plain Python lists (from .map())
    # — convert to torch tensors so the Trainer can batch them directly.
    train_dataset = train_dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset = eval_dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    training_args = _build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(cfg["output_dir"]):
        last_checkpoint = get_last_checkpoint(cfg["output_dir"])
        if last_checkpoint is not None:
            print(f"Found existing checkpoint: {last_checkpoint} — resuming from here.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"LoRA adapters saved to {cfg['output_dir']}")

    return trainer


if __name__ == "__main__":
    # Manual run: `python -m src.train` (or `!python -m src.train` in Colab)
    # WARNING: this runs the FULL training loop (10,000 examples)
    # Do not invoke this directly for testing — use the smoke test instead.

    from src.data_loader import load_and_prepare_datasets
    from src.preprocess import preprocess_datasets
    from src.tokenizer import PhiTokenizer, tokenize_datasets

    datasets = load_and_prepare_datasets()
    processed = preprocess_datasets(datasets)
    tokenized = tokenize_datasets(processed)

    phi_tok = PhiTokenizer()

    train_model(
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=phi_tok.tokenizer,
    )
