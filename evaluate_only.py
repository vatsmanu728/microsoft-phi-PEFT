#!/usr/bin/env python3
##### evaluate_only.py

"""
Standalone evaluation script. Loads already-trained LoRA adapters from
config['output_dir'] and reports ROUGE-1/2/L scores on the test set.

Unlike main.py, this script does NOT fall back to training — it assumes adapters already exist and fails clearly if they don't.
Use this when you just want to check results without risking an accidental training run.

Usage:
    python evaluate_only.py (or `!python evaluate_only.py` in Colab, from the project root)
"""

import os
import sys

from src.config import get_config
from src.data_loader import load_and_prepare_datasets
from src.preprocess import preprocess_datasets
from src.evaluate import evaluate_model, load_finetuned_model_and_tokenizer


def _adapters_exist(path: str) -> bool:
    """
    Checks specifically for the FINAL adapter_config.json at the top
    level — not just any content in the directory (a checkpoint-N
    subfolder alone doesn't count as the completed adapter).
    """
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def main():
    cfg = get_config()

    if not _adapters_exist(cfg["output_dir"]):
        print(f"No final LoRA adapters found at: {cfg['output_dir']}")
        print("Run main.py (or let src/train.py finish) first.")
        sys.exit(1)

    print("Loading test data...")
    datasets = load_and_prepare_datasets()
    processed_datasets = preprocess_datasets(datasets)

    print(f"Loading fine-tuned model from: {cfg['output_dir']}...")
    model, tokenizer = load_finetuned_model_and_tokenizer(cfg)

    print("Running evaluation...")
    results = evaluate_model(
        test_dataset=processed_datasets["test"],
        num_samples=cfg["eval_num_samples"],
        model=model,
        tokenizer=tokenizer,
    )

    print("\nFinal ROUGE scores:", results)
    return results


if __name__ == "__main__":
    main()
