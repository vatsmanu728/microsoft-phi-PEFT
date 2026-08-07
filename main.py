#!/usr/bin/env python3
##### main.py

"""
Conceptual entry point for the Phi-4 summarization pipeline. Run this to
go from raw config all the way to a trained (or already-trained) model
with reported ROUGE scores.

Usage:
    python main.py
"""

import os

from src.config import get_config
from src.data_loader import load_and_prepare_datasets
from src.preprocess import preprocess_datasets
from src.tokenizer import PhiTokenizer, tokenize_datasets
from src.model import load_lora_model
from src.train import train_model
from src.evaluate import evaluate_model, load_finetuned_model_and_tokenizer


def lora_adapters_exist(path: str) -> bool:
    """
    Checks whether the FINAL trained LoRA adapters exist — specifically
    looks for adapter_config.json at the top level, NOT just any content
    in the directory. A mid-training checkpoint-N subfolder existing
    inside path does NOT count as a completed adapter; the old
    "any content" check incorrectly returned True mid-training, causing
    this script to skip training and fail during evaluation.
    """
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def main():
    cfg = get_config()
    print("Config loaded. on_kaggle:", cfg["on_kaggle"], "| on_colab:", cfg["on_colab"])

    print("\nLoading and preparing datasets...")
    datasets = load_and_prepare_datasets()
    print("Train size:", len(datasets["train"]))
    print("Validation size:", len(datasets["validation"]))
    print("Test size:", len(datasets["test"]))

    print("\nPreprocessing (building prompt/target pairs)...")
    processed_datasets = preprocess_datasets(datasets)

    lora_exists = lora_adapters_exist(cfg["output_dir"])
    print("\nFinal LoRA adapters found at output_dir:", lora_exists)

    if not lora_exists:
        print("\nNo final adapters found. Tokenizing for training...")
        tokenized_datasets = tokenize_datasets(processed_datasets)
        phi_tokenizer = PhiTokenizer()

        print("\nLoading base model + attaching fresh LoRA adapters...")
        model = load_lora_model()

        print("\nStarting training...")
        train_model(
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=phi_tokenizer.tokenizer,
            model=model,
        )
        print("Training completed. Adapters saved to:", cfg["output_dir"])
    else:
        print("\nSkipping training — using existing final LoRA adapters.")

    print("\nLoading fine-tuned model (base + adapters) for evaluation...")
    model, tokenizer = load_finetuned_model_and_tokenizer(cfg)

    print("\nRunning evaluation...")
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
