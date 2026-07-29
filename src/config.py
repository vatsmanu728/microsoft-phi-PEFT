##### src/config.py

# Central configuration for the Phi-4 Text Summarization project
# (repo: microsoft-phi-PEFT).

"""
Design goals:
- Single source of truth for paths and hyperparameters.
- Auto-detects whether it's running on Kaggle, Colab, or locally, and adjusts paths accordingly.
- Phi-4 is fully open-weight (MIT license) on Hugging Face — no auth token, no gated-access request, no login step required anywhere in this file.
- Every other module (data_loader, preprocess, tokenizer, model, train, evaluate) should pull settings from here rather than hardcoding values.
"""

import os


def _is_kaggle() -> bool:
    """
    Detect if we're running inside a Kaggle notebook environment.

    NOTE: we deliberately do NOT check os.path.exists("/kaggle/input") —
    Colab's container image pre-creates an empty /kaggle folder as part of its own Kaggle-dataset-import integration, which causes false positives there.
    KAGGLE_KERNEL_RUN_TYPE is only ever set when code is actually executing inside a real Kaggle kernel, so it's the reliable signal.
    """
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def _is_colab() -> bool:
    """Detect if we're running inside a Google Colab environment."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _project_root() -> str:
    """Root of the project (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config() -> dict:
    root = _project_root()
    on_kaggle = _is_kaggle()
    on_colab = _is_colab()

    # --- Dataset paths ---
    if on_kaggle:
        dataset_raw_dir = "/kaggle/input/cnn-dailymail-summarization"
        dataset_processed_dir = "/kaggle/working/dataset/processed"
    else:
        dataset_raw_dir = os.path.join(root, "dataset", "raw")
        dataset_processed_dir = os.path.join(root, "dataset", "processed")

    # --- Model paths ---
    base_model_id = "microsoft/phi-4"

    if on_kaggle:
        base_model_dir = "/kaggle/working/models/base"
        lora_output_dir = "/kaggle/working/models/lora"
    elif on_colab:
        base_model_dir = "/content/models/base"
        drive_root = "/content/drive/MyDrive/microsoft-phi-PEFT"
        lora_output_dir = os.path.join(drive_root, "models", "lora")
    else:
        base_model_dir = os.path.join(root, "models", "base")
        lora_output_dir = os.path.join(root, "models", "lora")

    config = {
        "on_kaggle": on_kaggle,
        "on_colab": on_colab,
        "project_root": root,

        "dataset_raw_dir": dataset_raw_dir,
        "dataset_processed_dir": dataset_processed_dir,
        "train_file": os.path.join(dataset_raw_dir, "train.csv"),
        "validation_file": os.path.join(dataset_raw_dir, "validation.csv"),
        "test_file": os.path.join(dataset_raw_dir, "test.csv"),

        "train_samples": 10000,
        "validation_samples": 500,
        "test_samples": 500,

        "article_column": "article",
        "summary_column": "highlights",

        "base_model_id": base_model_id,
        "base_model_dir": base_model_dir,
        "output_dir": lora_output_dir,

        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",

        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_bias": "none",
        "lora_task_type": "CAUSAL_LM",

        "max_seq_length": 1024,
        "max_target_length": 256,
        "padding": "max_length",
        "truncation": True,

        "system_prompt": "You are a helpful assistant that writes concise, accurate summaries of news articles.",
        "instruction_template": "Summarize the following news article.\n\nArticle:\n{article}",

        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "epoch",    # use save_strategy="steps", save_steps=50, if quota limit
        "eval_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "seed": 42,

        "eval_num_samples": 500,
        "generation_max_new_tokens": 128,
        "do_sample": False,
        "rouge_types": ["rouge1", "rouge2", "rougeL"],
    }

    return config


if __name__ == "__main__":
    import json
    cfg = get_config()
    print(json.dumps(cfg, indent=2))
