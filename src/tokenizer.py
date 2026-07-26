"""
src/tokenizer.py

Loads the Phi-4 tokenizer and converts (prompt, target) pairs from
preprocess.py into tokenized (input_ids, attention_mask, labels) ready
for causal LM fine-tuning.

Key mechanics:
- Uses tokenizer.apply_chat_template() to format the system prompt +
  user instruction with Phi-4's own special tokens — we never hardcode
  <|...|> tokens ourselves, since guessing them wrong silently breaks
  training.
- The target summary (+ eos_token) is appended after the chat-formatted
  prompt to form the full training sequence.
- labels are a copy of input_ids with every prompt token (and every
  padding token) replaced by -100, so the loss is only computed on the
  summary tokens the model is meant to learn to generate.
"""

from transformers import AutoTokenizer
from datasets import DatasetDict

from src.config import get_config


class PhiTokenizer:
    """Thin wrapper around the Phi-4 tokenizer with sensible defaults set."""

    def __init__(self):
        cfg = get_config()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_model_id"],
            trust_remote_code=True,  # harmless no-op if not required; needed on some Phi tokenizer revisions
        )

        # Phi-4's tokenizer may not define a pad token by default.
        # Padding is required for batched training, so fall back to eos_token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


def _build_prompt_text(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Applies Phi-4's chat template to the system + user turns, with
    add_generation_prompt=True so the returned string ends exactly where
    the assistant's reply (our target summary) should begin.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize_example(example, tokenizer, cfg) -> dict:
    """
    Tokenizes a single (prompt, target) example into input_ids,
    attention_mask, and labels (with prompt + padding masked to -100).
    """
    max_len = cfg["max_seq_length"]
    system_prompt = cfg["system_prompt"]

    prompt_text = _build_prompt_text(tokenizer, system_prompt, example["prompt"])
    full_text = prompt_text + example["target"] + tokenizer.eos_token

    # Tokenize the prompt alone (no padding/truncation) just to find its length,
    # so we know how many leading tokens in the full sequence to mask.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), max_len)

    tokenized = tokenizer(
        full_text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    labels = list(input_ids)
    # Mask the prompt portion — loss should only be computed on the summary.
    for i in range(prompt_len):
        labels[i] = -100
    # Mask padding tokens too.
    for i, mask_val in enumerate(attention_mask):
        if mask_val == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_datasets(processed_datasets: DatasetDict) -> DatasetDict:
    """
    Takes the DatasetDict from preprocess.preprocess_datasets() (columns:
    prompt, target) and returns a DatasetDict with columns: input_ids,
    attention_mask, labels.
    """
    cfg = get_config()
    phi_tokenizer = PhiTokenizer()
    tokenizer = phi_tokenizer.tokenizer

    tokenized = {}
    for split_name, split_dataset in processed_datasets.items():
        tokenized[split_name] = split_dataset.map(
            lambda example: _tokenize_example(example, tokenizer, cfg),
            remove_columns=split_dataset.column_names,  # drop prompt/target
            desc=f"Tokenizing {split_name}",
        )

    return DatasetDict(tokenized)


if __name__ == "__main__":
    # Quick manual check: `python -m src.tokenizer` (or `!python -m src.tokenizer` in Colab)
    from src.data_loader import load_and_prepare_datasets
    from src.preprocess import preprocess_datasets

    datasets = load_and_prepare_datasets()
    processed_datasets = preprocess_datasets(datasets)
    tokenized_datasets = tokenize_datasets(processed_datasets)

    sample = tokenized_datasets["train"][0]
    print("Input IDs (first 30):", sample["input_ids"][:30])
    print("Labels (first 30):", sample["labels"][:30])
