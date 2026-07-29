##### src/preprocess.py

"""
Converts the raw (article, highlights) DatasetDict from data_loader.py into (prompt, target) pairs ready for tokenization:
 - prompt: config['instruction_template'] with the article inserted
 - target: the reference summary (highlights)

NOTE: this module does NOT apply any chat-template special tokens (<|user|>, <|assistant|>, etc.)
— that happens in tokenizer.py via tokenizer.apply_chat_template(), using config['system_prompt'] and the 'prompt' field produced here as the user turn.
Keeping this separation means preprocess.py has no model-specific formatting logic at all.
"""

import re
from datasets import DatasetDict
from src.config import get_config


def _clean_article_text(text: str) -> str:
    """
    Light cleanup of raw CNN/DailyMail article text:
    - collapses repeated whitespace/newlines
    - strips leading/trailing whitespace
    Does NOT truncate length — truncation to max_seq_length happens later in tokenizer.py, 
    where it can be done in token space rather than character space.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_prompt(article: str, instruction_template: str) -> str:
    """
    Insert the cleaned article into the configured instruction template.
    """
    cleaned = _clean_article_text(article)
    return instruction_template.format(article=cleaned)


def preprocess_datasets(datasets: DatasetDict) -> DatasetDict:
    """
    Takes the DatasetDict produced by data_loader.load_and_prepare_datasets()
    (columns: article, highlights) and returns a new DatasetDict with columns: prompt, target.
    """
    cfg = get_config()
    article_col = cfg["article_column"]
    summary_col = cfg["summary_column"]
    instruction_template = cfg["instruction_template"]

    def _map_fn(example):
        return {
            "prompt": _build_prompt(example[article_col], instruction_template),
            "target": example[summary_col].strip(),
        }

    processed = {}
    for split_name, split_dataset in datasets.items():
        processed[split_name] = split_dataset.map(
            _map_fn,
            remove_columns=split_dataset.column_names,  # drop raw article/highlights
            desc=f"Preprocessing {split_name}",
        )

    return DatasetDict(processed)


if __name__ == "__main__":
    # Quick manual check: `python -m src.preprocess` (or `!python -m src.preprocess` in Colab)

    from src.data_loader import load_and_prepare_datasets
    datasets = load_and_prepare_datasets()
    processed_datasets = preprocess_datasets(datasets)

    sample = processed_datasets["train"][0]
    print("PROMPT:\n", sample["prompt"][:500])
    print("\nTARGET SUMMARY:\n", sample["target"])
