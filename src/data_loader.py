"""
src/data_loader.py

Loads the CNN/DailyMail train/validation/test CSVs (id, article, highlights)
and subsamples them to the sizes defined in config.py, returning a single
Hugging Face DatasetDict ready for preprocess.py.
"""

import pandas as pd
from datasets import Dataset, DatasetDict

from src.config import get_config


def _load_and_subsample(csv_path: str, n_samples: int, seed: int) -> pd.DataFrame:
    """
    Load a CSV and return a reproducible random subsample of n_samples rows.
    If the CSV has fewer rows than n_samples, the whole file is returned
    (with a warning) rather than raising an error.
    """
    df = pd.read_csv(csv_path)

    if len(df) < n_samples:
        print(
            f"Warning: {csv_path} has only {len(df)} rows, "
            f"fewer than the requested {n_samples}. Using all available rows."
        )
        n_samples = len(df)

    return df.sample(n=n_samples, random_state=seed).reset_index(drop=True)


def load_and_prepare_datasets() -> DatasetDict:
    """
    Reads train/validation/test CSVs from config['dataset_raw_dir'],
    subsamples each split to the configured size, and returns a
    DatasetDict with keys: 'train', 'validation', 'test'.
    """
    cfg = get_config()
    seed = cfg["seed"]

    article_col = cfg["article_column"]
    summary_col = cfg["summary_column"]

    splits = {
        "train": (cfg["train_file"], cfg["train_samples"]),
        "validation": (cfg["validation_file"], cfg["validation_samples"]),
        "test": (cfg["test_file"], cfg["test_samples"]),
    }

    dataset_dict = {}

    for split_name, (csv_path, n_samples) in splits.items():
        df = _load_and_subsample(csv_path, n_samples, seed)

        missing_cols = [c for c in (article_col, summary_col) if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{csv_path} is missing expected column(s) {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        df = df[[article_col, summary_col]]
        dataset_dict[split_name] = Dataset.from_pandas(df, preserve_index=False)

    return DatasetDict(dataset_dict)


if __name__ == "__main__":
    datasets = load_and_prepare_datasets()

    print("Train size:", len(datasets["train"]))
    print("Validation size:", len(datasets["validation"]))
    print("Test size:", len(datasets["test"]))
    print("\nSample row:\n", datasets["train"][0])
