"""Load local Hugging Face Dataset.save_to_disk artifacts."""
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

from .config import IDENTITY_COLS

MODEL_COLUMNS = ["input_ids", "attention_mask", "label"]


def load_split(data_dir, split):
    path = Path(data_dir) / f"tokenized_{split}"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Copy the saved preprocessing output here "
            "or set --data-dir to its parent directory."
        )
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a single Dataset at {path}")
    missing = set(MODEL_COLUMNS + list(IDENTITY_COLS)) - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split} is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError(f"{split} is empty")
    labels = np.asarray(dataset["label"])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split} labels must be binary 0/1")
    for identity in IDENTITY_COLS:
        values = np.asarray(dataset[identity], dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{split}/{identity} must contain finite values in [0, 1]")
    return dataset


def prepare_training_dataset(dataset):
    labels = np.asarray(dataset["label"], dtype=np.int64)
    mentioned = (np.asarray(dataset["black"], dtype=float) >= 0.5).astype(np.int64)
    groups = 2 * mentioned + labels
    counts = np.bincount(groups, minlength=4)
    if (counts == 0).any():
        raise ValueError(f"All four training groups must be nonempty: {counts.tolist()}")
    if "group_id" in dataset.column_names:
        dataset = dataset.remove_columns("group_id")
    dataset = dataset.add_column("group_id", groups.tolist())
    return dataset.select_columns(MODEL_COLUMNS + ["group_id"]), counts
