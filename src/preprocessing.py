"""
Text preprocessing for the ELE680 CivilComments project.

Loads CivilComments-WILDS, prepares binary labels, tokenizes
comments using DistilBERT, preserves metadata, and saves the
processed datasets.
"""

from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from src.data import get_dataframes
from src.groups import IDENTITY_COLUMNS


MODEL_NAME = "distilbert-base-uncased"

# Keep consistent with the completed experiments.
MAX_LENGTH = 128

OUTPUT_DIR = Path("data/processed")


def ensure_label(dataframe):
    """Ensure that the DataFrame contains a binary label column."""
    dataframe = dataframe.copy()

    if "label" in dataframe.columns:
        return dataframe

    if "y" in dataframe.columns:
        dataframe["label"] = dataframe["y"].astype(int)
        return dataframe

    if "toxicity" in dataframe.columns:
        dataframe["label"] = (dataframe["toxicity"] >= 0.5).astype(int)
        return dataframe

    raise ValueError("Could not find label, y, or toxicity column.")


def validate_metadata(dataframe):
    """Verify that identity metadata required for group evaluation is preserved."""
    required_columns = ["comment_text", "label", "identity_any"] + IDENTITY_COLUMNS

    missing = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def tokenize_dataframe(dataframe, tokenizer):
    """Convert a pandas DataFrame into a tokenized Hugging Face Dataset."""
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenize_batch, batched=True)


def save_dataset(dataset, name):
    """Save a processed Hugging Face Dataset."""
    path = OUTPUT_DIR / name
    dataset.save_to_disk(str(path))
    print(f"Saved {name} to {path}")


def main():
    print("Loading CivilComments-WILDS...")
    train_df, val_df, test_df = get_dataframes()

    print("Raw split sizes:", len(train_df), len(val_df), len(test_df))

    # Prepare labels.
    train_df = ensure_label(train_df)
    val_df = ensure_label(val_df)
    test_df = ensure_label(test_df)

    # Verify metadata before tokenization.
    validate_metadata(train_df)
    validate_metadata(val_df)
    validate_metadata(test_df)
    print("Required metadata verified.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Maximum sequence length: {MAX_LENGTH}")

    print("Tokenizing training data...")
    train_dataset = tokenize_dataframe(train_df, tokenizer)

    print("Tokenizing validation data...")
    val_dataset = tokenize_dataframe(val_df, tokenizer)

    print("Tokenizing test data...")
    test_dataset = tokenize_dataframe(test_df, tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_dataset(train_dataset, "tokenized_train")
    save_dataset(val_dataset, "tokenized_validation")
    save_dataset(test_dataset, "tokenized_test")

    print("\nPreprocessing complete.")
    print("Training columns:", train_dataset.column_names)


if __name__ == "__main__":
    main()
"""
Text preprocessing for the ELE680 CivilComments project.

Loads CivilComments-WILDS, prepares binary labels, tokenizes
comments using DistilBERT, preserves metadata, and saves the
processed datasets.
"""

from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from src.data import get_dataframes
from src.groups import IDENTITY_COLUMNS


MODEL_NAME = "distilbert-base-uncased"

# Keep consistent with the completed experiments.
MAX_LENGTH = 128

OUTPUT_DIR = Path("data/processed")


def ensure_label(dataframe):
    """Ensure that the DataFrame contains a binary label column."""
    dataframe = dataframe.copy()

    if "label" in dataframe.columns:
        return dataframe

    if "y" in dataframe.columns:
        dataframe["label"] = dataframe["y"].astype(int)
        return dataframe

    if "toxicity" in dataframe.columns:
        dataframe["label"] = (dataframe["toxicity"] >= 0.5).astype(int)
        return dataframe

    raise ValueError("Could not find label, y, or toxicity column.")


def validate_metadata(dataframe):
    """Verify that identity metadata required for group evaluation is preserved."""
    required_columns = ["comment_text", "label", "identity_any"] + IDENTITY_COLUMNS

    missing = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def tokenize_dataframe(dataframe, tokenizer):
    """Convert a pandas DataFrame into a tokenized Hugging Face Dataset."""
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenize_batch, batched=True)


def save_dataset(dataset, name):
    """Save a processed Hugging Face Dataset."""
    path = OUTPUT_DIR / name
    dataset.save_to_disk(str(path))
    print(f"Saved {name} to {path}")


def main():
    print("Loading CivilComments-WILDS...")
    train_df, val_df, test_df = get_dataframes()

    print("Raw split sizes:", len(train_df), len(val_df), len(test_df))

    # Prepare labels.
    train_df = ensure_label(train_df)
    val_df = ensure_label(val_df)
    test_df = ensure_label(test_df)

    # Verify metadata before tokenization.
    validate_metadata(train_df)
    validate_metadata(val_df)
    validate_metadata(test_df)
    print("Required metadata verified.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Maximum sequence length: {MAX_LENGTH}")

    print("Tokenizing training data...")
    train_dataset = tokenize_dataframe(train_df, tokenizer)

    print("Tokenizing validation data...")
    val_dataset = tokenize_dataframe(val_df, tokenizer)

    print("Tokenizing test data...")
    test_dataset = tokenize_dataframe(test_df, tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_dataset(train_dataset, "tokenized_train")
    save_dataset(val_dataset, "tokenized_validation")
    save_dataset(test_dataset, "tokenized_test")

    print("\nPreprocessing complete.")
    print("Training columns:", train_dataset.column_names)


if __name__ == "__main__":
    main()
