"""
Dataset loading utilities for the ELE680 CivilComments project.
"""

from datasets import load_dataset


DATASET_NAME = "pietrolesci/civilcomments-wilds"
DATASET_CONFIG = "default"


def load_civilcomments():
    """
    Load the CivilComments-WILDS dataset from Hugging Face.

    Returns
    -------
    DatasetDict
        Dataset containing train, validation, and test splits.
    """
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def get_dataframes():
    """
    Load CivilComments-WILDS and return each split
    as a pandas DataFrame.

    Returns
    -------
    tuple
        train_df, val_df, test_df
    """
    dataset = load_civilcomments()

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, val_df, test_df