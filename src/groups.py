"""
Group definitions and group-level evaluation utilities.

CivilComments identity definitions are centralized here so
ERM and Group DRO use exactly the same groups.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


IDENTITY_COLUMNS = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]


GROUP_NAMES = {
    0: "No identity, non-toxic",
    1: "Identity, non-toxic",
    2: "No identity, toxic",
    3: "Identity, toxic",
}


def training_group_id(identity_any, label):
    """
    Assign an example to one of four Group DRO
    training groups.

    Group 0:
        No identity + non-toxic

    Group 1:
        Identity + non-toxic

    Group 2:
        No identity + toxic

    Group 3:
        Identity + toxic
    """
    return int(identity_any) + 2 * int(label)


def calculate_identity_group_metrics(
    labels,
    predictions,
    identity_data
):
    """
    Calculate accuracy for every positive identity
    crossed with binary toxicity label.

    Parameters
    ----------
    labels : array-like
        Ground-truth toxicity labels.

    predictions : array-like
        Predicted toxicity labels.

    identity_data : dict
        Dictionary containing the binary identity
        annotations for each identity field.

    Returns
    -------
    group_df : pandas.DataFrame
        Accuracy and sample count for every evaluated
        identity-label group.

    worst_group_accuracy : float
        Minimum accuracy across evaluated groups.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    rows = []

    for identity in IDENTITY_COLUMNS:
        identity_values = np.asarray(identity_data[identity])

        for label_value in [0, 1]:
            mask = (
                (identity_values == 1)
                & (labels == label_value)
            )

            n = int(mask.sum())

            if n == 0:
                accuracy = np.nan
            else:
                accuracy = accuracy_score(
                    labels[mask],
                    predictions[mask]
                )

            rows.append({
                "identity": identity,
                "label": label_value,
                "label_name": (
                    "non-toxic"
                    if label_value == 0
                    else "toxic"
                ),
                "n": n,
                "accuracy": accuracy,
            })

    group_df = pd.DataFrame(rows)

    valid_groups = group_df.dropna(
        subset=["accuracy"]
    )

    worst_group_accuracy = valid_groups["accuracy"].min()

    return group_df, worst_group_accuracy