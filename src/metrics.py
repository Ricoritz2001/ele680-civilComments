"""
Evaluation metrics shared by ERM and Group DRO.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def classification_metrics(
    labels,
    predictions,
):
    """
    Calculate standard binary classification metrics.

    Parameters
    ----------
    labels : array-like
        Ground-truth labels.

    predictions : array-like
        Predicted labels.

    Returns
    -------
    dict
        Accuracy, precision, recall, and F1.
    """
    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
    }
