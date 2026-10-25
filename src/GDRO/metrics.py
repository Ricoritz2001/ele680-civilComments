"""Split-specific overall and subgroup evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import IDENTITY_COLS, GROUP_NAMES

def identity_group_results(dataset, labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(dataset) != len(labels) or len(labels) != len(predictions):
        raise ValueError("Metadata, labels, and predictions must have the same length")

    records = []
    correct = predictions == labels
    for identity in IDENTITY_COLS:
        mentioned = np.asarray(dataset[identity], dtype=float) >= 0.5
        for toxicity in (0, 1):
            mask = mentioned & (labels == toxicity)
            count = int(mask.sum())
            if count:
                records.append({
                    "identity": identity,
                    "label": toxicity,
                    "n": count,
                    "accuracy": float(correct[mask].mean()),
                })
    return pd.DataFrame(records)


def black_group_results(dataset, labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(dataset) != len(labels) or len(labels) != len(predictions):
        raise ValueError("Metadata, labels, and predictions must have the same length")
    mentioned = (np.asarray(dataset["black"], dtype=float) >= 0.5).astype(int)
    ids = 2 * mentioned + labels
    records = []
    for group_id, name in enumerate(GROUP_NAMES):
        mask = ids == group_id
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Empty evaluation group: {name}")
        records.append({"group_id": group_id, "group": name, "n": count,
                        "accuracy": float((predictions[mask] == labels[mask]).mean())})
    return pd.DataFrame(records)


def evaluate_predictions(dataset, labels, predictions):
    groups = identity_group_results(dataset, labels, predictions)
    if len(groups) != 16:
        raise ValueError(f"Expected 16 nonempty identity-label subgroups, got {len(groups)}")
    black_groups = black_group_results(dataset, labels, predictions)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        # Retain the existing 16-subgroup checkpoint-selection metric.
        "worst_group_accuracy": float(groups["accuracy"].min()),
        "worst_identity_group_accuracy": float(groups["accuracy"].min()),
        "worst_black_group_accuracy": float(black_groups["accuracy"].min()),
    }


def make_compute_metrics(dataset):
    # Bind metrics to the matching split; never use validation metadata for test.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.asarray(logits).argmax(axis=-1)
        return evaluate_predictions(dataset, np.asarray(labels), predictions)
    return compute_metrics
