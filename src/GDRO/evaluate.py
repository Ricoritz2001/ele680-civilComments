"""Evaluate a selected model and export reproducible subgroup reports."""
import json
from pathlib import Path

import numpy as np

from .metrics import black_group_results, identity_group_results, make_compute_metrics


def evaluate_split(trainer, model_dataset, metadata, output_dir, split):
    original_metrics = trainer.compute_metrics
    try:
        trainer.compute_metrics = make_compute_metrics(metadata)
        output = trainer.predict(model_dataset, metric_key_prefix=split)
    finally:
        trainer.compute_metrics = original_metrics
    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(output.label_ids)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{split}_metrics.json").open("w") as handle:
        json.dump(output.metrics, handle, indent=2)
    black_group_results(metadata, labels, predictions).to_csv(
        directory / f"{split}_black_groups.csv", index=False
    )
    identity_group_results(metadata, labels, predictions).sort_values("accuracy").to_csv(
        directory / f"{split}_identity_groups.csv", index=False
    )
    np.savez(directory / f"{split}_predictions.npz", labels=labels, predictions=predictions)
    return output.metrics
