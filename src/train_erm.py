"""
Train and evaluate the DistilBERT ERM baseline.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.groups import IDENTITY_COLUMNS, calculate_identity_group_metrics
from src.metrics import classification_metrics


CONFIG_PATH = Path("configs/erm.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return classification_metrics(labels, predictions)


def main():
    config = load_config()
    training_config = config["training"]
    model_config = config["model"]
    data_config = config["data"]

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_config["seed"])

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = load_from_disk(data_config["train_path"])
    val_dataset = load_from_disk(data_config["validation_path"])
    test_dataset = load_from_disk(data_config["test_path"])

    print(
        "Dataset sizes:",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        num_labels=model_config["num_labels"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["eval_batch_size"],
        num_train_epochs=training_config["num_epochs"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=training_config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting ERM training...")
    trainer.train()

    print("\nGenerating test predictions...")
    test_output = trainer.predict(test_dataset)
    logits = test_output.predictions
    predictions = np.argmax(logits, axis=-1)
    labels = np.asarray(test_dataset["label"])
    probabilities = softmax(logits, axis=1)

    overall_metrics = classification_metrics(labels, predictions)

    identity_data = {
        identity: np.asarray(test_dataset[identity])
        for identity in IDENTITY_COLUMNS
    }

    group_metrics, worst_group_accuracy = calculate_identity_group_metrics(
        labels,
        predictions,
        identity_data,
    )
    overall_metrics["worst_group_accuracy"] = float(worst_group_accuracy)

    print("\nFinal ERM test results:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value:.4f}")

    prediction_df = test_dataset.to_pandas()
    prediction_df["true_label"] = labels
    prediction_df["prediction"] = predictions
    prediction_df["prob_non_toxic"] = probabilities[:, 0]
    prediction_df["prob_toxic"] = probabilities[:, 1]
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    group_metrics.to_csv(output_dir / "test_group_metrics.csv", index=False)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {key: float(value) for key, value in overall_metrics.items()},
            file,
            indent=4,
        )

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    print("\nERM results saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
