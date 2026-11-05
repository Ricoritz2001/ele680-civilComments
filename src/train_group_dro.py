"""
Train DistilBERT by using Group DRO.

The training groups are defined by identity_any x toxicity label,
creating four mutually exclusive groups.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.groups import (
    GROUP_NAMES,
    IDENTITY_COLUMNS,
    calculate_identity_group_metrics,
    training_group_id,
)
from src.metrics import classification_metrics


CONFIG_PATH = Path("configs/group_dro.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_group_id(example):
    return {
        "group_id": training_group_id(
            example["identity_any"],
            example["label"],
        )
    }


def create_collator(tokenizer):
    padding_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    def collate(features):
        model_features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in features
        ]
        batch = padding_collator(model_features)

        batch["labels"] = torch.tensor(
            [int(item["label"]) for item in features],
            dtype=torch.long,
        )
        batch["group_id"] = torch.tensor(
            [int(item["group_id"]) for item in features],
            dtype=torch.long,
        )

        for identity in IDENTITY_COLUMNS:
            batch[identity] = torch.tensor(
                [int(item[identity]) for item in features],
                dtype=torch.long,
            )

        return batch

    return collate


def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    labels_all = []
    predictions_all = []
    probabilities_all = []
    identity_data = {identity: [] for identity in IDENTITY_COLUMNS}

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            losses = loss_fn(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            total_loss += losses.sum().item()
            total_examples += len(labels)

            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(predictions.cpu().numpy())
            probabilities_all.extend(probabilities.cpu().numpy())

            for identity in IDENTITY_COLUMNS:
                identity_data[identity].extend(batch[identity].numpy())

    labels_all = np.asarray(labels_all)
    predictions_all = np.asarray(predictions_all)
    probabilities_all = np.asarray(probabilities_all)

    metrics = classification_metrics(labels_all, predictions_all)
    metrics["loss"] = total_loss / total_examples if total_examples else float("nan")

    group_metrics, worst_group_accuracy = calculate_identity_group_metrics(
        labels_all,
        predictions_all,
        {key: np.asarray(value) for key, value in identity_data.items()},
    )
    metrics["worst_group_accuracy"] = float(worst_group_accuracy)

    return {
        "metrics": metrics,
        "labels": labels_all,
        "predictions": predictions_all,
        "probabilities": probabilities_all,
        "group_metrics": group_metrics,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    adv_probs,
    dro_eta,
    device,
):
    model.train()
    total_robust_loss = 0.0
    number_groups = len(GROUP_NAMES)

    for batch in tqdm(loader, desc="Group DRO"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        group_ids = batch["group_id"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        example_losses = loss_fn(outputs.logits, labels)

        group_losses = torch.zeros(number_groups, device=device)
        group_present = torch.zeros(number_groups, dtype=torch.bool, device=device)

        for group_id in range(number_groups):
            mask = group_ids == group_id
            if mask.any():
                group_present[group_id] = True
                group_losses[group_id] = example_losses[mask].mean()

        with torch.no_grad():
            adv_probs[group_present] *= torch.exp(
                dro_eta * group_losses[group_present].detach()
            )
            adv_probs /= adv_probs.sum()

        weights = adv_probs[group_present]
        weights = weights / weights.sum()
        robust_loss = torch.sum(weights * group_losses[group_present])

        robust_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_robust_loss += robust_loss.item()

    return total_robust_loss / len(loader) if len(loader) else float("nan")


def main():
    config = load_config()
    training = config["training"]
    dro_config = config["group_dro"]
    model_config = config["model"]
    paths = config["data"]

    output_dir = Path(config["output"]["directory"])
    best_model_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = load_from_disk(paths["train_path"]).map(add_group_id)
    val_dataset = load_from_disk(paths["validation_path"]).map(add_group_id)
    test_dataset = load_from_disk(paths["test_path"]).map(add_group_id)

    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    collator = create_collator(tokenizer)

    train_groups = np.asarray(train_dataset["group_id"])
    number_groups = len(GROUP_NAMES)
    group_counts = np.bincount(train_groups, minlength=number_groups)

    if np.any(group_counts == 0):
        raise ValueError(
            f"At least one training group is empty. Group counts: {group_counts.tolist()}"
        )

    sample_weights = np.asarray(
        [1.0 / group_counts[group] for group in train_groups],
        dtype=np.float64,
    )

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        sampler=sampler,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        num_labels=model_config["num_labels"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    total_steps = len(train_loader) * training["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    adv_probs = torch.ones(number_groups, device=device) / number_groups

    best_worst_group = -np.inf
    best_epoch = None
    history = []

    for epoch in range(1, training["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{training['num_epochs']}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            adv_probs,
            dro_config["eta"],
            device,
        )

        validation = evaluate(model, val_loader, loss_fn, device)
        validation_metrics = validation["metrics"]
        print("Validation:", validation_metrics)

        history.append(
            {
                "epoch": epoch,
                "train_robust_loss": train_loss,
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
        )

        current_worst = validation_metrics["worst_group_accuracy"]
        if current_worst > best_worst_group:
            best_worst_group = current_worst
            best_epoch = epoch
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print("Saved new best model.")

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

    print("Best epoch:", best_epoch)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir
    ).to(device)

    test_results = evaluate(best_model, test_loader, loss_fn, device)
    metrics = test_results["metrics"]

    print("\nFinal Group DRO test results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    prediction_df = test_dataset.to_pandas()
    prediction_df["true_label"] = test_results["labels"]
    prediction_df["prediction"] = test_results["predictions"]
    prediction_df["prob_non_toxic"] = test_results["probabilities"][:, 0]
    prediction_df["prob_toxic"] = test_results["probabilities"][:, 1]
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    test_results["group_metrics"].to_csv(
        output_dir / "test_group_metrics.csv",
        index=False,
    )

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump({key: float(value) for key, value in metrics.items()}, file, indent=4)

    print("Results saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
