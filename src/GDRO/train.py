"""Server entry point: python -m GDRO [options]."""
import json
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, set_seed

from .collator import GroupDROCollator
from .config import GROUP_NAMES, parse_args
from .data import MODEL_COLUMNS, load_split, prepare_training_dataset
from .evaluate import evaluate_split
from .metrics import evaluate_predictions, make_compute_metrics
from .trainer import GroupDROTrainer

LOGGER = logging.getLogger(__name__)


def main(argv=None):
    config = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if any(config.output_dir.iterdir()) and config.resume_from_checkpoint is None:
        raise ValueError("Output directory is not empty. Choose a new --output-dir or resume a checkpoint.")
    set_seed(config.seed)
    train_metadata = load_split(config.data_dir, "train")
    val_metadata = load_split(config.data_dir, "validation")
    test_metadata = None if config.skip_test else load_split(config.data_dir, "test")
    train_dataset, counts = prepare_training_dataset(train_metadata)
    # Fail before training if an evaluation split lacks required subgroups.
    for metadata in (val_metadata, test_metadata):
        if metadata is not None:
            evaluate_predictions(metadata, metadata["label"], metadata["label"])
    LOGGER.info("Training groups: %s", dict(zip(GROUP_NAMES, counts.tolist())))
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name, local_files_only=config.local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2, local_files_only=config.local_files_only
    )
    args = TrainingArguments(
        output_dir=str(config.output_dir), learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs, weight_decay=config.weight_decay,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="worst_group_accuracy", greater_is_better=True,
        remove_unused_columns=False, logging_steps=config.logging_steps,
        save_total_limit=2, report_to="none", seed=config.seed, use_cpu=config.cpu,
    )
    if args.world_size != 1:
        raise ValueError("This implementation supports one training process. Use python -m GDRO on one GPU.")
    trainer = GroupDROTrainer(
        model=model, args=args, train_dataset=train_dataset,
        eval_dataset=val_metadata.select_columns(MODEL_COLUMNS), processing_class=tokenizer,
        data_collator=GroupDROCollator(tokenizer), compute_metrics=make_compute_metrics(val_metadata),
        n_groups=4, group_step_size=config.group_step_size,
    )
    with (config.output_dir / "run_config.json").open("w") as handle:
        json.dump(vars(config), handle, indent=2, default=str)
    LOGGER.info("PyTorch %s; training device: %s", torch.__version__, args.device)
    result = trainer.train(
        resume_from_checkpoint=str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
    )
    trainer.save_metrics("train", result.metrics)
    trainer.save_model(str(config.output_dir / "best_model"))
    tokenizer.save_pretrained(str(config.output_dir / "best_model"))
    with (config.output_dir / "selection.json").open("w") as handle:
        json.dump({"best_checkpoint": trainer.state.best_model_checkpoint,
                   "best_metric": trainer.state.best_metric,
                   "metric": "worst_identity_group_accuracy",
                   "end_of_training_q": trainer.q.cpu().tolist()}, handle, indent=2)
    for split, metadata in (("validation", val_metadata), ("test", test_metadata)):
        if metadata is not None:
            metrics = evaluate_split(trainer, metadata.select_columns(MODEL_COLUMNS),
                                     metadata, config.output_dir, split)
            LOGGER.info("%s metrics: %s", split, metrics)


if __name__ == "__main__":
    main()
