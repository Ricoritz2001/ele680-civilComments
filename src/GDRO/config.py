"""Command-line configuration and shared group definitions."""
import argparse
from pathlib import Path

IDENTITY_COLS = (
    "male", "female", "LGBTQ", "christian", "muslim",
    "other_religions", "black", "white",
)
GROUP_NAMES = (
    "No Black mention / Non-toxic", "No Black mention / Toxic",
    "Black mentioned / Non-toxic", "Black mentioned / Toxic",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/groupdro_distilbert")
    parser.add_argument("--model-name", default="distilbert-base-uncased",
                        help="Hugging Face model ID or local model directory; must match data tokenization.")
    parser.add_argument("--tokenizer-name", default=None,
                        help="Tokenizer ID/directory, if different from --model-name.")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--group-step-size", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--skip-test", action="store_true",
                        help="Use during validation-only hyperparameter tuning.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load model/tokenizer from local files or cache without downloading.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    for name in ("learning_rate", "epochs", "train_batch_size", "eval_batch_size", "logging_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.group_step_size < 0 or args.weight_decay < 0:
        parser.error("Group step size and weight decay must be nonnegative")
    return args
