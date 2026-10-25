# Group DRO on a server

Run from the repository root with Python 3.11. No Google Drive mount or ERM run is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r GDRO/requirements.txt
python -m GDRO --data-dir /path/to/data/processed --output-dir /path/to/results/gdro
```

Copy the complete directories saved by `02TextPre.ipynb` onto the server, including all Arrow and metadata files:

```text
/path/to/data/processed/
  tokenized_train/
  tokenized_validation/
  tokenized_test/
```

The defaults read `data/processed` and write `results/groupdro_distilbert` relative to the repository, independently of the shell's working directory. Explicit relative paths are relative to the shell's working directory. The datasets must include `input_ids`, `attention_mask`, binary `label`, and the eight identity columns from preprocessing. Tokenization must match the chosen model/tokenizer; this script does not tokenize raw text.

The first run downloads DistilBERT and its tokenizer. For offline servers, copy a compatible Hugging Face model/tokenizer directory and use:

```bash
python -m GDRO --model-name /path/to/distilbert --local-files-only --data-dir /path/to/data/processed
```

Use `python -m GDRO --help` for hyperparameters. Defaults preserve the notebook: four Black-mention × toxicity training groups, group-balanced sampling, step size 0.01, learning rate 2e-5, three epochs, and weight decay 0.01. Checkpoint selection uses the worst accuracy across the 16 overlapping identity–label subgroups. Reports also include the minimum over the four training-defined groups. Evaluation requires all these groups to be nonempty.

Use one GPU/process; CPU is supported with `--cpu`. Multi-process training is rejected because the group statistics and weights are not synchronized across processes.

For tuning, omit test evaluation:

```bash
python -m GDRO --skip-test --output-dir results/gdro_trial_1
```

For the final chosen settings, run with test evaluation enabled. Choose a fresh output directory for each experiment. Interrupted runs can resume from a retained checkpoint, restoring model, optimizer, scheduler, and Group DRO weights:

```bash
python -m GDRO --output-dir results/gdro_trial_1 --resume-from-checkpoint results/gdro_trial_1/checkpoint-1000 --skip-test
```

Resume with the same data, model, seed, sampling, and training settings. `best_model/` is an export for inference, not a resumable training checkpoint. `end_of_training_q` in `selection.json` describes the last training step and may differ from weights at the best checkpoint.

Outputs include `run_config.json`, `selection.json`, checkpoints, `best_model/`, training metrics, and for each evaluated split:

- `*_metrics.json`: overall classification metrics and both worst-group accuracies.
- `*_black_groups.csv`: four group accuracies and sizes.
- `*_identity_groups.csv`: 16 subgroup accuracies and sizes, sorted by accuracy.
- `*_predictions.npz`: labels and predictions in source dataset row order.

Modules: `config.py` defines configuration; `data.py` validates local datasets and builds groups; `collator.py` pads batches; `trainer.py` implements Group DRO and checkpoint state; `metrics.py` computes subgroup metrics; `evaluate.py` exports reports; `train.py` coordinates the run. The notebook remains available independently.

Run the offline synthetic integration tests with:

```bash
python -m unittest discover -s GDRO/tests -v
```

Method reference: [Sagawa et al. (2020)](https://arxiv.org/abs/1911.08731), with the exponential group-weight update also implemented in the [authors' repository](https://github.com/kohpangwei/group_DRO).
