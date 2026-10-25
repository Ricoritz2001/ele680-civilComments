# ELE680 Final Project 
# Toxicity Detection and Worst-Group Performance

This repository contains the ongoing work for the ELE680 final project on toxicity classification using the CivilComments-WILDS dataset.

The project investigates whether a toxicity classifier can achieve strong overall performance while performing poorly on particular identity related groups. The final goal is to compare standard Empirical Risk Minimization (ERM) with Group Distributionally Robust Optimization (Group DRO) and analyze the trade-off between average and worst-group performance.

## Team Members

- Ricardo
- Ainaz

## Current Progress

### `eda.ipynb`

Initial exploratory data analysis.

Current work includes:

- Exploration of the training set
- Inspection of dataset structure and features
- Statistical analysis of the training data
- Initial analysis of feature and label distributions

### `01EDR.ipynb`

Exploratory data and identity-group analysis.

Current work includes:

- Exploration of the training, validation, and test splits
- Inspection of columns and data types
- Missing-value and duplicate checks
- Toxic and non-toxic label distributions
- Comment-length analysis
- Identity metadata exploration
- Identity-group distributions across the three splits
- Toxicity rates across identity groups

### `02TextPre.ipynb`

Text and model-input preparation.

Current work includes:

- Binary toxicity-label preparation
- DistilBERT tokenizer setup
- Token-length analysis
- Maximum sequence-length configuration
- Tokenization of training, validation, and test data
- Preservation of identity metadata for later group-level evaluation
- Saving the tokenized datasets for model training

### `03ERM.ipynb`

Standard ERM baseline experiment.

Current work includes:

- DistilBERT sequence-classification model
- Standard Empirical Risk Minimization training
- Training and validation evaluation
- Overall classification metrics
- Generation of test predictions for later group-level analysis


## Models and Methods

The current baseline uses:

- Model: `distilbert-base-uncased`
- Task: Binary toxicity classification
- Non-toxic label: `0`
- Toxic label: `1`
- Baseline training method: ERM
- Planned robust method: Group DRO

The model architecture will remain the same for ERM and Group DRO so that the effect of the training method can be compared fairly.

## Dataset

The project uses the CivilComments-WILDS dataset. The dataset contains:

- Online comment text
- Binary toxicity labels
- Training, validation, and test splits
- Identity-related metadata used for group-level evaluation

## Project Status

**In progress**
