# ELE680 Final Project 
# Toxicity Detection and Worst-Group Performance

This repository contains the ongoing work for the ELE680 final project on toxicity classification using the CivilComments-WILDS dataset.

The project investigates whether a toxicity classifier can achieve strong overall performance while performing poorly on particular identity related groups. The final goal is to compare standard Empirical Risk Minimization (ERM) with Group Distributionally Robust Optimization (Group DRO) and analyze the trade-off between average and worst-group performance.

## Team Members

- Ricardo
- Ainaz

## Current Progress

The main analysis notebooks are maintained under `notebooks/`, while reusable code for data loading, preprocessing, evaluation, ERM training, and Group DRO training is organized under `src/`. Experiment configurations are stored in `configs/`, and generated metrics, predictions, figures, and comparison outputs are kept separately under `results/` and `figures/`.

Previous development work has been preserved in `_archive/`. This includes Ricardo's original training-set EDA as well as earlier preprocessing, ERM, and Group DRO notebooks. These files are retained for reference while the final pipeline is consolidated into a cleaner structure.

The current EDA builds on Ricardo's original analysis and extends it with validation-set comparisons, data-quality checks, and DistilBERT token-length analysis.

Ricardo's original GDRO implementation is currently retained under `src/` alongside the refactored Group DRO implementation. This allows both implementations to be reviewed and compared before selecting the final version used for submission.

The current project workflow is:

**EDA → Preprocessing → ERM Baseline → Group DRO → Model Comparison → Error Analysis**
## Project Status

**In progress**
