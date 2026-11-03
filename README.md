# Toxicity Classification

This repository contains the code and analysis for an ELE680 project on toxicity classification and worst-group performance using CivilComments-WILDS. The project compares standard Empirical Risk Minimization (ERM) with Group Distributionally Robust Optimization (Group DRO), and additionally investigates identity-targeted Group DRO models for Christian, Muslim, LGBTQ, and Black identity groups.

## Project Overview

The main experimental pipeline is:

1. Explore the CivilComments-WILDS training and validation data.
2. Tokenize comments using `distilbert-base-uncased`.
3. Train a DistilBERT ERM baseline.
4. Train a general Group DRO model using `identity_any x toxicity label` training groups.
5. Train separate identity-targeted Group DRO models for Christian, Muslim, LGBTQ, and Black groups.
6. Compare average and worst-group performance.
7. Analyze toxic vs. non-toxic identity-subset performance and model errors.


## Source Code

The reusable implementation is under `src/`:

- `data.py` - CivilComments-WILDS data loading utilities.
- `groups.py` - identity definitions, Group DRO group construction, and group-level evaluation utilities.
- `metrics.py` - shared classification metrics.
- `preprocessing.py` - DistilBERT tokenization and processed dataset generation.
- `train_erm.py` - ERM training and test evaluation.
- `train_group_dro.py` - refactored Group DRO training and evaluation.
- `groupdro/` -  Group DRO implementation and related code retained for comparison/reproducibility.


The  `notebooks/` provide the main experimental and analysis workflow:

- EDA notebook -  analyzes the training 
- results-analysis notebook - compares ERM and Group DRO using saved test predictions and metrics
- error-analysis notebook  - compares ERM and Group DRO predictions
- group-specific evaluation notebook  - evaluate saved model predictions for identities such as Christian, Muslim, LGBTQ, and Black without retraining the models.
- Training notebooks  - ERM and GroupDRO



## Environment Setup


Install the required dependencies from the project root:

```bash
pip install -r requirements.txt
```


## Data and Preprocessing

The project uses CivilComments-WILDS with the original train, validation, and test splits. Run preprocessing from the project root:

```bash
python -m src.preprocessing
```

## Reproducing ERM

Run the ERM training pipeline from the project root:

```bash
python -m src.train_erm
```

Saved `test_predictions.csv` files can be used for later group and error analyses without retraining the model.

## Reproducing General Group DRO

Run the refactored general Group DRO implementation with:

```bash
python -m src.train_group_dro
```
Saved `test_predictions.csv` files can be used for later group and error analyses without retraining the model.


## Identity-Targeted Group DRO Experiments

In addition to the general Group DRO experiment, separate Group DRO models were trained for four identity groups:

- Christian
- Muslim
- LGBTQ
- Black

The targeted experiments are compared with ERM separately for toxic and non-toxic comments belonging to the corresponding identity group.

## Large Model Files

Some trained model checkpoints and full training-result folders are not committed to GitHub because of file-size limitations. They are provided through external Drive folders.

### Trained models and results

**Trained models:** [Google Drive](https://drive.google.com/drive/folders/1jAU3P89XZax9BPU409D5FXi59VLXb-9A)

These experiments were trained using the available GPU environment at UiS.

**Results:** [Google Drive](https://drive.google.com/drive/folders/1A7tpO_CvgODWQ4y6QIraIYqytEy-CZh2?usp=sharing)

These experiments were trained using the available GPU environment in Googel Colab.

After downloading either Drive folder, place the relevant model/result folders under the corresponding `results/` location expected by the notebooks or update the notebook path variables to the downloaded location.

