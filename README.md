# Mitigating Data Pollution in Political Bias Prediction with Reddit Data
This repository is used for experiments described in 'Mitigating Data Pollution in Political Bias Prediction with Reddit Data'.

[//]: # (The distinct software license&#40;s&#41; for the code AND the data &#40;if provided&#41;.)
[//]: # (Do we have a data license?)

## Overview
- [tl;dr](#tldr)
- [Reproduction](#reproduction)
  - [Data Preprocessing](#data-preprocessing)
  - [Data Debiasing](#data-debiasing)
  - [Logistic Regression](#logistic-regression)
  - [XGBoost](#xgboost)
  - [BERT](#bert)
- [Dependencies](#dependencies)

## tl;dr
[//]: # (A tl;dr which highlights some points why someone who found your research code should care about this repository.)
This project is focused on the data polution in Reddit posts. It tackles bias and uses pre-processing to remove clutter for the models. The different models (XGBoost, Logistic Regression, BERT, RNN) are hypertuned and evaluated using F1 and accuracy. 

## Reproduction
Follow the instruction in the below subsection. The first two section have to be done first in that order. The section for the models can be done in any order, once the debiased datasets are created.
Unfortunately, we cannot share the Reddit data directly under a data agreement.
> The code was tested with Python 3.12 on Windows 10/11

### Data Preprocessing
1. Run `language-detect.py` to remove the majority non-English rows
2. Run `DataPreprocessing.py` to sample 33% for each political label and remove the 95th percentile of user rows/posts
3. Run `cleaning.py` to preprocess the data
4. Run `512traintestval.py` to tokenize them into 512 tokens and split for training, testing, and validation sets

### Data Debiasing
1. Run `xgboost.py`, specifically `debiasing_dataset()` function to remove the biased words that are detected by XGBoost's SHAP importance.

### Logistic Regression
- For Logistic Regression, run `logisticregression_hypertuning.ipynb` for all test sets.

### XGBoost
- Run `xgboost.py`, specifically `evaluate_baseline_debiased()` and `run_experiments()` functions to run the baseline first and on the debiased testset after. It includes hypertuning and running the model on the best parameters found.

### BERT
1. Run `BERT_transformer.py` to train models
> Change train_path, val_path and test_path based on the path to the data. Make sure the train, val and test datasets are from the same type (original or either of the debiased datasets) 
Change debiased_dataset based on whether you are running the original dataset or one of the debiased datasets. If original dataset, set debiased_dataset = True. False otherwise.
2. Repeat step 1 with the original dataset and all debiased datasets.
3. Run `BERT_transformer_eval.py` to test the model trained with the original dataset on the debiased test datasets.
> Change test_path based on the path to the data.
Change model_path to the path of the saved original model. It should be saved under model/BERT-mini-{time}. If you trained the original model in step 1 first, then it should be the first in the 'model' directory.
Debiased_dataset should remain True for testing on the debiased dataset

## Dependencies
The packages that are used throughout all files
```
nltk==3.9.1
numpy==1.26.4
pandas==2.2.3
torch==2.5.0+cu121
scikit-learn==1.6.0
tqdm==4.66.5
transformers==4.46.3
optuna==4.1.0
logging==0.4.9.6
datasets==3.1.0
seaborn==0.13.2
matplotlib==3.9.2
tabulate==0.9.0
xgboost==2.1.3
shap==0.46.0
contractions==0.1.73
gensim==4.3.3
```
