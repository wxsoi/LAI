# LAI

Data Preprocessing
1. Run language-detect.py to remove the majority non-English rows
2. Run datapreprocessing.py to sample 33% for each political label
3. Run cleaning.py to preprocess the data
4. Run 512traintestval.py to tokenize them into 512 tokens and split for training, testing, and validation sets

Data Debiasing
1. ...
2. ...

Logistic Regression
- For Logistic Regression, run logisticregression_hypertuning.ipynb for all test sets.

XGBoost
- ...

BERT
1. Run BERT_transformer.py to train models
Change train_path, val_path and test_path based on the path to the data. Make sure the train, val and test datasets are from the same type (original or either of the debiased datasets) 
Change debiased_dataset based on whether you are running the original dataset or one of the debiased datasets. If original dataset, set debiased_dataset = True. False otherwise.
2. Repeat step 1 with the original dataset and all debiased datasets.
3. Run BERT_transformer_eval.py to test the model trained with the original dataset on the debiased test datasets.
Change test_path based on the path to the data.
Change model_path to the path of the saved original model. It should be saved under model/BERT-mini-{time}. If you trained the original model in step 1 first, then it should be the first in the 'model' directory.
Debiased_dataset should remain True for testing on the debiased dataset