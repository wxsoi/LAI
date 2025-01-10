from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
import torch
import pandas as pd
import os.path
from os import path
import logging
from datetime import datetime
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

def encode(dataset, dataset_bias):
    """
    Tokenizes text in df with truncation
    :param dataset: dataframe with a 'processed_post' and/or and 'debiased_post' column
    :param dataset_bias: true/false whether to use 'debiased_post' or 'processed_post' column
    :return: tokenized dataset
    """
    if dataset_bias:
        outputs = tokenizer(
            dataset['debiased_post'], truncation=True, padding='max_length',
                max_length=512)
    else:
        outputs = tokenizer(
            dataset['processed_post'], truncation=True, padding='max_length',
            max_length=512)
    return outputs

if __name__ == '__main__':
    # Constants (Change)
    debiased_dataset = True     # to determine which column of text the model will use; True for debiased_post
    test_path = "./data/test_debiased_1000.csv"     # path to test dataset

    # Final saved model can be found under the 'model/BERT-mini-{time}' directory
    model_path = "./logging/a_FINAL_FULL_mini_12/final_data_bert-mini_2025-01-08_03-57-48"

    # Seeds
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------------------------------------------------------------------
    #                   DATA SETUP
    # ----------------------------------------------------------------------------------------------------

    # Import test data from csv
    test_df = pd.read_csv(test_path)

    # Convert dfs to suitable Dataset format for transformers
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer and trained model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

    # Check for and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Apply the tokenization function to test dataset and rename the label columns for expected format
    test_dataset = test_dataset.map(lambda x: encode(x, debiased_dataset), batched=True)
    test_dataset = test_dataset.rename_column('label', 'labels')

    # ----------------------------------------------------------------------------------------------------
    #                   TRAINED MODEL EVALUATION METRICS
    # ----------------------------------------------------------------------------------------------------

    # Configure logging
    if not path.exists('logging'):
        os.mkdir('logging')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename=f"./logging/optuna_logs_{current_time}.log", level=logging.INFO, format="%(message)s")

    logging.info("CALCULATING METRICS FOR FINAL TRAINED MODEL")
    logging.info("-----------------------------------\n")

    # Initialize the Trainer
    training_args = TrainingArguments(output_dir='./results', per_device_eval_batch_size=16)
    trainer = Trainer(model=model, args=training_args)

    # Model label predictions on unseen test dataset
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)

    # Get actual labels from unseen test dataset
    actual_labels = [example['labels'] for example in test_dataset]

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')
    class_report = classification_report(actual_labels, predicted_labels, target_names=["0", "1", "2"])     # Summary of above

    # Confusion matrix
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1", "Actual 2"],
                              columns=["Pred 0", "Pred 1", "Pred 2"])

    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info("\nClassification Report:\n")
    logging.info(class_report)
    logging.info("\nConfusion Matrix:\n")
    logging.info(conf_matrix_df)
    logging.info("")

    # Logging metrics
    logging.info("FINISHED EVALUATION")
    logging.info("-----------------------------------\n")
