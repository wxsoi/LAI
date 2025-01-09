from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import Dataset
import optuna
import torch
import pandas as pd
import os.path
from os import path
import logging
from datetime import datetime
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

# Callback function to log the evaluation losses for each training epoch
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Step: {state.global_step}, Loss: {state.log_history[-1].get('loss')}")
        for key, value in state.log_history[-1].items():
            logging.info(f"{key}: {value}")

        # Add a whitespace line after the complete logging event
        logging.info("\n")

# Create function for printing
def print_custom(text):
    print('\n')
    print(text)
    print('-'*100)

# Create callback function to log Optuna trial information
def print_trial_info(study, trial):
    logging.info(f"Trial {trial.number}:")
    logging.info(f"  Value = {trial.value}")
    logging.info(f"  Hyperparameters: {trial.params}")
    logging.info(f"  Best Trial so far: {study.best_trial.number}")
    logging.info(f"  Best Value so far: {study.best_trial.value}")
    logging.info(f"  Time Taken: {trial.duration}")

    logging.info("\n")


def objective(trial: optuna.Trial):
    """
    Trains a transformer model with hyperparameters chosen by Optuna, with the objective to minimize evaluation loss.
    :param trial: current trial number
    :return: evaluation loss
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=trial.suggest_loguniform('learning_rate', low=1e-6, high=1e-4),
        weight_decay=trial.suggest_loguniform('weight_decay', 1e-3, 0.15),
        num_train_epochs=trial.suggest_int('num_train_epochs', low=2, high=10),
        per_device_train_batch_size=trial.suggest_int('per_device_train_batch_size', low=8, high=64),
        per_device_eval_batch_size=32,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback()]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()

    return eval_result['eval_loss']     # eval loss


# Tokenize text in df with truncation
def encode(dataset):
    outputs = tokenizer(
        dataset['processed_post'], truncation=True, padding='max_length',
            max_length=512)
    return outputs

if __name__ == '__main__':
    # CONSTANTS
    train_path = "./data/train_debiased_1000.csv"
    val_path = "./data/val_debiased_1000.csv"
    test_path = "./data/test_debiased_1000.csv"
    model_variant = "bert-small"     # tiny, mini, small, medium
    num_trials = 3

    # Configure logging
    if not path.exists('logging'):
        os.mkdir('logging')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename=f"./logging/optuna_logs_{current_time}.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Convert df's to suitable Dataset format for transformers
    train_dataset = Dataset.from_pandas(train_df)
    evaluation_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer and model
    model_name = f"prajjwal1/{model_variant}"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Check for and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    train_dataset = train_dataset.map(encode, batched=True)
    evaluation_dataset = evaluation_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)

    # Data collator to dynamically pad sequences in each batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    #----------------------------------------------------------------------------------------------------
    #                    CREATE OPTUNA STUDY
    #----------------------------------------------------------------------------------------------------

    logging.info("STARTING HYPERPARAMETER TUNING")
    logging.info("-----------------------------------\n")

    print_custom('Triggering Optuna study')
    study = optuna.create_study(study_name='hp-search-electra', direction='minimize')    # minimize evaluation loss
    study.optimize(func=objective, n_trials=num_trials, callbacks=[print_trial_info])    # callback for logging trials

    logging.info("FINISHED HYPERPARAMETER TUNING")
    logging.info("-----------------------------------\n")

    #----------------------------------------------------------------------------------------------------
    #                    PRINT BEST STUDY HYPERPARAMETERS
    #----------------------------------------------------------------------------------------------------

    print_custom('Finding study best parameters')
    best_lr = float(study.best_params['learning_rate'])
    best_weight_decay = float(study.best_params['weight_decay'])
    best_epoch = int(study.best_params['num_train_epochs'])
    best_batch_size = int(study.best_params['per_device_train_batch_size'])

    print_custom('Extract best study params')
    print(f'The best learning rate is: {best_lr}')
    print(f'The best weight decay is: {best_weight_decay}')
    print(f'The best epoch is : {best_epoch}')
    print(f'The best batch size is : {best_batch_size}')

    logging.info("EXTRACTING BEST PARAMETERS")
    logging.info(f"Best Learning Rate: {best_lr}")
    logging.info(f"Best Weight Decay: {best_weight_decay}")
    logging.info(f"Best Epoch: {best_epoch}")
    logging.info(f"Best Batch Size: {best_batch_size}")
    logging.info("-----------------------------------\n")

    # ----------------------------------------------------------------------------------------------------
    #                   TRAIN BASED ON OPTUNA'S SELECTED HYPERPARAMETERS
    # ----------------------------------------------------------------------------------------------------

    logging.info("TRAINING MODEL WITH BEST PARAMETERS")
    logging.info("-----------------------------------\n")

    print_custom('Training the model on the custom parameters')

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=best_lr,
        weight_decay=best_weight_decay,
        num_train_epochs=best_epoch,
        per_device_train_batch_size=best_batch_size,
        per_device_eval_batch_size=32,
        logging_strategy="steps",
        logging_dir='./logs',
        logging_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback()]
    )

    result = trainer.train()
    trainer.evaluate()

    logging.info("FINISHED TRAINING")
    logging.info("-----------------------------------\n")

    # ----------------------------------------------------------------------------------------------------
    #                   SAVING MODEL
    # ----------------------------------------------------------------------------------------------------

    logging.info("SAVING BEST TUNED MODEL")
    logging.info("-----------------------------------\n")

    print_custom('Saving the best Optuna tuned model')
    if not path.exists('model'):
        os.mkdir('model')

    model_path = "model/{}".format(f"final_data_{model_variant}_{current_time}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # ----------------------------------------------------------------------------------------------------
    #                   TRAINED MODEL METRICS
    # ----------------------------------------------------------------------------------------------------

    # Temporarily changing logging format
    formatter = logging.Formatter("%(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    logging.info("CALCULATING METRICS FOR FINAL TRAINED MODEL")
    logging.info("-----------------------------------\n")

    # Model label predictions on unseen test dataset
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)

    # Get actual labels from unseen test dataset
    actual_labels = [example['label'] for example in test_dataset]

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

    # Logging metrics
    logging.info("FINISHED EVALUATION")
    logging.info("-----------------------------------\n")

    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info("\nClassification Report:\n")
    logging.info(class_report)
    logging.info("\nConfusion Matrix:\n")
    logging.info(conf_matrix_df)
