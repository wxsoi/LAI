import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import shap

def hypertune_model():
    """
    Trains and hyperparameter-tunes an XGBoost classifier using GridSearchCV.

    Steps:
    - Loads train, test, and validation datasets.
    - Vectorizes text using TF-IDF with max_features set to 1000.
    - Applies GridSearchCV to tune hyperparameters for the XGBoost classifier.
    - Trains the final model using the best hyperparameters on training data.

    Returns:
        model (xgb.Booster): The trained XGBoost model.
        tfidf (TfidfVectorizer): The TF-IDF vectorizer fitted on the training data.
        dtrain (xgb.DMatrix): The training dataset in XGBoost's DMatrix format.
    """
    # Load the datasets
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    val = pd.read_csv('val.csv')

    # Split datasets into X and y
    X_train = train['processed_post']
    y_train = train['label']

    X_test = test['processed_post']
    y_test = test['label']

    X_val = val['processed_post']
    y_val = val['label']

    # Vectorize text
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    X_val_tfidf = tfidf.transform(X_val).toarray()

    # Hypertuning
    # Initialize the model
    xgb_clf = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        learning_rate = 0.01,
        eta= 0.3,
        eval_metric = 'mlogloss',
        seed=42
    )

    # Define the parameter grid
    param_grid = {
        'max_depth': [4, 6, 8],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    # Perform the random search
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train_tfidf, y_train)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)

    # Convert data to DMatrix
    dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
    dtest = xgb.DMatrix(X_test_tfidf, label=y_test)
    dval = xgb.DMatrix(X_val_tfidf, label=y_val)

    evals_result = {}

    model = xgb.train(best_params, dtrain, num_boost_round=500, evals=[(dval, 'test')], early_stopping_rounds=50, evals_result=evals_result)

    # Evaluation metrics
    # Predict on the test set
    y_test_pred = model.predict(dtest)
    y_test_pred = y_test_pred.astype(int)

    # Compute accuracy
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

    # Compute F1 Score
    print(f'Classification report: {classification_report(y_test, y_test_pred)}')

    return model, tfidf, dtrain


def debiasing_dataset(model, tfidf, dtrain):
    """
    Identifies and removes biased words from the dataset based on SHAP values.

    Steps:
    - Uses SHAP values to calculate the importance of each feature (word) for each label.
    - Computes relative difference to identify biased words for specific labels.
    - Removes biased words from train, validation, and test datasets.
    - Saves debiased datasets for multiple relative difference thresholds.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        tfidf (TfidfVectorizer): TF-IDF vectorizer fitted on the training data.
        dtrain (xgb.DMatrix): The training dataset in XGBoost's DMatrix format.
    """
    tfidf_feature_names = tfidf.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    explanation = explainer(dtrain)

    shap_values = explanation.values

    feature_names = list(tfidf_feature_names)
    
    # Aggregate shap scores for each label
    n_labels = shap_values.shape[2]
    shap_aggregated = {}
    for label_idx in range(n_labels):
        shap_aggregated[label_idx] = np.mean(np.abs(shap_values[:, :, label_idx]), axis=0)

    n_labels = len(shap_aggregated)  # Number of labels
    n_features = len(feature_names)  # Number of features (words)

    # Dictionaries to store biased words for each label
    biased_words = {label: [] for label in range(n_labels)}
    max_other_zero_words = {label: [] for label in range(n_labels)}

    # Define a relative difference threshold (lowest, 1)
    relative_threshold = 1

    for word_idx, word in enumerate(feature_names):
        # Get scores for the current word for all labels
        scores = {label: shap_aggregated[label][word_idx] for label in range(n_labels)}

        for label in scores:
            # Get scores for other labels
            other_labels = [l for l in scores if l != label]
            max_other_score = max([scores[l] for l in other_labels], default=0)

            # Compute the relative difference
            relative_diff = (scores[label] - max_other_score) / max_other_score

            # Check if it exceeds the threshold
            if relative_diff > relative_threshold:
                biased_words[label].append((word, scores[label], max_other_score, relative_diff))

    # Define thresholds and corresponding filenames
    thresholds = [10, 5, 1]
    filenames = ['train.csv', 'test.csv', 'val.csv']

    # Define a function to remove biased words from a single post
    def remove_biased_words(text, biased_words):
        words = text.split()  # Tokenize text into words
        filtered_words = [word for word in words if word not in biased_words]
        return ' '.join(filtered_words)  # Recombine filtered words into a sentence

    # Loop over each threshold
    for cutoff_threshold in thresholds:
        # Combine biased words based on the relative difference threshold
        biased_word_set = set()
        for label_words in biased_words.values():
            for word, _, _, relative_diff in label_words:
                if relative_diff > cutoff_threshold:
                    biased_word_set.add(word)

        for filename in filenames:
            df = pd.read_csv(filename)
            # Apply the function to the 'processed_post' column
            df['debiased_post'] = df['processed_post'].apply(
                lambda x: remove_biased_words(x, biased_word_set)
            )

            df['debiased_post'] = df['debiased_post'] .fillna('')

            name = f'{filename.split('.')[0]}_debiased_{cutoff_threshold}00.csv'
            # Save the cleaned dataset to a file
            df.to_csv(name, index=False)

            print(f"Biased words with relative difference > {cutoff_threshold}00% have been removed. Saved to '{filename.split('.')[0]}_debiased_{cutoff_threshold}00'.csv.")


def evaluate_baseline_debiased(model, tfidf):
    """
    Evaluates the model's performance on debiased test datasets.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        tfidf (TfidfVectorizer): TF-IDF vectorizer fitted on the training data.
    """
    thresholds = [10, 5, 1]
    for cutoff_threshold in thresholds:
        print(f'Threshold: {cutoff_threshold}00')
        test = pd.read_csv(f'test_debiased_{cutoff_threshold}00.csv')
        X_test = test['debiased_post'].fillna('')
        y_test = test['label']
        X_test_tfidf = tfidf.transform(X_test).toarray()
        dtest = xgb.DMatrix(X_test_tfidf, label=y_test)
        # Predict on the test set
        y_test_pred = model.predict(dtest)
        y_test_pred = y_test_pred.astype(int)

        # Compute accuracy
        accuracy_test = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

        # Compute F1 Score
        print(f'Classification report: {classification_report(y_test, y_test_pred)}')
        print('--------------------------')


def run_experiments():
    """
    Conducts experiments to evaluate the performance of debiased datasets at different thresholds.

    This function:
    - Iterates through predefined thresholds to process debiased datasets.
    - Loads debiased training, validation, and testing datasets corresponding to each threshold.
    - Splits the datasets into features (X) and labels (y).
    - Applies TF-IDF vectorization to text data.
    - Initializes and tunes an XGBoost classifier using GridSearchCV to find optimal hyperparameters.
    - Trains the model with the best parameters on the training dataset.
    - Evaluates the model's performance on the test dataset, reporting accuracy and F1 scores.

    Results for each threshold are printed to the console, including best parameters, test accuracy,
    and a classification report.

    Returns:
        None
    """
    thresholds = [10, 5, 1]
    
    for cutoff_threshold in thresholds:
        print(f'Running for threshold {cutoff_threshold}')
        train = pd.read_csv(f'train_debiased_{cutoff_threshold}00.csv')
        test = pd.read_csv(f'test_debiased_{cutoff_threshold}00.csv')
        val = pd.read_csv(f'val_debiased_{cutoff_threshold}00.csv')

        # Split datasets into X and y
        X_train = train['debiased_post'].fillna('')
        y_train = train['label']

        X_test = test['debiased_post'].fillna('')
        y_test = test['label']

        X_val = val['debiased_post'].fillna('')
        y_val = val['label']

        # Vectorize text
        tfidf = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf.fit_transform(X_train).toarray()
        X_test_tfidf = tfidf.transform(X_test).toarray()
        X_val_tfidf = tfidf.transform(X_val).toarray()

        # Convert data to DMatrix
        dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
        dtest = xgb.DMatrix(X_test_tfidf, label=y_test)
        dval = xgb.DMatrix(X_val_tfidf, label=y_val)

        # Initialize the model
        xgb_clf = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eta= 0.3,
            eval_metric = 'mlogloss',
            learning_rate = 0.01,
            seed=42
        )

        # Define the parameter distribution
        param_dist = {
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 1, 5]
        }

        # Perform the random search
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_dist,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        grid_search.fit(X_train_tfidf, y_train)

        # Best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print("Best Parameters:", best_params)

        evals_result = {}

        model = xgb.train(best_params, dtrain, num_boost_round=500, evals=[(dval, 'test')], early_stopping_rounds=50, evals_result=evals_result)

        # Predict on test  set
        y_test_pred = model.predict(dtest)

        # Convert predictions to integer type (as XGBoost predictions are floats)
        y_test_pred = y_test_pred.astype(int)

        # Compute accuracy
        accuracy_test = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {accuracy_test * 100:.2f}%  for {cutoff_threshold}")

        # Compute F1 Score
        print(f'Classification report: {classification_report(y_test, y_test_pred)}')

        print('---------------------------------------------------')



model, tfidf, dtrain = hypertune_model()
debiasing_dataset(model, tfidf, dtrain)
evaluate_baseline_debiased(model, tfidf)
run_experiments()