
# Run once
import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
# import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')
    df = pd.concat([df_train, df_test])

    label_column = 'label'

    X_train = df_train.drop(columns=[label_column])
    y_train = df_train[label_column]

    X_test = df_test.drop(columns=[label_column])
    y_test = df_test[label_column]

    # Step 1: Vectorize the text
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train['processed_post']).toarray()
    X_test_tfidf = tfidf.transform(X_test['processed_post']).toarray()

    # Step 2: Combine TF-IDF and numeric features
    X_train_combined = np.hstack((X_train_tfidf, X_train[['nr_of_words', 'nr_of_characters']].values))
    X_test_combined = np.hstack((X_test_tfidf, X_test[['nr_of_words', 'nr_of_characters']].values))

    # Get feature names: Combine TF-IDF feature names and numeric feature names
    tfidf_feature_names = tfidf.get_feature_names_out()
    numeric_feature_names = ['nr_of_words', 'nr_of_characters']
    all_feature_names = list(tfidf_feature_names) + numeric_feature_names

    # Step 4: Train the model
    model = SVC()
    model.fit(X_train_combined, y_train)

    # Step 5: Evaluate the model
    accuracy = model.score(X_test_combined, y_test)
    print(f"Accuracy: {accuracy}")

    #
    # # Step 6: Compute SHAP values
    # explainer = shap.LinearExplainer(model, X_test_combined)
    # shap_values = explainer.shap_values(X_test_combined)
    #
    # for class_idx, class_name in enumerate(model.classes_):
    #     shap.summary_plot(shap_values[class_idx], X_test_combined, feature_names=all_feature_names,
    #                       title=f"SHAP Summary for Class: {class_name}")
    #
    #
    #
    # # In[ ]:
    #
    #
    # # Aggregate SHAP values by feature
    # for class_idx, class_name in enumerate(model.classes_):
    #     print(f"\nTop features for class: {class_name}")
    #     shap_values_class = shap_values[..., class_idx]
    #     mean_importance = np.abs(shap_values_class.values).mean(axis=0)
    #     important_features = sorted(zip(tfidf_feature_names, mean_importance), key=lambda x: x[1], reverse=True)[:10]
    #     for feature, importance in important_features:
    #         print(f"{feature}: {importance}")
    #
    #
    # # In[ ]:


    # Make predictions on the test set
    y_pred = model.predict(X_test_combined)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Visualize the Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Separate TF-IDF by label
    tfidf_labelwise = {}
    for label in [0, 1, 2]:  # Replace with your unique labels
        subset = df[df['label'] == label]['processed_post']
        tfidf_labelwise[label] = TfidfVectorizer(max_features=1000)
        tfidf_labelwise[label].fit(subset)

    global_tfidf = TfidfVectorizer(max_features=1000)
    global_tfidf.fit(df['processed_post'])

    # Compare specific terms
    term = "government"
    global_score = global_tfidf.idf_[global_tfidf.vocabulary_.get(term, -1)]
    print(f"Global score for {term}: {global_score}")

    for label in [0, 1, 2]:
        label_score = tfidf_labelwise[label].idf_[tfidf_labelwise[label].vocabulary_.get(term, -1)]
        print(f"Label-specific score for {term} (label {label}): {label_score}")

    # Compare specific terms
    term = "government"
    global_score = global_tfidf.idf_[global_tfidf.vocabulary_.get(term, -1)]
    print(f"Global score for {term}: {global_score}")

    for label in [0, 1, 2]:
        label_score = tfidf_labelwise[label].idf_[tfidf_labelwise[label].vocabulary_.get(term, -1)]
        print(f"Label-specific score for {term} (label {label}): {label_score}")


    # In[ ]:


    # # Access the TF-IDF vectorizer from the pipeline
    # tfidf_vectorizer = tfidf


    # In[ ]:


    # # Transform the entire dataset using the TF-IDF vectorizer
    # X_tfidf = tfidf_vectorizer.transform(df['processed_post'])

    # # Convert the sparse TF-IDF matrix to a DataFrame for easier manipulation
    # tfidf_df = pd.DataFrame(
    #     X_tfidf.toarray(),
    #     columns=tfidf_vectorizer.get_feature_names_out()
    # )

    # # Add the class labels to the DataFrame
    # tfidf_df['label'] = df['label'].values

    # # Compute average TF-IDF scores for each class
    # class_tfidf = tfidf_df.groupby('label').mean()


    # In[ ]:


    # # Rank words for each class
    # ranked_words = {}
    # for label in class_tfidf.index:  # Loop through each class
    #     sorted_words = class_tfidf.loc[label].sort_values(ascending=False)
    #     ranked_words[label] = sorted_words.head(10)  # Top 10 words per class

    # # Print ranked words
    # for label, words in ranked_words.items():
    #     print(f"\nTop words for class '{label}':")
    #     print(words)

