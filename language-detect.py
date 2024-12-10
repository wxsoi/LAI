import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load dataset
df = pd.read_csv("./data/political_leaning.csv")
df.rename(columns={'auhtor_ID': 'author_ID'}, inplace=True)

# Initialize model
model = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=0 #gpu
)

all_texts = df["post"].values.tolist()  # post column to text list
all_langs = []

# Use tqdm to track progress
for text in tqdm(all_texts, desc="Detecting languages"):
    result = model(text, truncation=True)
    all_langs.append(result[0]["label"])

df["language"] = all_langs  # add result to dataframe
print(df)
df.to_csv('./data/out.csv', index=False)    # save result

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# # Load the model and tokenizer
# model_name = "papluca/xlm-roberta-base-language-detection"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
#
# # Define a function to detect the language of a text
# def detect_language(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#     predicted_label = torch.argmax(probs, dim=1).item()
#     labels = model.config.id2label  # Get the label mapping
#     return labels[predicted_label]
#

#
# # Apply the language detection function to the DataFrame
# df["detected_language"] = df["post"].apply(detect_language)
