import pandas as pd
from transformers import pipeline
df = pd.read_csv("./data/political_leaning.csv")
df.rename(columns={'auhtor_ID': 'author_ID'}, inplace=True)

model = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)
all_texts = df["post"].values.tolist()
all_langs = model(all_texts, truncation=True, batch_size=10)
df["language"] = [d["label"] for d in all_langs]

# Display the result
print(df)
df.to_csv('./data/out.csv')

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
