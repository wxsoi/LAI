import pandas as pd
from tabulate import tabulate

# Load the dataset
df = pd.read_csv("./data/cleaned.csv")
print(tabulate(df.head(), headers='keys', tablefmt='psql'))

def split_text_exact(text, max_words=512):
    words = text.split()    # All individual words in a list

    chunks = []
    # Loop through the words list in steps of max_words
    for i in range(0, len(words), max_words):
        # Take a slice of max_words from the words list
        chunk = words[i:i + max_words]
        # Join the words in the chunk into a single string
        chunk_text = ' '.join(chunk)
        # Add the chunk to the chunks list
        chunks.append(chunk_text)
    return chunks

# Apply the split_text_exact function to the processed_post column
df['processed_post_split'] = df['processed_post'].apply(split_text_exact)

# Explode the list of chunks into separate rows
df_split = df.explode('processed_post_split').reset_index(drop=True)

# Calculate the number of words in each chunk and update the 'nr_of_words' column
df_split['nr_of_words'] = df_split['processed_post_split'].apply(lambda x: len(x.split()))

# Remove rows with less than 30 words
filtered_df = df_split[df_split['nr_of_words'] >= 100]

filtered_df = filtered_df.drop('processed_post', axis=1)
filtered_df = filtered_df.rename(columns={'processed_post_split': 'processed_post'})

# Resulting df
print(tabulate(filtered_df.head(), headers='keys', tablefmt='psql'))

filtered_df.to_csv('data/cleaned_split_512v2.csv', encoding='utf-8', index=False, header=True)
