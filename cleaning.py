import re
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from nltk.corpus import words, stopwords
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from rapidfuzz import fuzz, process

# Load English words and stopwords into sets for fast lookups
english_words = set(words.words())
stop_words = set(stopwords.words('english'))
word_pattern = re.compile(r'\b\w+\b')  # Matches words

def is_english_word(word):
    """
    Determines if a word is in English or a close typo. Retains if so, removes otherwise.

    Args:
        word (str): The word to check.
        typo_threshold (int): Minimum similarity (0-100) to retain a typo.

    Returns:
        bool: True if the word is English or close to English; False otherwise.
    """
    if word in english_words:
        return word  # Word is valid English

    # Check for close matches using rapidfuzz
    best_match = process.extractOne(word, english_words, scorer=fuzz.ratio)
    if best_match[1] >= 85:
        return best_match[0]

    return ""


def remove_reddit_formatting(text):
    # Define a series of regex patterns and replacements to remove formatting
    patterns = [
        (r'\*\*([^*]+)\*\*', r'\1'),  # Bold
        (r'__([^_]+)__', r'\1'),          # Bold
        (r'\*([^*]+)\*', r'\1'),        # Italic
        (r'_([^_]+)_', r'\1'),            # Italic
        (r'\*\*\*([^*]+)\*\*\*', r'\1'),  # Bold-italic
        (r'___([^_]+)___', r'\1'),        # Bold-italic
        (r'~~([^~]+)~~', r'\1'),          # Strikethrough
        (r'>!([^!]+)!<', r'\1'),          # Spoilers
        (r'\^\(([^)]+)\)', r'\1'),     # Superscript (parentheses style)
        (r'\^([^ ]+)', r'\1'),           # Superscript (standalone)
        (r'`([^`]+)`', r'\1'),            # Code
        (r'r/([^ ]+)', r''),            # r/subreddit
        (r'u/([^ ]+)', r''),            # u/user
        (r'\[([^\]]+)\]\([^\)]+\)', r'\1'),  # Markdown links
        (r'\[([^\]]+)\]\[[^\]]+\]', r'\1'),  # Reference links
        (r'\[(\d+)\]: [^\s]+', r''),   # Reference link definitions
        (r'^(#+)\s*(.+)', r'\2'),        # Headings
        (r'^\s*[-*]\s+', r''),           # Unordered list items
        (r'\d+\.\s+', r''),             # Ordered list items
        (r'<([^_]+)>', r''),  # web link <>
        (r'> ([^ ]+)', r'\1'),  # > quotes
        (r'[^\x00-\x7F]+', r''), # nonstandard symbols
        (r'&gt', r''),  # remove &gt idk what this is but its all over
        (r'&lt', r''),  # &lt idk
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text


def clean_and_correct_text(text):
    """
    Cleans and corrects text by removing Reddit formatting, non-English words, typos,
    and stopwords.
    """
    text = remove_reddit_formatting(text)  # First, clean Reddit-specific formatting
    tokens = word_pattern.findall(text)  # Tokenize the cleaned text
    corrected_tokens = [is_english_word(word) for word in tokens if word]  # Correct words
    filtered_tokens = [word for word in corrected_tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


def parallelize_dataframe(df, func, num_partitions=None):
    """
    Splits a DataFrame into partitions and applies a function in parallel.

    Args:
        df (DataFrame): The DataFrame to process.
        func (function): The function to apply to each partition.
        num_partitions (int): Number of partitions (defaults to CPU count).

    Returns:
        DataFrame: The processed DataFrame.
    """
    num_partitions = num_partitions or cpu_count()
    df_split = np.array_split(df, num_partitions)
    with Pool(num_partitions) as pool:
        with tqdm(total=num_partitions, desc="Processing Partitions") as pbar:
            results = []
            for result in pool.imap_unordered(func, df_split):
                results.append(result)
                pbar.update(1)
        df = pd.concat(pool.map(func, df_split))
    return df

def process_partition(df_partition):
    """
    Processes a partition of the DataFrame.
    """
    df_partition["processed_post"] = df_partition["post"].apply(
        lambda x: clean_and_correct_text(x) if isinstance(x, str) else x
    )
    return df_partition


if __name__ == '__main__':
    df = pd.read_csv("./data/political_leaning.csv")
    df.rename(columns={'auhtor_ID': 'author_ID'}, inplace=True)
    df = df[df["author_ID"] == 't2_431z3f5']
    #df = df.head(100)
    # Apply parallel processing
    df = parallelize_dataframe(df, process_partition)

    # Continue with the rest of the pipeline
    df['nr_of_words'] = df['processed_post'].apply(lambda x: len(x.split()))
    df['nr_of_characters'] = df['processed_post'].apply(len)

    # Label Encoding
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['political_leaning'])
    df = df.drop(columns=['political_leaning'], axis=1)
    #df = df.drop(columns=['post', 'political_leaning'], axis=1)
    df.to_csv('./data/processed.csv', index=False)