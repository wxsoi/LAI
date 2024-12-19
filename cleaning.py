import re
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from gensim.parsing.preprocessing import remove_stopwords
import contractions
from textblob import Word
import stanza

# Initialize the Stanza pipeline
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

# Define a series of regex patterns and replacements to remove formatting
patterns = [
    (re.compile(r'\*\*([^*]+)\*\*'), r'\1'),  # Bold
    (re.compile(r'__([^_]+)__'), r'\1'),  # Bold
    (re.compile(r'\*([^*]+)\*'), r'\1'),  # Italic
    (re.compile(r'_([^_]+)_'), r'\1'),  # Italic
    (re.compile(r'\*\*\*([^*]+)\*\*\*'), r'\1'),  # Bold-italic
    (re.compile(r'___([^_]+)___'), r'\1'),  # Bold-italic
    (re.compile(r'~~([^~]+)~~'), r'\1'),  # Strikethrough
    (re.compile(r'>!([^!]+)!<'), r'\1'),  # Spoilers
    (re.compile(r'\^\(([^)]+)\)'), r'\1'),  # Superscript (parentheses style)
    (re.compile(r'\^([^ ]+)'), r'\1'),  # Superscript (standalone)
    (re.compile(r'`([^`]+)`'), r'\1'),  # Code
    (re.compile(r'r/([^ ]+)'), r''),  # r/subreddit
    (re.compile(r'u/([^ ]+)'), r''),  # u/user
    (re.compile(r'\[([^\]]+)\]\([^\)]+\)'), r'\1'),  # Markdown links
    (re.compile(r'\[([^\]]+)\]\[[^\]]+\]'), r'\1'),  # Reference links
    (re.compile(r'\[(\d+)\]: [^\s]+'), r''),  # Reference link definitions
    (re.compile(r'^(#+)\s*(.+)'), r'\2'),  # Headings
    (re.compile(r'^\s*[-*]\s+'), r''),  # Unordered list items
    (re.compile(r'\d+\.\s+'), r''),  # Ordered list items
    (re.compile(r'<([^_]+)>'), r''),  # web link <>
    (re.compile(r'> ([^ ]+)'), r'\1'),  # > quotes
    (re.compile(r'[^\x00-\x7F]+'), r''),  # nonstandard symbols
    (re.compile(r'&gt'), r''),  # remove &gt
    (re.compile(r'&lt'), r''),  # &lt
    (re.compile(r'&nbsp'), r''),  # &nbsp
    (re.compile(r'&amp'), r''),  # &amp
    (re.compile(r'(.)\1+'), r'\1\1'),  # removing >2 consecutive letters in a word (english usually has only 2)
    (re.compile(r'\b(?!(?<!\w)(\d{2}|\d{4})(?!\w))\d+\b'), r''),  # removing non 2 or 4 digit numbers but keep numbers connected
    # to words such as ww2
]


def is_english_word(tokens):
    """
    Determines if a word is in English or a close typo. Retains if so, removes otherwise.

    Args:
        tokens (list): The list of tokens to check

    Returns:
        correct_tokens (list): The list of tokens with >=85% probability being English
    """
    correct_tokens = []
    for token in tokens:
        suggestion = Word(token).spellcheck()[0]
        if suggestion[1] >= 0.85: # 85% threshold
            correct_tokens.append(suggestion[0])
    return correct_tokens


def remove_reddit_formatting(text):
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


def clean_and_correct_text(text):
    """
    Cleans and corrects text by removing Reddit formatting, non-English words, typos,
    and stopwords.
    """
    text = remove_reddit_formatting(text)  # First, clean Reddit-specific formatting
    text = text.lower()
    text = contractions.fix(text) # expand contractions
    text = remove_stopwords(text) # Use gensim's remove_stopwords
    text = re.sub(r'[^\w\s]+', '', text) # remove symbols
    # tokens = nltk.word_tokenize(text) # tokenize
    # correct_tokens = is_english_word(tokens) # check if its english and autocorrect
    # corrected_words = ' '.join(tokens) # Join tokens back into a string
    doc = nlp(text) # lemmatization
    text = ' '.join([word.lemma for sentence in doc.sentences for word in sentence.words])
    return text

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
    start_time = time.time()  # Start timer
    df = pd.read_csv("./data/political_leaning.csv")
    df.rename(columns={'auhtor_ID': 'author_ID'}, inplace=True)
    df = df.head(100)

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

    df.to_csv('./data/first100test.csv', index=False)

    end_time = time.time()  # End timer
    print(f"Total execution time: {end_time - start_time:.2f} seconds")