import re
import pandas as pd
from nltk.corpus import words, stopwords
from difflib import get_close_matches
import nltk
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK corpora (do this only once in your environment)
nltk.download('words')
nltk.download('stopwords')

# Load English words and stopwords into sets for fast lookups
english_words = set(words.words())
stop_words = set(stopwords.words('english'))

# Precompile regex patterns
word_pattern = re.compile(r'\b\w+\b')  # Matches words
bot_command_pattern = re.compile(r'!\w+(\s+\d+\s+\w+)?')
footnote_pattern = re.compile(r'\^\^\S+|\^\^\[.*?\]')
markdown_link_pattern = re.compile(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)')
subreddit_pattern = re.compile(r'\/?r\/[A-Za-z0-9_]+')
user_pattern = re.compile(r'\/?u\/[A-Za-z0-9_-]+')
inline_code_pattern = re.compile(r'`{1,3}.*?`{1,3}', flags=re.DOTALL)
block_quote_pattern = re.compile(r'^>.*$', flags=re.MULTILINE)
markdown_format_pattern = re.compile(r'(\*{1,3})(.*?)\1')
strikethrough_pattern = re.compile(r'~~(.*?)~~')
spoiler_pattern = re.compile(r'>!.*?!<')
raw_link_pattern = re.compile(r'https?:\/\/\S+')
list_item_pattern = re.compile(r'^\s*[-\*]\s+', flags=re.MULTILINE)
numbered_list_pattern = re.compile(r'^\s*\d+\.\s+', flags=re.MULTILINE)


def correct_english_word(word, threshold=0.8):
    """
    Corrects a word if it is a typo or invalid English word.

    Args:
        word (str): The word to correct.
        threshold (float): Similarity threshold for corrections.

    Returns:
        str: Corrected word if valid, else an empty string.
    """
    word = word.lower()
    if word in english_words:  # Exact match
        return word
    close_matches = get_close_matches(word, english_words, n=1, cutoff=threshold)
    return close_matches[0] if close_matches else ""  # Return closest match or empty


def clean_reddit_formatting(text):
    """
    Removes Reddit-specific formatting and normalizes the text.
    """
    text = bot_command_pattern.sub('', text)
    text = footnote_pattern.sub('', text)
    text = markdown_link_pattern.sub(r'\1', text)
    text = subreddit_pattern.sub('', text)
    text = user_pattern.sub('', text)
    text = inline_code_pattern.sub('', text)
    text = block_quote_pattern.sub('', text)
    text = markdown_format_pattern.sub(r'\2', text)
    text = strikethrough_pattern.sub(r'\1', text)
    text = spoiler_pattern.sub('', text)
    text = raw_link_pattern.sub('', text)
    text = list_item_pattern.sub('', text)
    text = numbered_list_pattern.sub('', text)
    text = re.sub(r'\n+', '\n', text).strip()  # Remove extra newlines and trim
    return text


def clean_and_correct_text(text):
    """
    Cleans and corrects text by removing Reddit formatting, non-English words, typos,
    and stopwords.
    """
    text = clean_reddit_formatting(text)  # First, clean Reddit-specific formatting
    tokens = word_pattern.findall(text)  # Tokenize the cleaned text
    corrected_tokens = [correct_english_word(word) for word in tokens]
    filtered_tokens = [word.lower() for word in corrected_tokens if word and word not in stop_words]
    return ' '.join(filtered_tokens)


df = pd.read_csv("./data/political_leaning.csv")
df.rename(columns={'auhtor_ID': 'author_ID'}, inplace=True)
df["processed_post"] = df["post"].apply(lambda x: clean_and_correct_text(x) if isinstance(x, str) else x)

# Calculate  nr of words and nr of characters for each post
df['nr_of_words'] = df['clean_post'].apply(lambda x: len(x.split()))
df['nr_of_characters'] = df['clean_post'].apply(len)

# Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['political_leaning'])  # 0, 1, 2 for center, left, right respectively

# Drop not needed columns
df = df.drop(columns=['post', 'political_leaning'], axis=1)

df.to_csv('./data/processed.csv', index=False)