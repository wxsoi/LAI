import re
from nltk.corpus import words
from nltk.corpus import stopwords
from difflib import get_close_matches
import nltk

nltk.download('words')  # Download the English word corpus
nltk.download('stopwords') # Download  stopwords

# Load English words into a set for quick lookup
english_words = set(words.words())

def clean_reddit_formatting(text):
    """
    Cleans and processes Reddit-specific formatting from the input text.

    Args:
        text (str): The raw Reddit post/comment text.

    Returns:
        str: Cleaned text with formatting removed or normalized.
    """
    # Remove bot commands like "!remindme"
    text = re.sub(r'!\w+(\s+\d+\s+\w+)?', '', text)

    # Remove footnotes (e.g., ^^text or ^^^text)
    text = re.sub(r'\^\^\S+|\^\^\[.*?\]', '', text)

    # Remove Markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1', text)

    # Normalize subreddit mentions (e.g., r/subreddit or /r/subreddit)
    text = re.sub(r'\/?r\/[A-Za-z0-9_]+', '', text)

    # Normalize user mentions (e.g., u/username or /u/username)
    text = re.sub(r'\/?u\/[A-Za-z0-9_-]+', '', text)

    # Remove inline code and code blocks
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)

    # Remove block quotes (e.g., > text)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)

    # Remove Markdown formatting (*italic*, **bold**, ***bold italic***)
    text = re.sub(r'(\*{1,3})(.*?)\1', r'\2', text)

    # Remove strikethrough (~~text~~)
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    # Remove spoilers (>!spoiler!<)
    text = re.sub(r'>!.*?!<', '', text)

    # Remove raw links
    text = re.sub(r'https?:\/\/\S+', '', text)

    # Normalize list items (- Item or numbered lists)
    text = re.sub(r'^\s*[-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove extra newlines and whitespace
    text = re.sub(r'\n+', '\n', text).strip()

    return text


def is_english_with_typos(word, threshold=0.8):
    """
    Determines if a word is English or a close typo of an English word.

    Args:
        word (str): The word to check.
        threshold (float): The similarity threshold for typos (0-1 scale).

    Returns:
        bool: True if the word is English or a typo of an English word.
    """
    word = word.lower()
    if word in english_words:  # Exact match
        return True
    # Check for close matches (1 typo allowed)
    close_matches = get_close_matches(word, english_words, n=1, cutoff=threshold)
    return len(close_matches) > 0


def clean_non_english_and_stopwords(text):
    """
    Cleans the input text by removing non-English words (considering typos) 
    and removing stopwords in one pass.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The cleaned text with non-English words and stopwords removed.
    """
    # Compile the stopwords set
    stop_words = set(stopwords.words('english'))

    # Split text into words and process
    tokens = re.findall(r'\b\w+\b', text)  # Splits into words based on word boundaries
    filtered_tokens = [
        word for word in tokens 
        if is_english_with_typos(word) and word.lower() not in stop_words #remove stopwords, non-english, typos
    ]
    
    # Reconstruct and return the cleaned text
    return ' '.join(filtered_tokens)