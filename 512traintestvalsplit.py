import pandas as pd
from sklearn.model_selection import train_test_split

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

df = pd.read_csv("./data/english-sampled-cleaned.csv")

# Apply the split_text_exact function to the processed_post column
df['processed_post_split'] = df['processed_post'].apply(split_text_exact)

# Explode the list of chunks into separate rows
df = df.explode('processed_post_split').reset_index(drop=True)

# Calculate the number of words in each chunk and update the 'nr_of_words' column
df['nr_of_words'] = df['processed_post_split'].apply(lambda x: len(x.split()))

# Remove rows with less than 30 words
df = df[df['nr_of_words'] >= 100]
df = df.drop('processed_post', axis=1)
df = df.rename(columns={'processed_post_split': 'processed_post'})
print(df["label"].value_counts(normalize=True)*len(df))

# Check if any author is associated with multiple political leanings
multiple_leanings = df.groupby("author_ID")["label"].nunique()

# Filter out users with more than one unique political leaning
conflicting_users = multiple_leanings[multiple_leanings > 1]
if not conflicting_users.empty:
    print("Users with multiple political leanings detected:")
    print(conflicting_users)
else:
    print("No users with multiple political leanings found.")

# Group data by author_ID to ensure unique users in each split
user_groups = df.groupby("author_ID")

# Create a new DataFrame with one representative row per user
unique_users = user_groups.first().reset_index()

# Split the unique users into train, test, and validation sets
train_users, temp_users = train_test_split(
    unique_users,
    test_size=0.3,  # 30% for test + validation
    stratify=unique_users["label"],
    random_state=1
)

test_users, val_users = train_test_split(
    temp_users,
    test_size=(1/3),  # 33% of 30% -> 10% of total
    stratify=temp_users["label"],
    random_state=1
)

# Extract the original data for these user splits
train_set = df[df["author_ID"].isin(train_users["author_ID"])]
test_set = df[df["author_ID"].isin(test_users["author_ID"])]
val_set = df[df["author_ID"].isin(val_users["author_ID"])]

# Verify the splits
train_ratio = len(train_set) / len(df)
test_ratio = len(test_set) / len(df)
val_ratio = len(val_set) / len(df)

print("Train ratio:", train_ratio)
print("Test ratio:", test_ratio)
print("Validation ratio:", val_ratio)

# Verify the distribution of political leaning of each set
print("Train political leaning distribution:")
print(train_set["label"].value_counts(normalize=True))

print("Test political leaning distribution:")
print(test_set["label"].value_counts(normalize=True))

print("Validation political leaning distribution:")
print(val_set["label"].value_counts(normalize=True))

# Check for overlaps in author_ID between the splits
train_ids = set(train_set["author_ID"])
test_ids = set(test_set["author_ID"])
val_ids = set(val_set["author_ID"])

train_test_overlap = train_ids.intersection(test_ids)
train_val_overlap = train_ids.intersection(val_ids)
test_val_overlap = test_ids.intersection(val_ids)

if train_test_overlap:
    print(f"Overlap between train and test sets: {train_test_overlap}")
else:
    print("No overlap between train and test sets.")

if train_val_overlap:
    print(f"Overlap between train and validation sets: {train_val_overlap}")
else:
    print("No overlap between train and validation sets.")

if test_val_overlap:
    print(f"Overlap between test and validation sets: {test_val_overlap}")
else:
    print("No overlap between test and validation sets.")
