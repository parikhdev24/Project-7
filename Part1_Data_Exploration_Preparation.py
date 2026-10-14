
# ==========================================================
# NLP PROJECT: DISASTER TWEET CLASSIFICATION
# PART 1: DATA EXPLORATION AND PREPARATION
# ==========================================================

import os
import re
import html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------
# 1. LOAD THE DATASET
# ----------------------------------------------------------

DATA_FILE = "twitter_disaster (1).xlsx"
OUTPUT_DIR = "part1_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(DATA_FILE)

print("=" * 60)
print("NLP DISASTER TWEET CLASSIFICATION")
print("PART 1: DATA EXPLORATION AND PREPARATION")
print("=" * 60)

# ----------------------------------------------------------
# 2. EXPLORE DATASET STRUCTURE
# ----------------------------------------------------------

print("\n1. DATASET INFORMATION")
print("-" * 40)

print("Dataset shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Verify required columns
required_columns = ["text", "target"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing. "
            f"Available columns: {df.columns.tolist()}"
        )

# ----------------------------------------------------------
# 3. CLEAN BASIC DATA ISSUES
# ----------------------------------------------------------

# Remove rows with missing text or labels
df = df.dropna(subset=["text", "target"]).copy()

# Keep only valid binary labels: 0 and 1
df = df[df["target"].isin([0, 1])].copy()
df["target"] = df["target"].astype(int)

# Convert tweet text to strings and remove empty tweets
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()

print("\nDataset shape after basic cleaning:", df.shape)

print("\nClass labels:")
print("0 = Non-disaster")
print("1 = Disaster")

print("\nClass counts:")
print(df["target"].value_counts().sort_index())

# ----------------------------------------------------------
# 4. VISUALIZE CLASS DISTRIBUTION
# ----------------------------------------------------------

class_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=["Non-disaster (0)", "Disaster (1)"],
    y=[
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
)

plt.title("Distribution of Disaster and Non-disaster Tweets")
plt.xlabel("Tweet Class")
plt.ylabel("Number of Tweets")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=300
)

plt.show()

print("\nClass percentages:")
print(
    (df["target"].value_counts(normalize=True) * 100).round(2)
)

# ----------------------------------------------------------
# 5. ANALYZE FREQUENT WORDS AND KEYWORDS
# ----------------------------------------------------------

def show_frequent_words(texts, title, filename, top_n=20):
    """Find and plot the most frequent words."""

    vectorizer = CountVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        max_features=10000,
        ngram_range=(1, 1)
    )

    word_matrix = vectorizer.fit_transform(texts)

    frequencies = word_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    word_frequency = pd.DataFrame({
        "word": words,
        "frequency": frequencies
    })

    word_frequency = word_frequency.sort_values(
        "frequency",
        ascending=False
    ).head(top_n)

    print(f"\n{title}")
    print(word_frequency.to_string(index=False))

    plot_data = word_frequency.sort_values("frequency")

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["word"],
        plot_data["frequency"]
    )

    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.show()

    return word_frequency


# Disaster tweet keywords
disaster_words = show_frequent_words(
    df.loc[df["target"] == 1, "text"],
    "Top Words in Disaster Tweets",
    "disaster_keywords.png"
)

# Non-disaster tweet keywords
non_disaster_words = show_frequent_words(
    df.loc[df["target"] == 0, "text"],
    "Top Words in Non-disaster Tweets",
    "non_disaster_keywords.png"
)

# ----------------------------------------------------------
# 6. CLEAN TEXT DATA
# ----------------------------------------------------------

def clean_text(text):
    """Remove URLs, usernames, punctuation and special characters."""

    text = html.unescape(str(text))

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", " ", text)

    # Preserve hashtag words but remove the # symbol
    text = re.sub(r"#(\w+)", r"\1", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

print("\nOriginal tweet example:")
print(df["text"].iloc[0])

print("\nCleaned tweet example:")
print(df["clean_text"].iloc[0])

# ----------------------------------------------------------
# 7. TOKENIZE TEXT INTO INDIVIDUAL WORDS
# ----------------------------------------------------------

df["tokens"] = df["clean_text"].apply(
    lambda text: text.split()
)

print("\nTokenization example:")
print("Cleaned text:", df["clean_text"].iloc[0])
print("Tokens:", df["tokens"].iloc[0])

# ----------------------------------------------------------
# 8. CONVERT LABELS INTO NUMERICAL FORMAT
# ----------------------------------------------------------

# The dataset already uses:
# 0 = Non-disaster
# 1 = Disaster

df["label"] = df["target"].astype(int)

print("\nLabel encoding:")
print(df[["text", "target", "label"]].head())

# ----------------------------------------------------------
# 9. SPLIT DATA INTO TRAINING AND TESTING SETS
# ----------------------------------------------------------

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAIN/TEST SPLIT RESULTS")
print("-" * 40)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

# ----------------------------------------------------------
# 10. SAVE PREPARED DATA AND SPLITS
# ----------------------------------------------------------

# Save cleaned dataset
df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_tweets.csv"),
    index=False,
    encoding="utf-8-sig"
)

# Save training and testing data separately
pd.DataFrame({
    "clean_text": X_train,
    "label": y_train
}).to_csv(
    os.path.join(OUTPUT_DIR, "train_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "clean_text": X_test,
    "label": y_test
}).to_csv(
    os.path.join(OUTPUT_DIR, "test_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

# Save keyword analysis results
disaster_words.to_csv(
    os.path.join(OUTPUT_DIR, "disaster_word_frequency.csv"),
    index=False
)

non_disaster_words.to_csv(
    os.path.join(OUTPUT_DIR, "non_disaster_word_frequency.csv"),
    index=False
)

# ----------------------------------------------------------
# 11. COMPLETION SUMMARY
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("PART 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Output folder:", OUTPUT_DIR)

print("\nGenerated files:")
print("1. cleaned_tweets.csv")
print("2. train_data.csv")
print("3. test_data.csv")
print("4. disaster_word_frequency.csv")
print("5. non_disaster_word_frequency.csv")
print("6. class_distribution.png")
print("7. disaster_keywords.png")
print("8. non_disaster_keywords.png")
