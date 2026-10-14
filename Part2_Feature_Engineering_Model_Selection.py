
# ============================================================
# NLP PROJECT: DISASTER TWEET CLASSIFICATION
# PART 2: FEATURE ENGINEERING AND MODEL SELECTION
# ============================================================

import os
import re
import html
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

OUTPUT_DIR = "part2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PART1_FILE = os.path.join(
    "part1_output", "cleaned_tweets.csv"
)
EXCEL_FILE = "twitter_disaster (1).xlsx"

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

if os.path.exists(PART1_FILE):
    df = pd.read_csv(PART1_FILE)
    print("Loaded cleaned dataset from Part 1.")
elif os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    print("Loaded original Excel dataset.")
else:
    raise FileNotFoundError(
        "Dataset not found. Run Part 1 first, or place "
        "twitter_disaster (1).xlsx in the project folder."
    )

if "text" not in df.columns or "target" not in df.columns:
    raise ValueError(
        "Dataset must contain 'text' and 'target' columns."
    )

df = df.dropna(subset=["text", "target"]).copy()
df = df[df["target"].isin([0, 1])].copy()
df["text"] = df["text"].astype(str)
df["target"] = df["target"].astype(int)

df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# ------------------------------------------------------------
# 3. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    text = html.unescape(str(text))

    # Remove URLs and HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove usernames
    text = re.sub(r"@\w+", " ", text)

    # Preserve hashtag words
    text = re.sub(r"#(\w+)", r"\1", text)

    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

def extract_features(row):
    original_text = row["text"]
    cleaned = row["clean_text"]

    # Sentiment features
    sentiment = TextBlob(original_text).sentiment

    # TextBlob polarity ranges from -1 to +1.
    # Shift it to 0 to 1 for non-negative model inputs.
    polarity = (sentiment.polarity + 1) / 2

    # Subjectivity ranges from 0 to 1
    subjectivity = sentiment.subjectivity

    # Word and character features
    words = cleaned.split()

    # Count hashtags and mentions in the original tweet
    hashtags = len(re.findall(r"#\w+", original_text))
    mentions = len(re.findall(r"@\w+", original_text))

    # Other text features
    urls = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            original_text
        )
    )

    return pd.Series({
        "character_count": len(original_text),
        "word_count": len(words),
        "average_word_length": (
            np.mean([len(word) for word in words])
            if words else 0
        ),
        "hashtag_count": hashtags,
        "mention_count": mentions,
        "url_count": urls,
        "exclamation_count": original_text.count("!"),
        "question_count": original_text.count("?"),
        "uppercase_count": sum(
            1 for char in original_text if char.isupper()
        ),
        "sentiment_polarity": polarity,
        "sentiment_subjectivity": subjectivity
    })


feature_columns = [
    "character_count",
    "word_count",
    "average_word_length",
    "hashtag_count",
    "mention_count",
    "url_count",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "sentiment_polarity",
    "sentiment_subjectivity"
]

engineered_features = df.apply(extract_features, axis=1)
df[feature_columns] = engineered_features

print("\nEngineered features:")
print(df[["clean_text"] + feature_columns].head())

print("\nFeature summary:")
print(df[feature_columns].describe().round(2))

# Save engineered features
df.to_csv(
    os.path.join(OUTPUT_DIR, "engineered_features.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# ------------------------------------------------------------

X = df[["clean_text"] + feature_columns].copy()
y = df["target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 6. TF-IDF + NUMERIC FEATURE TRANSFORMER
# ------------------------------------------------------------

# TF-IDF captures word frequencies and word pairs.
# MinMaxScaler keeps numerical features non-negative,
# allowing Multinomial Naive Bayes to use them.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode"
            ),
            "clean_text"
        ),
        (
            "numeric",
            MinMaxScaler(),
            feature_columns
        )
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# 7. DEFINE CANDIDATE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}

# ------------------------------------------------------------
# 8. STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

comparison_results = []

print("\n" + "=" * 60)
print("CROSS-VALIDATION MODEL COMPARISON")
print("=" * 60)

for name, classifier in models.items():

    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", classifier)
    ])

    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    result = {
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean()
    }

    comparison_results.append(result)

    print(
        f"Accuracy: {result['Accuracy']:.4f} | "
        f"Precision: {result['Precision']:.4f} | "
        f"Recall: {result['Recall']:.4f} | "
        f"F1-score: {result['F1-score']:.4f}"
    )

comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.sort_values(
    "F1-score",
    ascending=False
).reset_index(drop=True)

print("\nModel comparison table:")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

# Plot cross-validation F1 scores
plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison_df,
    x="F1-score",
    y="Model"
)

plt.title("Model Comparison: 5-Fold Cross-Validation F1-score")
plt.xlabel("Mean F1-score")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 9. HYPERPARAMETER TUNING
# ------------------------------------------------------------

# Tune Logistic Regression using grid search.
# This keeps computation manageable while testing
# different regularization strengths and n-gram settings.

tuning_pipeline = Pipeline([
    (
        "features",
        ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=20000,
                        min_df=2,
                        sublinear_tf=True,
                        strip_accents="unicode"
                    ),
                    "clean_text"
                ),
                (
                    "numeric",
                    MinMaxScaler(),
                    feature_columns
                )
            ]
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

param_grid = {
    "features__text__ngram_range": [(1, 1), (1, 2)],
    "features__text__max_features": [10000, 20000],
    "classifier__C": [0.1, 1.0, 5.0]
}

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(grid_search.best_score_, 4))

# Save all grid-search results
grid_results = pd.DataFrame(grid_search.cv_results_)

grid_results.to_csv(
    os.path.join(OUTPUT_DIR, "grid_search_results.csv"),
    index=False
)

# ------------------------------------------------------------
# 10. FINAL TEST-SET EVALUATION
# ------------------------------------------------------------

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("FINAL TEST-SET RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-disaster", "Disaster"],
        zero_division=0
    )
)

pd.DataFrame([{
    "Best_model": "Tuned Logistic Regression",
    "CV_F1_score": grid_search.best_score_,
    "Test_accuracy": accuracy,
    "Test_precision": precision,
    "Test_recall": recall,
    "Test_F1_score": f1
}]).to_csv(
    os.path.join(OUTPUT_DIR, "final_metrics.csv"),
    index=False
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-disaster", "Disaster"],
    yticklabels=["Non-disaster", "Disaster"]
)

plt.title("Tuned Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 11. SAVE BEST MODEL AND TEST PREDICTIONS
# ------------------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    "best_disaster_model.pkl"
)

joblib.dump(best_model, model_path)

predictions = pd.DataFrame({
    "tweet": df.loc[X_test.index, "text"].values,
    "actual_label": y_test.values,
    "predicted_label": y_pred
})

predictions["actual_class"] = predictions["actual_label"].map({
    0: "Non-disaster",
    1: "Disaster"
})

predictions["predicted_class"] = predictions["predicted_label"].map({
    0: "Non-disaster",
    1: "Disaster"
})

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("PART 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("Best model saved to:", model_path)
print("All reports and graphs saved in:", OUTPUT_DIR)
