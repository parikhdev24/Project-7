
# ============================================================
# NLP PROJECT: DISASTER TWEET CLASSIFICATION
# PART 3: MODEL EVALUATION AND VALIDATION
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

from sklearn.model_selection import train_test_split
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
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

OUTPUT_DIR = "part3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PART1_FILE = os.path.join(
    "part1_output", "cleaned_tweets.csv"
)

EXCEL_FILE = "twitter_disaster (1).xlsx"

SAVED_BEST_MODEL = os.path.join(
    "part2_output", "best_disaster_model.pkl"
)

FEATURE_COLUMNS = [
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

RANDOM_STATE = 42

# ------------------------------------------------------------
# 2. LOAD AND PREPARE DATA
# ------------------------------------------------------------

if os.path.exists(PART1_FILE):
    df = pd.read_csv(PART1_FILE)
    print("Loaded Part 1 dataset.")
elif os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    print("Loaded original Excel dataset.")
else:
    raise FileNotFoundError(
        "Dataset not found. Run Part 1 first or place "
        "twitter_disaster (1).xlsx beside this script."
    )

if not {"text", "target"}.issubset(df.columns):
    raise ValueError("Dataset requires text and target columns.")

df = df.dropna(subset=["text", "target"]).copy()
df = df[df["target"].isin([0, 1])].copy()
df["text"] = df["text"].astype(str)
df["target"] = df["target"].astype(int)

df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# ------------------------------------------------------------
# 3. TEXT CLEANING AND FEATURE ENGINEERING
# Must match the Part 2 preprocessing.
# ------------------------------------------------------------

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text, cleaned):
    sentiment = TextBlob(text).sentiment
    words = cleaned.split()

    return {
        "character_count": len(text),
        "word_count": len(words),
        "average_word_length": (
            np.mean([len(word) for word in words])
            if words else 0
        ),
        "hashtag_count": len(re.findall(r"#\w+", text)),
        "mention_count": len(re.findall(r"@\w+", text)),
        "url_count": len(
            re.findall(r"https?://\S+|www\.\S+", text)
        ),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_count": sum(
            1 for char in text if char.isupper()
        ),
        "sentiment_polarity": (
            sentiment.polarity + 1
        ) / 2,
        "sentiment_subjectivity": sentiment.subjectivity
    }


df["clean_text"] = df["text"].apply(clean_text)

feature_df = pd.DataFrame(
    [
        extract_features(text, cleaned)
        for text, cleaned in zip(df["text"], df["clean_text"])
    ],
    index=df.index
)

for column in FEATURE_COLUMNS:
    df[column] = feature_df[column]

X = df[["clean_text"] + FEATURE_COLUMNS]
y = df["target"]

# Same split configuration as Part 2.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 4. BUILD THE FEATURE TRANSFORMER
# ------------------------------------------------------------

def create_pipeline(classifier):
    preprocessor = ColumnTransformer([
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
            FEATURE_COLUMNS
        )
    ])

    return Pipeline([
        ("features", preprocessor),
        ("classifier", classifier)
    ])


# ------------------------------------------------------------
# 5. TRAIN CANDIDATE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": create_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    ),
    "Multinomial Naive Bayes": create_pipeline(
        MultinomialNB()
    ),
    "Random Forest": create_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    ),
    "Linear SVM": create_pipeline(
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
}

print("\nTraining candidate models...")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

# Load the tuned model from Part 2 if available.
if os.path.exists(SAVED_BEST_MODEL):
    models["Tuned Logistic Regression"] = joblib.load(
        SAVED_BEST_MODEL
    )
    print("\nLoaded tuned model from Part 2.")
else:
    print(
        "\nPart 2 tuned model not found. "
        "Evaluating the four candidate models instead."
    )

# ------------------------------------------------------------
# 6. EVALUATE ALL MODELS
# ------------------------------------------------------------

results = []
predictions_by_model = {}
scores_by_model = {}

for name, model in models.items():

    print(f"\nEvaluating {name}...")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Decision scores support ROC/PR curves for SVM.
    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(X_test)[:, 1]
        train_scores = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, "decision_function"):
        test_scores = model.decision_function(X_test)
        train_scores = model.decision_function(X_train)
    else:
        test_scores = test_pred.astype(float)
        train_scores = train_pred.astype(float)

    metrics = {
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(
            y_test, test_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, test_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, test_scores),
        "Average Precision": average_precision_score(
            y_test, test_scores
        )
    }

    results.append(metrics)
    predictions_by_model[name] = test_pred
    scores_by_model[name] = {
        "test": test_scores,
        "train": train_scores
    }

    print(
        f"Train accuracy: {metrics['Train Accuracy']:.4f}\n"
        f"Test accuracy:  {metrics['Test Accuracy']:.4f}\n"
        f"Precision:      {metrics['Precision']:.4f}\n"
        f"Recall:         {metrics['Recall']:.4f}\n"
        f"F1-score:       {metrics['F1-score']:.4f}\n"
        f"ROC-AUC:        {metrics['ROC-AUC']:.4f}"
    )

results_df = pd.DataFrame(results).sort_values(
    "F1-score", ascending=False
)

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_evaluation_results.csv"),
    index=False
)

# ------------------------------------------------------------
# 7. PLOT MODEL METRICS
# ------------------------------------------------------------

metric_columns = [
    "Test Accuracy", "Precision", "Recall", "F1-score"
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metric_columns,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)
plt.title("Comparison of Model Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "metrics_comparison.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 8. CONFUSION MATRICES
# ------------------------------------------------------------

for name, predictions in predictions_by_model.items():

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-disaster", "Disaster"],
        yticklabels=["Non-disaster", "Disaster"]
    )

    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", name)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"confusion_matrix_{safe_name}.png"
        ),
        dpi=300
    )
    plt.close()

# ------------------------------------------------------------
# 9. ROC CURVES
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

for name, scores in scores_by_model.items():

    fpr, tpr, _ = roc_curve(
        y_test, scores["test"]
    )
    auc = roc_auc_score(y_test, scores["test"])

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curves.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 10. PRECISION-RECALL CURVES
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

for name, scores in scores_by_model.items():

    precision_values, recall_values, _ = (
        precision_recall_curve(y_test, scores["test"])
    )

    ap = average_precision_score(
        y_test, scores["test"]
    )

    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (AP={ap:.3f})"
    )

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "precision_recall_curves.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 11. CLASSIFICATION REPORT FOR THE BEST TEST F1-SCORE
# ------------------------------------------------------------

# This identifies the leading model on this test set for
# reporting. For a rigorous deployment decision, use the
# Part 2 cross-validation results to select the model and
# reserve the test set for final confirmation.

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
best_predictions = predictions_by_model[best_name]

print("\nBest test-set F1-score model:", best_name)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["Non-disaster", "Disaster"],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 12. CHECK FOR OVERFITTING / UNDERFITTING
# ------------------------------------------------------------

results_df["Accuracy Gap"] = (
    results_df["Train Accuracy"] -
    results_df["Test Accuracy"]
)

print("\nTrain/Test Accuracy Gap:")
print(
    results_df[
        ["Model", "Train Accuracy", "Test Accuracy", "Accuracy Gap"]
    ].round(4).to_string(index=False)
)

print("\nInterpretation guide:")
print("- Large positive gap: possible overfitting.")
print("- Low training and testing scores: possible underfitting.")
print("- Similar train/test scores with good validation metrics:")
print("  better evidence of generalization.")
print("These are diagnostic clues, not absolute rules.")

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_evaluation_results.csv"),
    index=False
)

# ------------------------------------------------------------
# 13. ROBUSTNESS CHECKS
# ------------------------------------------------------------

def make_model_input(tweet):
    cleaned = clean_text(tweet)
    features = extract_features(tweet, cleaned)

    row = {"clean_text": cleaned}
    row.update(features)

    return pd.DataFrame([row])


robustness_examples = [
    "Flooding has destroyed several homes.",
    "FLOODING HAS DESTROYED SEVERAL HOMES!!!",
    "Flooding has destroyed several homes https://example.com",
    "Wildfire spreading near homes #emergency",
    "That test was a disaster!",
    "This new phone is fire!"
]

robustness_rows = []

for tweet in robustness_examples:

    row = make_model_input(tweet)

    prediction = int(best_model.predict(row)[0])

    if hasattr(best_model, "predict_proba"):
        probability = float(
            best_model.predict_proba(row)[0, 1]
        )
    else:
        probability = np.nan

    robustness_rows.append({
        "tweet": tweet,
        "predicted_label": prediction,
        "predicted_class": (
            "Disaster" if prediction == 1
            else "Non-disaster"
        ),
        "disaster_probability": probability
    })

robustness_df = pd.DataFrame(robustness_rows)

robustness_df.to_csv(
    os.path.join(OUTPUT_DIR, "robustness_checks.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nRobustness test examples:")
print(robustness_df.to_string(index=False))

# ------------------------------------------------------------
# 14. SAVE DEPLOYMENT MODEL
# ------------------------------------------------------------

# Keep the Part 2 tuned model as the deployment candidate
# if available; otherwise save the best model from this
# evaluation run. Prefer CV-based selection for production.
deployment_model = (
    models["Tuned Logistic Regression"]
    if "Tuned Logistic Regression" in models
    else best_model
)

joblib.dump(
    deployment_model,
    os.path.join(OUTPUT_DIR, "deployment_model.pkl")
)

print("\n" + "=" * 65)
print("PART 3 COMPLETED SUCCESSFULLY!")
print("=" * 65)
print("Evaluation outputs saved in:", OUTPUT_DIR)
print("Deployment candidate saved as: deployment_model.pkl")
