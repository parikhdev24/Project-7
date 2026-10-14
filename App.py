
import os
import re
import html
import uuid
import joblib
import numpy as np
import pandas as pd

from io import BytesIO
from flask import (
    Flask, render_template, request,
    send_file, flash, redirect, url_for
)
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-development-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "part3_output", "deployment_model.pkl"),
    os.path.join(BASE_DIR, "part2_output", "best_disaster_model.pkl")
]

MODEL_PATH = next(
    (path for path in MODEL_CANDIDATES if os.path.exists(path)),
    None
)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Trained model not found. Run Part 2 or Part 3 first."
    )

model = joblib.load(MODEL_PATH)

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

OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    """Use the same text cleaning logic as model training."""

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
    """Generate the numeric features expected by the model."""

    sentiment = TextBlob(text).sentiment
    words = cleaned.split()

    return {
        "character_count": len(text),
        "word_count": len(words),
        "average_word_length": (
            float(np.mean([len(word) for word in words]))
            if words else 0.0
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


def make_model_input(tweet):
    cleaned = clean_text(tweet)

    row = {"clean_text": cleaned}
    row.update(extract_features(tweet, cleaned))

    # Preserve the exact column order expected by the pipeline.
    return pd.DataFrame(
        [[row[column] for column in ["clean_text"] + FEATURE_COLUMNS]],
        columns=["clean_text"] + FEATURE_COLUMNS
    )


def predict_tweet(tweet):
    """Return a classification and, where available, a score."""

    model_input = make_model_input(tweet)
    prediction = int(model.predict(model_input)[0])

    result = {
        "tweet": tweet,
        "clean_text": model_input.iloc[0]["clean_text"],
        "label": prediction,
        "classification": (
            "Disaster-related"
            if prediction == 1
            else "Non-disaster-related"
        ),
        "score": None,
        "score_label": None
    }

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])
        result["score"] = round(probability * 100, 2)
        result["score_label"] = "Estimated disaster probability"

    elif hasattr(model, "decision_function"):
        # Decision scores are NOT probabilities.
        decision = float(model.decision_function(model_input)[0])
        result["score"] = round(decision, 4)
        result["score_label"] = "Model decision score (not a probability)"

    return result


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    batch_results = None
    download_id = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        uploaded_file = request.files.get("csv_file")

        if uploaded_file and uploaded_file.filename:
            if not uploaded_file.filename.lower().endswith(".csv"):
                flash("Please upload a CSV file.")
                return render_template("index.html", tweet=tweet)

            try:
                batch_df = pd.read_csv(uploaded_file)

                if "text" not in batch_df.columns:
                    flash("CSV must contain a column named 'text'.")
                    return render_template("index.html", tweet=tweet)

                if len(batch_df) > 5000:
                    flash("Please upload no more than 5,000 tweets at once.")
                    return render_template("index.html", tweet=tweet)

                batch_df = batch_df.dropna(subset=["text"]).copy()
                batch_df["text"] = batch_df["text"].astype(str)
                batch_df = batch_df[
                    batch_df["text"].str.strip() != ""
                ]

                if batch_df.empty:
                    flash("No non-empty tweets were found in the CSV.")
                    return render_template("index.html", tweet=tweet)

                records = [
                    predict_tweet(text)
                    for text in batch_df["text"].tolist()
                ]

                batch_results = pd.DataFrame(records)

                download_id = uuid.uuid4().hex
                output_path = os.path.join(
                    OUTPUT_DIR, f"{download_id}.csv"
                )
                batch_results.to_csv(
                    output_path, index=False, encoding="utf-8-sig"
                )

                batch_results = batch_results.head(100).to_dict(
                    orient="records"
                )

            except Exception as exc:
                app.logger.exception("CSV processing failed")
                flash(f"Could not process the CSV: {exc}")

        elif tweet:
            try:
                result = predict_tweet(tweet)
            except Exception:
                app.logger.exception("Tweet prediction failed")
                flash(
                    "Prediction failed. Check that the model "
                    "matches the Part 2/Part 3 feature pipeline."
                )
        else:
            flash("Enter a tweet or upload a CSV file.")

    return render_template(
        "index.html",
        result=result,
        tweet=tweet,
        batch_results=batch_results,
        download_id=download_id
    )


@app.route("/download/<download_id>")
def download_results(download_id):
    # Accept only generated UUID-style filenames.
    if not re.fullmatch(r"[a-f0-9]{32}", download_id):
        return "Invalid download ID", 400

    output_path = os.path.join(
        OUTPUT_DIR, f"{download_id}.csv"
    )

    if not os.path.isfile(output_path):
        return "File not found or already removed.", 404

    return send_file(
        output_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="disaster_tweet_predictions.csv"
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
