import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === CONFIGURATION ===
NEWS_API_KEY = "34b3813e4cd24d15b928e89473e7b58e"
NEW_DATA_FILE = "realtime_news_data.csv"
TRAIN_THRESHOLD = 10
CONFIDENCE_THRESHOLD = 0.9

# Load model/vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize News API client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Fetch latest news
def fetch_latest_news(language="en", num_articles=100):
    response = newsapi.get_everything(
        q="fake OR real",
        language=language,
        sort_by="publishedAt",
        page_size=num_articles
    )
    articles = response.get("articles", [])
    return [(article["title"] + " " + (article.get("description") or "")) for article in articles if article.get("title")]

# Classify news and get high-confidence predictions
def classify_news(news_list):
    transformed_news = vectorizer.transform(news_list)
    probabilities = model.predict_proba(transformed_news)
    predictions = model.predict(transformed_news)

    high_confidence_data = []
    for news, pred, prob in zip(news_list, predictions, probabilities):
        confidence = max(prob)
        if confidence >= CONFIDENCE_THRESHOLD:
            high_confidence_data.append({"text": news, "label": pred, "confidence": confidence})

    return high_confidence_data

# Save new data for retraining
def save_new_data(news_list):
    df_new = pd.DataFrame(news_list, columns=["text", "label", "confidence"])
    if os.path.exists(NEW_DATA_FILE):
        df_new.to_csv(NEW_DATA_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(NEW_DATA_FILE, mode="w", header=True, index=False)
    print(f"✅ Saved {len(df_new)} new articles to {NEW_DATA_FILE}")

# Show confusion matrix
def show_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')

    # Label each cell
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha='center', va='center', color='black', fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Real (0)', 'Fake (1)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Real (0)', 'Fake (1)'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix with Labels")
    plt.tight_layout()
    plt.show()

# Retrain the model if new high-confidence data is available
def retrain_model():
    print("\n🔄 Retraining model with new data...")

    # Load the original dataset
    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("True.csv")
    df_fake["label"] = 1
    df_real["label"] = 0
    df = pd.concat([df_fake, df_real])

    # Load new high-confidence data
    if os.path.exists(NEW_DATA_FILE):
        df_new = pd.read_csv(NEW_DATA_FILE)
        df_new = df_new.drop(columns=["confidence"], errors="ignore")
        df = pd.concat([df, df_new])

    df = df[['text', 'label']].dropna()

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_model = MultinomialNB()
    new_model.fit(X_train, y_train)

    joblib.dump(new_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    y_pred = new_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n✅ Model updated! New Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:\n", report)
    show_confusion(y_test, y_pred)

# Check if retraining is necessary
def should_retrain():
    if os.path.exists(NEW_DATA_FILE):
        df_new = pd.read_csv(NEW_DATA_FILE)
        return len(df_new) >= TRAIN_THRESHOLD
    return False

# Main process
if __name__ == "__main__":
    print("Fetching live news...")
    latest_news = fetch_latest_news(num_articles=100)

    if not latest_news:
        print("No news articles found.")
    else:
        print("\nClassifying news articles...\n")
        high_confidence_news = classify_news(latest_news)

        if high_confidence_news:
            print(f"✅ {len(high_confidence_news)} high-confidence articles saved for retraining.")
            save_new_data(high_confidence_news)

        # Evaluate existing model confusion matrix if no retraining
        if not should_retrain():
            print("\n📌 Retraining skipped. Showing confusion matrix of current model.")
            X_unlabeled = vectorizer.transform([x['text'] for x in high_confidence_news])
            y_pred_unlabeled = [x['label'] for x in high_confidence_news]
            show_confusion(y_pred_unlabeled, y_pred_unlabeled)  # Confusion matrix for classified unlabeled news

        # Retrain if threshold reached
        if should_retrain():
            retrain_model()
            print("🔁 Training data reset. Ready for new samples!")

        print("\nProcess completed.")
