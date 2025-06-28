import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Constants
REALTIME_FILE = "realtime_news_unlabeled.csv"
LABELED_FILE = "realtime_news_labeled.csv"

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def load_labeled_data():
    """Load the labeled data (classified data from `realtime_news.py`)."""
    if os.path.exists(LABELED_FILE):
        df = pd.read_csv(LABELED_FILE)
        return df
    else:
        return pd.DataFrame(columns=["text", "label"])

def simulate_ground_truth(df):
    """Simulate ground truth labels for the unlabeled data."""
    # This is just a simulation. Replace it with proper ground truth if available.
    # Randomly simulate "real" (0) or "fake" (1) labels for the unlabeled data
    df['true_label'] = np.random.choice([0, 1], size=len(df))
    return df

def evaluate_model(X_test, y_pred, dataset_name="Test Data"):
    """Evaluate model performance on a given dataset."""
    print(f"🟢 Evaluation on {dataset_name}")

    # Simulate ground truth for evaluation
    y_true = simulate_ground_truth(X_test)  # Simulated true labels

    print(f"🟢 Evaluation on {dataset_name} - Accuracy: {np.mean(y_true['true_label'] == y_pred):.4f}")

    # Display classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true['true_label'], y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true['true_label'], y_pred)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha='center', va='center', color='black', fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Real (0)', 'Fake (1)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Real (0)', 'Fake (1)'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.tight_layout()
    plt.show()

def classify_unlabeled_data():
    """Classify the unlabeled data and generate confusion matrix."""
    if os.path.exists(REALTIME_FILE):
        df_unlabeled = pd.read_csv(REALTIME_FILE)
        df_unlabeled = df_unlabeled[['text']]  # Assuming 'text' is the column with the news articles

        # Vectorize the text and make predictions
        X_unlabeled = vectorizer.transform(df_unlabeled['text'])
        y_pred_unlabeled = model.predict(X_unlabeled)

        # Save the predictions (label) in the dataframe
        df_unlabeled['label'] = y_pred_unlabeled

        # Now evaluate the confusion matrix on the classified unlabeled data
        print("\n📝 Evaluating the confusion matrix on Unlabeled Data:")
        evaluate_model(df_unlabeled, y_pred_unlabeled, "Unlabeled Data")

        # Save the labeled articles to the labeled file for further training
        df_unlabeled.to_csv(LABELED_FILE, index=False)

    else:
        print("⚠️ No real-time data found to classify.")

if __name__ == "__main__":
    # Classify the new unlabeled data
    classify_unlabeled_data()
