from flask import Flask, render_template, request
import joblib
import shap
import numpy as np
from scipy.sparse import csr_matrix
from newsapi import NewsApiClient
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

matplotlib.use('Agg')

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

newsapi = NewsApiClient(api_key="34b3813e4cd24d15b928e89473e7b58e")

background_text = ["background text"]
background_matrix = vectorizer.transform(background_text).toarray()
explainer = shap.Explainer(lambda x: model.predict_proba(x), background_matrix, max_evals=1000)

SHAP_THRESHOLD = 0.01
MIN_WORDS_REQUIRED = 20  #  minimum word requirement


@app.route("/")
def home():
    articles = newsapi.get_top_headlines(language="en", country="us")["articles"]
    return render_template("index.html", articles=articles)


@app.route("/check", methods=["POST"])
def check_news():
    text = request.form["news_text"]
    word_count = len(text.strip().split())

    if word_count < MIN_WORDS_REQUIRED:
        warning_msg = f"❌ Please enter at least {MIN_WORDS_REQUIRED} words for proper classification. You entered {word_count}."
        articles = newsapi.get_top_headlines(language="en", country="us")["articles"]
        return render_template("index.html", articles=articles, prediction=warning_msg)

    input_text = vectorizer.transform([text])
    input_array = input_text.toarray()

    prediction = model.predict(input_text)
    result = "Fake News" if prediction[0] == 1 else "Real News"

    shap_values = explainer(input_array)
    feature_names = vectorizer.get_feature_names_out()

    shap_values_filtered = []
    input_array_filtered = []
    feature_names_filtered = []

    for i in range(len(feature_names)):
        if abs(shap_values.values[0, i, 1]) >= SHAP_THRESHOLD:
            shap_values_filtered.append(shap_values.values[0, i, 1])
            input_array_filtered.append(input_array[0, i])
            feature_names_filtered.append(feature_names[i])

    shap_values_filtered = np.array(shap_values_filtered).reshape(1, -1)
    input_array_filtered = np.array(input_array_filtered).reshape(1, -1)
    feature_names_filtered = np.array(feature_names_filtered)

    plt.figure()
    shap.summary_plot(shap_values_filtered, input_array_filtered, feature_names=feature_names_filtered, show=False)
    plt.title("SHAP Beeswarm Plot for Prediction")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    summary = generate_beeswarm_summary(shap_values_filtered, feature_names_filtered, result)

    return render_template("result.html", text=text, result=result, plot_url=plot_url, summary=summary)


def generate_beeswarm_summary(shap_values, feature_names, prediction):
    top_indices = np.argsort(np.abs(shap_values[0]))[-5:][::-1]
    top_words = feature_names[top_indices]
    top_shap_values = shap_values[0][top_indices]

    impact_direction = []
    for value in top_shap_values:
        impact_direction.append("supports 'Fake News'" if value > 0 else "supports 'Real News'")

    summary = f"The following words had the most influence on the prediction:\n"
    for word, direction in zip(top_words, impact_direction):
        summary += f"- '{word}' ({direction})\n"

    return summary


if __name__ == "__main__":
    app.run(debug=True)
