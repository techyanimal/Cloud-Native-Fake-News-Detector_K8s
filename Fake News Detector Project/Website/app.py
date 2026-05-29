from flask import Flask, request, jsonify, render_template
import joblib
from scipy.sparse import hstack
import re
import os

# Initialize the Flask app
app = Flask(__name__)

#Setting correct path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "MLcode",
    "Fake_news_predictor.pkl"
)

TEXT_PATH = os.path.join(
    BASE_DIR,
    "..",
    "MLcode",
    "tfidf_text.pkl"
)

TITLE_PATH = os.path.join(
    BASE_DIR,
    "..",
    "MLcode",
    "tfidf_title.pkl"
)
# Load the trained model and transformers
model = joblib.load(MODEL_PATH)
tfidf_title = joblib.load(TITLE_PATH)
tfidf_text = joblib.load(TEXT_PATH)

# Text processing function
def text_processing(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|https\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data['title']
    text = data['text']
    new_title = text_processing(title)
    new_text = text_processing(text)
    new_title_tfidf = tfidf_title.transform([new_title])
    new_text_tfidf = tfidf_text.transform([new_text])
    new_data_combined = hstack([new_title_tfidf, new_text_tfidf])
    prediction = model.predict(new_data_combined)
    
    result = "REAL" if prediction[0] == 0 else "FAKE"
    
    return jsonify({
        "prediction": result })

if __name__ == "__main__":
    app.run(debug=True)
