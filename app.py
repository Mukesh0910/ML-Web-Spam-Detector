# app.py

from flask import Flask, render_template, request
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Model and Vectorizer ---
# Paths to your saved model and vectorizer
MODEL_PATH = os.path.join('models', 'spam_classifier_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

model = None
vectorizer = None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Machine learning model and vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or Vectorizer file not found. Make sure '{MODEL_PATH}' and '{VECTORIZER_PATH}' exist.")
    print("Please run 'train_model.py' first to generate these files.")
    exit() # Exit if files are missing, as the app cannot run without them

# --- 3. NLTK Data (Ensure this is available for preprocessing) ---
# Corrected error handling for NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK stopwords data not found. Downloading...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK wordnet data not found. Downloading...")
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK punkt data not found. Downloading...")
    nltk.download('punkt')

print("All necessary NLTK data verified and loaded.")

# Initialize NLTK components for preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 4. Text Preprocessing Function (Must be IDENTICAL to train_model.py) ---
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    # Join back into a single string
    return ' '.join(processed_words)

# --- 5. Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form['message']
        if input_text:
            processed_input = preprocess_text(input_text)
            vectorized_input = vectorizer.transform([processed_input])
            prediction_raw = model.predict(vectorized_input)[0]
            prediction_result = "SPAM" if prediction_raw == 1 else "HAM"
        else:
            prediction_result = "Please enter a message to classify."

    return render_template('index.html', prediction=prediction_result, input_text=input_text)

# --- 6. Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True)