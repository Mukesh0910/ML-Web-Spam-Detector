# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB # You can also try from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import joblib # For saving/loading models
import os

# --- 1. NLTK Data Download (Ensure these are available) ---
# This part ensures that necessary NLTK data is present.
# You already ran 'python -m nltk.downloader all', so this is mostly a safeguard.
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK data not found. Downloading necessary components...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# --- 2. Data Loading ---
print("Loading dataset...")
data_path = os.path.join('data', 'spam.csv')
try:
    df = pd.read_csv(data_path, encoding='latin-1') # 'latin-1' encoding is common for this dataset
except FileNotFoundError:
    print(f"Error: '{data_path}' not found. Please ensure 'spam.csv' is in the 'data/' folder.")
    exit() # Exit if dataset not found

# Rename columns for clarity (original columns are 'v1' and 'v2')
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']] # Keep only the 'label' and 'message' columns

# Convert labels to numerical (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("Dataset loaded and labels mapped.")
print(f"Total messages: {len(df)}")
print(f"Spam messages: {df['label'].sum()}")
print(f"Ham messages: {len(df) - df['label'].sum()}")


# --- 3. Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

print("\nApplying preprocessing to messages... This may take a moment.")
df['processed_message'] = df['message'].apply(preprocess_text)
print("Preprocessing complete.")

# --- 4. Feature Extraction (TF-IDF Vectorization) ---
X = df['processed_message']
y = df['label']

print("Training TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
X_vectorized = tfidf_vectorizer.fit_transform(X)
print(f"TF-IDF Vectorizer trained. Number of features: {X_vectorized.shape[1]}")


# --- 5. Split Data ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- 6. Model Training (Multinomial Naive Bayes) ---
print("\nTraining Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete.")

# --- 7. Model Evaluation ---
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Spam): {precision:.4f}") # Precision for class 1 (Spam)
print(f"Recall (Spam): {recall:.4f}")     # Recall for class 1 (Spam)
print(f"F1-Score (Spam): {f1:.4f}")       # F1-Score for class 1 (Spam)
print("\nConfusion Matrix:")
print(conf_matrix)
print("   (Predicted)")
print("       Ham   Spam")
print(f" (Actual) Ham {conf_matrix[0][0]:<5} {conf_matrix[0][1]}")
print(f"          Spam {conf_matrix[1][0]:<5} {conf_matrix[1][1]}")


# --- 8. Save Model and Vectorizer ---
print("\nSaving trained model and TF-IDF vectorizer...")
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(model, os.path.join(models_dir, 'spam_classifier_model.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
print(f"Model and Vectorizer saved successfully to '{models_dir}/'.")

# --- Quick Test of Saved Model (Optional - for verification) ---
print("\n--- Quick Test with Saved Model ---")
loaded_model = joblib.load(os.path.join(models_dir, 'spam_classifier_model.pkl'))
loaded_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

test_message_spam = "Congratulations! You've won a FREE holiday! Call 0800..."
test_message_ham = "Hey, let's meet up for coffee tomorrow at 10 AM?"
test_message_spam2 = "WINNER! Claim your cash prize! Text WIN to 12345."


def predict_message(message, model, vectorizer, preprocessor_func):
    processed = preprocessor_func(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "SPAM" if prediction == 1 else "HAM"

print(f"'{test_message_spam}' is predicted as: {predict_message(test_message_spam, loaded_model, loaded_vectorizer, preprocess_text)}")
print(f"'{test_message_ham}' is predicted as: {predict_message(test_message_ham, loaded_model, loaded_vectorizer, preprocess_text)}")
print(f"'{test_message_spam2}' is predicted as: {predict_message(test_message_spam2, loaded_model, loaded_vectorizer, preprocess_text)}")