# 1. Import Library
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 2. Medefinsikan SpamClassifier Class (Menggunakan Bayesian Network)
class SpamClassifierBN:
    def __init__(self):
        """Initializes the model and other components."""
        self.model_pipeline = None
        self.X_test = None
        self.y_test = None

    def _preprocess_text(self, text):
        """Private function to clean text."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def train(self, file_path):
        """Loads data, preprocesses, and trains the classification model."""
        # --- Melakukan Load and Clean Data ---
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Make sure it's in the same directory.")
            return

        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df = df.rename(columns={'v1': 'Category', 'v2': 'Message'})
        df.dropna(inplace=True)
        df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

        # --- Preprocessing ---
        df['processed_message'] = df['Message'].apply(self._preprocess_text)
        
        # Kita hanya akan menggunakan fitur teks untuk Naive Bayes
        X = df['processed_message']
        y = df['Category']
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model_pipeline = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
            ('classifier', MultinomialNB())  
        ])
        
        print("--- Starting Bayesian Network Model Training ---")
        self.model_pipeline.fit(X_train, y_train)
        print("--- Model Successfully Trained! ---")
    
    def evaluate(self):
        """Evaluates the model on the test set and prints the results."""
        if self.model_pipeline is None:
            print("Model has not been trained yet. Please run .train() first.")
            return

        y_pred = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=['Ham', 'Spam'])
        
        print("\n" + "="*40)
        print("MODEL EVALUATION RESULTS (BAYESIAN NETWORK)")
        print("="*40)
        print(f"\nMODEL ACCURACY: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)

    def predict_interactive(self):
        """Runs an interactive loop for real-time predictions."""
        if self.model_pipeline is None:
            print("Model has not been trained yet. Please run .train() first.")
            return

        while True:
            input_text = input("\nEnter a message to check (or type 'exit' to quit): ")
            if input_text.lower() == 'exit':
                break

            # Prediksi langsung pada teks input (pipeline akan mengurus preprocessing)
            prediction = self.model_pipeline.predict([input_text])
            prediction_proba = self.model_pipeline.predict_proba([input_text])
            
            spam_prob = prediction_proba[0][1] * 100
            
            if prediction[0] == 1:
                print(f"➡️  Prediction: SPAM  ({spam_prob:.2f}% spam probability)")
            else:
                print(f"➡️  Prediction: HAM ({spam_prob:.2f}% spam probability)")
        
        print("\nInteractive session ended.")

# 3.Program Utama
if __name__ == "__main__":
    # Meningisialisasi Classifier
    classifier_bn = SpamClassifierBN()
    
    # Melatih model with the data
    classifier_bn.train('spam.csv')
    
    # Menjalankan Interaktif prediction mode
    classifier_bn.predict_interactive()
    
    classifier_bn.evaluate()