# 1. Import Library
import pandas as pd
import re
from googletrans import Translator, LANGUAGES 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 2. Definisikan SpamClassifier Class (Menggunakan Bayesian Network)
class SpamClassifierBN:
    def __init__(self):
        """Initializes the model and other components."""
        self.model_pipeline = None
        self.X_test = None
        self.y_test = None
        self.translator = Translator() 

    def _preprocess_text(self, text):
        """Private function to clean text."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def train(self, file_path):
        """Loads data, preprocesses, and trains the classification model."""
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Make sure it's in the same directory.")
            return

        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df = df.rename(columns={'v1': 'Category', 'v2': 'Message'})
        df.dropna(inplace=True)
        df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

        df['processed_message'] = df['Message'].apply(self._preprocess_text)
        
        X = df['processed_message']
        y = df['Category']
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Pipeline ini menggunakan Naive Bayes (Bayesian Network) sebagai classifier
        self.model_pipeline = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
            ('classifier', MultinomialNB())  
        ])
        
        print("--- Starting Bayesian Network Model Training ---")
        self.model_pipeline.fit(X_train, y_train)
        print("--- Model Berhasil Dilatih! ---")
    
    def evaluate(self):
        """Evaluates the model on the test set and prints the results."""
        if self.model_pipeline is None:
            print("Model has not been trained yet. Please run .train() first.")
            return

        y_pred = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=['Ham', 'Spam'])
        
        print("\n" + "="*40)
        print("Hasil Evaluasi Model (BAYESIAN NETWORK)")
        print("="*40)
        print(f"\nMODEL ACCURACY: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)

    def predict_interactive(self):
        """Runs an interactive loop for real-time predictions WITH TRANSLATION."""
        if self.model_pipeline is None:
            print("Model Belum Dilatih")
            return

        while True:
            input_text = input("\nMasukkan Pesan (ID/EN) (atau 'exit' untuk keluar): ")
            if input_text.lower() == 'exit':
                break
            
            if len(input_text.split()) < 2:
                print("Input terlalu pendek. Mohon masukkan kalimat lengkap.")
                continue

            try:
                detected = self.translator.detect(input_text)
                print(f"Bahasa terdeteksi: {LANGUAGES.get(detected.lang, detected.lang)}")
                
                text_to_predict = input_text
                if detected.lang != 'en':
                    print("Menerjemahkan ke Bahasa Inggris...")
                    translated = self.translator.translate(input_text, dest='en')
                    text_to_predict = translated.text
                    print(f"Hasil translasi: '{text_to_predict}'")
            
            except Exception as e:
                print(f"Gagal melakukan translasi: {e}")
                continue

            prediction = self.model_pipeline.predict([text_to_predict])
            prediction_proba = self.model_pipeline.predict_proba([text_to_predict])
            
            spam_prob = prediction_proba[0][1] * 100
            
            if prediction[0] == 1:
                print(f"➡️  Prediksi: SPAM ({spam_prob:.2f}% kemungkinan spam)")
            else:
                print(f"➡️  Prediksi: BUKAN SPAM (Ham) ({spam_prob:.2f}% kemungkinan spam)")
        
        print("\nSesi interaktif selesai.")

# Program Utama
if __name__ == "__main__":
    classifier_bn = SpamClassifierBN()
    classifier_bn.train('spam.csv')
    classifier_bn.predict_interactive()
    classifier_bn.evaluate()