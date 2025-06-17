# Import Library
import pandas as pd
import re
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from googletrans import Translator, LANGUAGES 

# Definisi Kelas SpamClassifier dengan Bayesian Network 
class SpamClassifierBN_Advanced:
    def __init__(self):
        """Inisialisasi model, mesin inferensi, dan kata kunci fitur."""
        self.model = None
        self.inference_engine = None
        self.translator = Translator()
        self.feature_keywords = {
            'has_prize_word': ['prize', 'won', 'win', 'cash', 'reward', 'claim'],
            'has_urgent_word': ['urgent', 'now', 'immediately', 'hurry', 'action'],
            'has_action_word': ['call', 'txt', 'text', 'reply', 'stop', 'click'],
            'has_free_word': ['free', 'offer', 'promo', 'discount'],
            'has_money_symbol': ['£', '$', '€']
        }

    def _create_features(self, text):
        text = str(text).lower()
        features = {}
        for feature_name, keywords in self.feature_keywords.items():
            features[feature_name] = 1 if any(word in text for word in keywords) else 0
        return features

    def train(self, file_path):
        # Memuat dan Membersihkan Data Awal 
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Error: File '{file_path}' tidak ditemukan. Pastikan file ada di folder yang sama.")
            return

        df = df.rename(columns={"v1": "Category", "v2": "Message"})
        df = df[["Category", "Message"]]
        df.dropna(inplace=True)
        
        print("🚀 Membuat fitur level tinggi dari data...")
        features_df = df['Message'].apply(self._create_features).apply(pd.Series)
        
        training_data = pd.concat([features_df, df['Category']], axis=1)
        training_data.rename(columns={'Category': 'Spam'}, inplace=True)
        training_data['Spam'] = training_data['Spam'].map({'ham':0, 'spam':1})

        # Mendefinisikan STRUKTUR Jaringan Bayesian 
        self.model = BayesianNetwork([
            ('has_prize_word', 'has_action_word'),
            ('has_prize_word', 'Spam'),
            ('has_urgent_word', 'Spam'),
            ('has_action_word', 'Spam'),
            ('has_free_word', 'Spam'),
            ('has_money_symbol', 'Spam')
        ])
        
        print("🚀 Melatih model (Belajar Tabel Probabilitas dari Data)...")
        self.model.fit(training_data, estimator=BayesianEstimator, prior_type="BDeu")
        print("✅ Model berhasil dilatih!")

        self.inference_engine = VariableElimination(self.model)

    def predict_interactive(self):
        if self.model is None:
            print("Model belum dilatih. Jalankan metode .train() terlebih dahulu.")
            return

        while True:
            inp = input("\nMasukkan Pesan (ID/EN) untuk diperiksa (atau 'exit'): ")
            if inp.lower() == "exit":
                break

            if len(inp.strip()) < 3:
                print("Input terlalu pendek. Mohon masukkan kalimat yang lebih panjang.")
                continue

            try:
                detected = self.translator.detect(inp)
                print(f"Bahasa terdeteksi: {LANGUAGES.get(detected.lang, detected.lang)}")
                
                text_to_process = inp
                if detected.lang != 'en':
                    print("Menerjemahkan ke Bahasa Inggris...")
                    translated = self.translator.translate(inp, dest='en')
                    text_to_process = translated.text
                    print(f"Hasil translasi: '{text_to_process}'")
            except Exception as e:
                print(f"Gagal melakukan translasi: {e}")
                continue

            evidence_features = self._create_features(text_to_process)
            print(f"Fitur terdeteksi dari teks: {evidence_features}")

            try:
                # Lakukan inferensi pada fitur-fitur yang sudah dibuat
                prediction_dict = self.inference_engine.map_query(variables=['Spam'], evidence=evidence_features)
                prediction = prediction_dict['Spam']
                
                prob_dist = self.inference_engine.query(variables=['Spam'], evidence=evidence_features)
                spam_prob = prob_dist.get_value(Spam=1) * 100

                if prediction == 1:
                    print(f"➡️  Prediksi: SPAM ({spam_prob:.2f}% kemungkinan)")
                else:
                    print(f"➡️  Prediksi: BUKAN SPAM (Ham) ({spam_prob:.2f}% kemungkinan)")
            except Exception as e:
                print(f"Terjadi kesalahan saat melakukan prediksi: {e}")
                continue

        print("\n🛑 Sesi interaktif selesai.")

# Program Utama
if __name__ == "__main__":
    clf = SpamClassifierBN_Advanced()
    clf.train("spam.csv")
    clf.predict_interactive()