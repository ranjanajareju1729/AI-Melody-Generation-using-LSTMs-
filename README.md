# AI-Melody-Generation-using-LSTMs-
A deep learning system for generating original melodies using LSTM networks. Transforms MIDI/Kern data into symbolic sequences and learns musical patterns.
## 🎹 Demo  
https://github.com/user-attachments/assets/bbf71f0f-39e5-4f1a-8ab0-4dbe15c70e49
## 📦 Installation  
```bash
git clone https://github.com/yourusername/AI-Melody-Generator.git
cd AI-Melody-Generator
pip install -r requirements.txt

🚀 Usage
Preprocess Data:
python src/preprocessing.py

Train Model:
python src/train.py

Generate Melodies:
from src.melody_generator import MelodyGenerator
mg = MelodyGenerator("models/melody_generator.keras")
melody = mg.generate_melody(seed="60 _ 62 _ 64", temperature=0.7)
mg.save_melody(melody, "my_melody.mid")

📚 Dataset
Source: Deutschl Ballad Corpus

Preprocessing:
Transposed to C major/A minor
Quantized to 0.25-beat steps
Encoded as symbolic sequences (e.g., ["60", "_", "62"])

🧠 Model Architecture
LSTM(
  Embedding(64) → LSTM(256) → Dropout(0.3) → LSTM(256) → Dense(vocab_size)
)
)

📊 Results
![Figure 2025-03-02 055549 (1)](https://github.com/user-attachments/assets/ea1ad2ec-0446-49a7-9687-3b4098f7e4b4)
🤝 Contributing
Pull requests welcome! For major changes, open an issue first.
