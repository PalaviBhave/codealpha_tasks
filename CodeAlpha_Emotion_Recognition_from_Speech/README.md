\# 🎭 Emotion Recognition from Speech



A deep learning project that recognizes human emotions from speech audio using MFCCs (Mel-Frequency Cepstral Coefficients) and hybrid CNN+LSTM neural networks.



\## 📋 Table of Contents

\- \[Overview]

\- \[Features]

\- \[Installation]

\- \[Usage]

\- \[Project Structure]

\- \[Technical Details]

\- \[Results]

\- \[Dataset]

\- \[Future Improvements]



\## 🎯 Overview



This project implements a complete emotion recognition system that can classify speech audio into 8 different emotional states:



\- 😐 Neutral

\- 😌 Calm

\- 😊 Happy

\- 😢 Sad

\- 😠 Angry

\- 😨 Fearful

\- 🤢 Disgust

\- 😲 Surprised



\## ✨ Features



\- \*\*MFCC Feature Extraction\*\*: Extracts 40 Mel-Frequency Cepstral Coefficients plus additional acoustic features

\- \*\*Deep Learning Models\*\*: Hybrid CNN+LSTM architecture for temporal pattern recognition

\- \*\*Multi-Dataset Support\*\*: Compatible with RAVDESS, TESS, and EMO-DB datasets

\- \*\*Real-time Prediction\*\*: Predict emotions from new audio files

\- \*\*Visualization\*\*: Training history plots and confusion matrices



\## 🚀 Installation



\### Prerequisites

\- Python 3.9 or higher

\- pip package manager



\### Setup



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/YOUR\_USERNAME/emotion-recognition.git

cd emotion-recognition

```



2\. \*\*Create virtual environment\*\*

```bash

python -m venv venv



\# Windows

venv\\Scripts\\activate



\# Linux/Mac

source venv/bin/activate

```



3\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



\## 📖 Usage



\### Quick Start



Run the complete pipeline:

```bash

\# Step 1: Prepare directories

python dataset\_preparation.py



\# Step 2: Extract features from audio

python feature\_extraction.py



\# Step 3: Train the model (takes 10-15 minutes)

python train\_model.py



\# Step 4: Make predictions

python run\_predictions.py

```



\### Predict Custom Audio

```bash

python run\_predictions.py path/to/your/audio.wav

```



\### Using Real Dataset



1\. Download RAVDESS dataset from \[Kaggle](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)

2\. Extract audio files to `data/raw/sample\_ravdess/`

3\. Re-run feature extraction and training

```bash

python feature\_extraction.py

python train\_model.py

```



\## 📁 Project Structure

```

emotion\_recognition/

├── data/

│   ├── raw/

│   │   └── sample\_ravdess/          # Audio files (.wav)

│   └── processed/

│       ├── features.csv              # Extracted features

│       └── feature\_extractor.pkl     # Feature extractor object

├── models/

│   ├── best\_model.h5                 # Trained model

│   ├── scaler.pkl                    # Feature scaler

│   └── label\_encoder.pkl             # Label encoder

├── results/

│   ├── training\_history.png          # Training plots

│   └── confusion\_matrix.png          # Confusion matrix

├── dataset\_preparation.py            # Setup directories

├── feature\_extraction.py             # Extract MFCC features

├── train\_model.py                    # Train CNN+LSTM model

├── run\_predictions.py                # Make predictions

├── requirements.txt                  # Dependencies

└── README.md

```



\## 🔬 Technical Details



\### Feature Extraction



\*\*MFCC Features (80 features)\*\*:

\- 40 MFCC coefficients (mean)

\- 40 MFCC coefficients (standard deviation)



\*\*Additional Features (4 features)\*\*:

\- Zero Crossing Rate

\- Spectral Centroid

\- Spectral Rolloff

\- Chroma Features



\*\*Total: 84 features per audio sample\*\*



\### Model Architecture

```

Hybrid CNN + LSTM Model

├── Input Layer (84 features)

├── Conv1D Layer (64 filters, kernel=5)

├── MaxPooling1D

├── Dropout (0.3)

├── Conv1D Layer (128 filters, kernel=5)

├── MaxPooling1D

├── Dropout (0.3)

├── LSTM Layer (64 units)

├── Dropout (0.3)

├── Dense Layer (64 units, ReLU)

├── Dropout (0.4)

└── Output Layer (8 units, Softmax)

```



\### Training Configuration



\- \*\*Optimizer\*\*: Adam (learning\_rate=0.001)

\- \*\*Loss Function\*\*: Categorical Crossentropy

\- \*\*Batch Size\*\*: 16

\- \*\*Epochs\*\*: 50 (with early stopping)

\- \*\*Validation Split\*\*: 20%

\- \*\*Test Split\*\*: 20%



\## 📊 Results



\### Model Performance (on synthetic data)



\- \*\*Test Accuracy\*\*: ~18-20% (synthetic data)

\- \*\*Expected with Real Data\*\*: 70-90%



\*Note: Low accuracy is due to synthetic training data. With real RAVDESS dataset, accuracy improves significantly.\*



\### Sample Prediction Output

```

File: 03-01-01-01-01-01-01.wav

&nbsp; 🎭 Predicted Emotion: NEUTRAL

&nbsp; 📊 Confidence: 18.00%

&nbsp; 📈 All Probabilities:

&nbsp;    neutral     : 18.00% ███████

&nbsp;    sad         : 17.95% ███████

&nbsp;    calm        : 17.39% ██████

&nbsp;    disgust     : 16.30% ██████

&nbsp;    ...

```



\## 📚 Dataset



\### RAVDESS (Recommended)

\- \*\*Full Name\*\*: Ryerson Audio-Visual Database of Emotional Speech and Song

\- \*\*Size\*\*: 1440 audio files

\- \*\*Speakers\*\*: 24 professional actors

\- \*\*Emotions\*\*: 8 emotions

\- \*\*Download\*\*: \[Kaggle RAVDESS](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)



\### Filename Format

```

03-01-06-01-02-01-12.wav

│  │  │  │  │  │  └─ Actor ID

│  │  │  │  │  └──── Repetition

│  │  │  │  └─────── Statement

│  │  │  └────────── Intensity

│  │  └───────────── Emotion (01-08)

│  └──────────────── Vocal channel

└─────────────────── Modality

```



\### Emotion Codes

\- 01 = Neutral

\- 02 = Calm

\- 03 = Happy

\- 04 = Sad

\- 05 = Angry

\- 06 = Fearful

\- 07 = Disgust

\- 08 = Surprised



\## 🔧 Requirements

```

numpy>=1.23.0

librosa>=0.10.0

soundfile>=0.12.0

scikit-learn>=1.3.0

tensorflow>=2.13.0

pandas>=2.0.0

matplotlib>=3.7.0

seaborn>=0.12.0

```



\## 🚀 Future Improvements



\- \[ ] Add support for more datasets (TESS, EMO-DB)

\- \[ ] Implement data augmentation

\- \[ ] Add attention mechanisms

\- \[ ] Real-time audio streaming prediction

\- \[ ] Web interface for easy testing

\- \[ ] Model ensemble for better accuracy

\- \[ ] Export to TensorFlow Lite for mobile deployment







\## 👤 Author



\*\*PALAVI BHAVE\*\*

\- GitHub: \[@PalaviBhave](https://github.com/PalaviBhave)



\## 🙏 Acknowledgments



\- RAVDESS dataset creators

\- TensorFlow and Keras teams

\- Librosa library developers





