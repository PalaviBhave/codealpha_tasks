"""
Test Single Prediction - Diagnostic
"""

import numpy as np
import pickle
import librosa
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("Testing single prediction...\n")

# Load model
print("Loading model...")
model = keras.models.load_model('models/best_model.h5')
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
print("✓ Loaded\n")

# Test file
audio_file = 'data/raw/sample_ravdess/03-01-01-01-01-01-01.wav'
print(f"Processing: {audio_file}")

# Extract features
print("Extracting features...")
audio, sr = librosa.load(audio_file, duration=3.0, sr=22050)
print(f"  Audio shape: {audio.shape}")
print(f"  Sample rate: {sr}")

mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
print(f"  MFCC shape: {mfccs.shape}")

mfccs_mean = np.mean(mfccs.T, axis=0)
mfccs_std = np.std(mfccs.T, axis=0)

zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))

features = np.concatenate([mfccs_mean, mfccs_std, [zcr, spectral_centroid, spectral_rolloff, chroma]])
print(f"  Feature shape: {features.shape}")

# Scale
features = scaler.transform(features.reshape(1, -1))
features = features.reshape(1, -1, 1)
print(f"  Scaled shape: {features.shape}")

# Predict
print("\nPredicting...")
prediction = model.predict(features, verbose=0)
print(f"  Prediction shape: {prediction.shape}")
print(f"  Raw prediction: {prediction[0]}")

emotion_idx = np.argmax(prediction[0])
emotion = encoder.classes_[emotion_idx]
confidence = prediction[0][emotion_idx]

print(f"\n✓ Predicted Emotion: {emotion.upper()}")
print(f"✓ Confidence: {confidence*100:.2f}%")

print("\nAll probabilities:")
for i, em in enumerate(encoder.classes_):
    print(f"  {em:12s}: {prediction[0][i]*100:6.2f}%")