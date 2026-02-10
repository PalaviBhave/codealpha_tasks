"""
Emotion Recognition - Prediction Script
"""

import numpy as np
import pickle
import librosa
from pathlib import Path
from tensorflow import keras
import warnings
import sys
warnings.filterwarnings('ignore')

print("=" * 70)
print("EMOTION RECOGNITION - PREDICTION SYSTEM")
print("=" * 70)
sys.stdout.flush()

# Extract features function
def extract_audio_features(audio_path, n_mfcc=40):
    """Extract features from audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, duration=3.0, sr=22050)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Additional features
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))
        
        # Combine
        features = np.concatenate([
            mfccs_mean, 
            mfccs_std, 
            [zcr, spectral_centroid, spectral_rolloff, chroma]
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Load model and artifacts
print("\nStep 1: Loading model and artifacts...")
sys.stdout.flush()
try:
    model = keras.models.load_model('models/best_model.h5')
    print("  ✓ Model loaded")
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("  ✓ Scaler loaded")
    
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("  ✓ Label encoder loaded")
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n❌ Error loading files: {e}")
    print("Please ensure you've run 'python train_model.py' first")
    sys.exit(1)

# Get audio files
print("\nStep 2: Finding audio files...")
sys.stdout.flush()
audio_dir = Path('data/raw/sample_ravdess')
audio_files = list(audio_dir.glob('*.wav'))

if len(audio_files) == 0:
    print(f"❌ No audio files found in {audio_dir}")
    sys.exit(1)

print(f"  ✓ Found {len(audio_files)} audio files")
sys.stdout.flush()

# Select first 10 for demo
demo_files = audio_files[:10]

print(f"\nStep 3: Predicting emotions for {len(demo_files)} files...")
print("=" * 70)
sys.stdout.flush()

# Predict each file
for i, audio_file in enumerate(demo_files, 1):
    print(f"\n[{i}/{len(demo_files)}] {audio_file.name}")
    sys.stdout.flush()
    
    # Extract features
    features = extract_audio_features(str(audio_file))
    
    if features is None:
        print("  ❌ Failed to extract features")
        sys.stdout.flush()
        continue
    
    # Prepare for model
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape(1, -1, 1)
    
    # Predict
    prediction = model.predict(features_scaled, verbose=0)
    
    # Get results
    emotion_idx = np.argmax(prediction[0])
    emotion = label_encoder.classes_[emotion_idx]
    confidence = prediction[0][emotion_idx]
    
    # Display results
    print(f"  🎭 Predicted Emotion: {emotion.upper()}")
    print(f"  📊 Confidence: {confidence*100:.2f}%")
    print(f"  📈 All Probabilities:")
    sys.stdout.flush()
    
    all_probs = list(zip(label_encoder.classes_, prediction[0]))
    all_probs.sort(key=lambda x: x[1], reverse=True)
    
    for emotion_name, prob in all_probs:
        bar_length = int(prob * 40)
        bar = '█' * bar_length
        print(f"     {emotion_name:12s}: {prob*100:5.2f}% {bar}")
    
    sys.stdout.flush()

print("\n" + "=" * 70)
print("PREDICTION COMPLETE!")
print("=" * 70)
sys.stdout.flush()