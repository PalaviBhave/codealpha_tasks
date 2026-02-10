"""
Emotion Prediction from Audio
Load trained model and predict emotions from new audio files
"""

import numpy as np
import pickle
import librosa
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Import AudioFeatureExtractor from feature_extraction
import sys
sys.path.append('.')
from feature_extraction import AudioFeatureExtractor

class EmotionPredictor:
    def __init__(self, model_path='models/best_model.h5',
                 scaler_path='models/scaler.pkl',
                 label_encoder_path='models/label_encoder.pkl',
                 feature_extractor_path='data/processed/feature_extractor.pkl'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to scaler
            label_encoder_path: Path to label encoder
            feature_extractor_path: Path to feature extractor
        """
        print("Loading model and artifacts...")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load feature extractor
        with open(feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        print("✓ Model and artifacts loaded successfully")
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Predicted emotion and confidence scores
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(audio_path)
        
        if features is None:
            return None, None, None
        
        # Normalize
        features = features.reshape(1, -1)
        features = self.scaler.transform(features)
        
        # Reshape for model input
        features = features.reshape(features.shape[0], features.shape[1], 1)
        
        # Predict
        prediction = self.model.predict(features, verbose=0)
        
        # Get emotion
        emotion_index = np.argmax(prediction[0])
        emotion = self.label_encoder.classes_[emotion_index]
        confidence = prediction[0][emotion_index]
        
        # Get all probabilities
        all_emotions = {}
        for i, em in enumerate(self.label_encoder.classes_):
            all_emotions[em] = float(prediction[0][i])
        
        return emotion, confidence, all_emotions
    
    def predict_batch(self, audio_files):
        """Predict emotions for multiple audio files"""
        results = []
        
        for audio_file in audio_files:
            emotion, confidence, all_probs = self.predict_emotion(audio_file)
            results.append({
                'file': audio_file,
                'emotion': emotion,
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        
        return results

def demo_prediction():
    """Demo prediction on sample files"""
    print("=" * 60)
    print("EMOTION PREDICTION DEMO")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = EmotionPredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease run 'python train_model.py' first to train the model")
        return
    
    # Get sample audio files
    audio_dir = Path('data/raw/sample_ravdess')
    audio_files = list(audio_dir.glob('*.wav'))[:10]  # First 10 files
    
    if len(audio_files) == 0:
        print("\n⚠ No audio files found for prediction")
        print(f"Please add .wav files to: {audio_dir}")
        return
    
    print(f"\nPredicting emotions for {len(audio_files)} files...\n")
    
    for audio_file in audio_files:
        emotion, confidence, all_probs = predictor.predict_emotion(str(audio_file))
        
        if emotion is not None:
            print(f"File: {audio_file.name}")
            print(f"Predicted Emotion: {emotion.upper()}")
            print(f"Confidence: {confidence:.2%}")
            print("All probabilities:")
            
            # Sort by probability
            sorted_probs = sorted(all_probs.items(), 
                                key=lambda x: x[1], reverse=True)
            for em, prob in sorted_probs:
                bar = '█' * int(prob * 30)
                print(f"  {em:12s}: {prob:.2%} {bar}")
            print("-" * 60)

def predict_custom_audio(audio_path):
    """Predict emotion from custom audio file"""
    predictor = EmotionPredictor()
    emotion, confidence, all_probs = predictor.predict_emotion(audio_path)
    
    if emotion is not None:
        print(f"\nPredicted Emotion: {emotion.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print("\nAll probabilities:")
        for em, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {em}: {prob:.2%}")
    else:
        print("Error: Could not process audio file")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Predict custom audio file
        audio_path = sys.argv[1]
        if Path(audio_path).exists():
            predict