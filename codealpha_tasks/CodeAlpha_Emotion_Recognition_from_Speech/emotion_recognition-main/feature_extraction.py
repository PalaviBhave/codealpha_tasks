"""
Feature Extraction for Emotion Recognition
Extracts MFCCs and other audio features from speech files
"""

import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, n_mfcc=40, n_fft=2048, hop_length=512):
        """
        Initialize feature extractor
        
        Args:
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, audio_path, duration=3.0, sr=22050):
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            duration: Duration to process (seconds)
            sr: Sample rate
            
        Returns:
            MFCC features (flattened)
        """
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_path, duration=duration, sr=sr)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, 
                                         n_mfcc=self.n_mfcc,
                                         n_fft=self.n_fft, 
                                         hop_length=self.hop_length)
            
            # Calculate statistics for each MFCC coefficient
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.std(mfccs.T, axis=0)
            
            # Combine mean and std
            features = np.concatenate([mfccs_mean, mfccs_std])
            
            return features
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_additional_features(self, audio_path, duration=3.0, sr=22050):
        """
        Extract additional audio features
        
        Features:
        - Zero Crossing Rate
        - Spectral Centroid
        - Spectral Rolloff
        - Chroma features
        """
        try:
            audio, sample_rate = librosa.load(audio_path, duration=duration, sr=sr)
            
            # Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral Centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            # Spectral Rolloff
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
            
            # Chroma Features
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate))
            
            return np.array([zcr, spectral_centroid, spectral_rolloff, chroma])
        
        except Exception as e:
            print(f"Error extracting additional features from {audio_path}: {e}")
            return None
    
    def extract_all_features(self, audio_path):
        """Extract all features combined"""
        mfcc_features = self.extract_mfcc(audio_path)
        additional_features = self.extract_additional_features(audio_path)
        
        if mfcc_features is not None and additional_features is not None:
            return np.concatenate([mfcc_features, additional_features])
        return None

def get_emotion_from_filename(filename):
    """
    Extract emotion from RAVDESS filename
    Format: 03-01-XX-01-02-01-12.wav
    XX is emotion code (01-08)
    """
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    try:
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return emotion_map.get(emotion_code, 'unknown')
    except:
        pass
    
    return 'unknown'

def process_dataset(data_dir='data/raw/sample_ravdess', output_dir='data/processed'):
    """
    Process all audio files in dataset
    
    Args:
        data_dir: Directory containing audio files
        output_dir: Directory to save processed features
    """
    print("Starting feature extraction...\n")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(n_mfcc=40)
    
    # Get all audio files
    audio_files = list(Path(data_dir).glob('*.wav'))
    
    if len(audio_files) == 0:
        print("⚠ No audio files found!")
        print(f"Please add .wav files to: {data_dir}")
        print("\nCreating sample dataset for demonstration...")
        create_sample_data()
        audio_files = list(Path(data_dir).glob('*.wav'))
    
    features_list = []
    labels_list = []
    filenames_list = []
    
    print(f"Processing {len(audio_files)} audio files...\n")
    
    for i, audio_file in enumerate(audio_files, 1):
        # Extract features
        features = extractor.extract_all_features(str(audio_file))
        
        if features is not None:
            # Get emotion label
            emotion = get_emotion_from_filename(audio_file.name)
            
            features_list.append(features)
            labels_list.append(emotion)
            filenames_list.append(audio_file.name)
            
            if i % 20 == 0 or i == len(audio_files):
                print(f"✓ Processed {i}/{len(audio_files)} files")
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    df['emotion'] = labels_list
    df['filename'] = filenames_list
    
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path / 'features.csv', index=False)
    
    # Save feature extractor
    with open(output_path / 'feature_extractor.pkl', 'wb') as f:
        pickle.dump(extractor, f)
    
    print(f"\n✓ Features saved to: {output_path / 'features.csv'}")
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Feature dimensions: {len(df.columns) - 2}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    return df

def create_sample_data():
    """Create sample synthetic audio files for demonstration"""
    print("Creating synthetic sample audio files...")
    
    import soundfile as sf
    
    sample_dir = Path('data/raw/sample_ravdess')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Emotion codes
    emotions = ['01', '02', '03', '04', '05', '06', '07', '08']
    emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    sr = 22050
    duration = 3
    
    # Create 15 samples per emotion (total 120 samples)
    for i, emotion_code in enumerate(emotions):
        emotion_name = emotion_names[i]
        print(f"Creating samples for {emotion_name}...")
        
        for rep in range(1, 16):  # 15 samples per emotion
            filename = f"03-01-{emotion_code}-01-01-{rep:02d}-01.wav"
            filepath = sample_dir / filename
            
            # Generate audio with different characteristics per emotion
            t = np.linspace(0, duration, sr * duration)
            
            # Base noise
            audio = np.random.randn(sr * duration) * 0.05
            
            # Add emotion-specific frequency patterns
            if emotion_code == '01':  # neutral
                audio += 0.1 * np.sin(2 * np.pi * 150 * t)
            elif emotion_code == '02':  # calm
                audio += 0.08 * np.sin(2 * np.pi * 120 * t)
            elif emotion_code == '03':  # happy
                audio += 0.12 * np.sin(2 * np.pi * 250 * t)
                audio += 0.08 * np.sin(2 * np.pi * 500 * t)
            elif emotion_code == '04':  # sad
                audio += 0.09 * np.sin(2 * np.pi * 100 * t)
            elif emotion_code == '05':  # angry
                audio += 0.15 * np.sin(2 * np.pi * 300 * t)
                audio += 0.1 * np.sin(2 * np.pi * 600 * t)
            elif emotion_code == '06':  # fearful
                audio += 0.11 * np.sin(2 * np.pi * 280 * t)
                audio += 0.07 * np.sin(2 * np.pi * 450 * t)
            elif emotion_code == '07':  # disgust
                audio += 0.1 * np.sin(2 * np.pi * 180 * t)
            elif emotion_code == '08':  # surprised
                audio += 0.13 * np.sin(2 * np.pi * 320 * t)
            
            # Add some variation for each sample
            audio += 0.03 * np.random.randn(sr * duration)
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            sf.write(str(filepath), audio, sr)
    
    print(f"✓ Created {len(emotions) * 15} sample audio files (120 total)")
    print("Note: These are synthetic samples for demonstration.")
    print("Replace with real RAVDESS audio files for actual training.")

if __name__ == "__main__":
    # Process dataset
    df = process_dataset()
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print("=" * 60)
    print("\nNext step: Run 'python train_model.py' to train the model")