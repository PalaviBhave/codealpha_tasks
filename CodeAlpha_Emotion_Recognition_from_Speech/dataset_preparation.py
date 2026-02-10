"""
Dataset Preparation for Emotion Recognition
Downloads RAVDESS dataset and organizes it
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'data/raw', 'data/processed', 'models', 'results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created successfully")

def download_ravdess_sample():
    """
    Creates a sample dataset structure similar to RAVDESS
    Since automatic download requires Kaggle API setup,
    this creates sample data for demonstration
    """
    print("\nNote: For full RAVDESS dataset, download from:")
    print("https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio")
    print("\nCreating sample dataset structure...")
    
    # RAVDESS filename format: 03-01-06-01-02-01-12.wav
    # Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor
    
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    sample_dir = Path('data/raw/sample_ravdess')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mapping file for users to add their own audio files
    mapping_file = sample_dir / 'INSTRUCTIONS.txt'
    with open(mapping_file, 'w') as f:
        f.write("RAVDESS Dataset Instructions\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. Download RAVDESS dataset from Kaggle\n")
        f.write("2. Extract audio files to this directory\n")
        f.write("3. Filename format: 03-01-XX-01-02-01-12.wav\n")
        f.write("   where XX is the emotion code:\n\n")
        for code, emotion in emotions.items():
            f.write(f"   {code}: {emotion}\n")
        f.write("\nExample filenames:\n")
        f.write("   03-01-03-01-01-01-01.wav (happy)\n")
        f.write("   03-01-05-01-01-01-01.wav (angry)\n")
        f.write("   03-01-04-01-01-01-01.wav (sad)\n")
    
    print(f"✓ Sample structure created at: {sample_dir}")
    print(f"✓ See {mapping_file} for instructions")
    
    return emotions

def prepare_dataset_info():
    """Display dataset information"""
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION DATASET SETUP")
    print("=" * 60)
    print("\nSupported Datasets:")
    print("1. RAVDESS (Recommended)")
    print("2. TESS")
    print("3. EMO-DB")
    print("\nThis project uses RAVDESS format.")
    print("\nEmotion Labels:")
    emotions = {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgust': 6,
        'surprised': 7
    }
    for emotion, code in emotions.items():
        print(f"  {code}: {emotion}")
    
    return emotions

if __name__ == "__main__":
    print("Starting dataset preparation...\n")
    create_directories()
    emotions_map = download_ravdess_sample()
    prepare_dataset_info()
    print("\n✓ Dataset preparation complete!")
    print("\nNext steps:")
    print("1. Add your audio files to data/raw/sample_ravdess/")
    print("2. Run: python feature_extraction.py")