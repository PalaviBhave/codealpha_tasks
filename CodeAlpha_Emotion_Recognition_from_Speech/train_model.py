"""
Train Emotion Recognition Model
Uses CNN, RNN, and LSTM for emotion classification
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class EmotionRecognitionModel:
    def __init__(self, input_shape, num_classes):
        """
        Initialize model
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of emotion classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_cnn_model(self):
        """Build CNN model"""
        model = models.Sequential([
            layers.Input(shape=(self.input_shape, 1)),
            
            # Conv Block 1
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Conv Block 2
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Conv Block 3
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_lstm_model(self):
        """Build LSTM model"""
        model = models.Sequential([
            layers.Input(shape=(self.input_shape, 1)),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_hybrid_model(self):
        """Build CNN + LSTM hybrid model"""
        model = models.Sequential([
            layers.Input(shape=(self.input_shape, 1)),
            
            # CNN layers
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        return self.history

def load_and_prepare_data(data_path='data/processed/features.csv'):
    """Load and prepare data for training"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Remove filename column
    if 'filename' in df.columns:
        df = df.drop('filename', axis=1)
    
    # Separate features and labels
    X = df.drop('emotion', axis=1).values
    y = df['emotion'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    
    return X, y_encoded, label_encoder

def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()

def main():
    print("=" * 60)
    print("EMOTION RECOGNITION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    X, y, label_encoder = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape for CNN/LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # One-hot encode labels
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Build model
    print("\n" + "=" * 60)
    print("Building model...")
    model_builder = EmotionRecognitionModel(X_train.shape[1], num_classes)
    
    # Choose model type (you can change this)
    print("Model type: Hybrid CNN + LSTM")
    model_builder.build_hybrid_model()
    
    # Compile
    model_builder.compile_model(learning_rate=0.001)
    
    # Print model summary
    model_builder.model.summary()
    
    # Train
    print("\n" + "=" * 60)
    print("Training model...")
    print("=" * 60)
    
    history = model_builder.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating model...")
    print("=" * 60)
    
    # Load best model
    best_model = keras.models.load_model('models/best_model.h5')
    
    # Predict
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                                target_names=label_encoder.classes_))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred_classes, label_encoder.classes_)
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving artifacts...")
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✓ Label encoder saved")
    print("✓ Scaler saved")
    print("✓ Best model saved")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run 'python predict.py' to make predictions")

if __name__ == "__main__":
    main()