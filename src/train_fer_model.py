# src/train_fer_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "FER" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "FER" / "test"
MODELS_DIR = PROJECT_ROOT / "models"

# Force the exact class order from your description
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def build_fer_cnn():
    """Builds a Convolutional Neural Network for 48x48 Grayscale Images"""
    model = keras.Sequential([
        layers.Input(shape=(48, 48, 1)),
        # Block 1
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (5,5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),

        # Flatten & Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax') # 7 emotion classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_emotion_model():
    print("\n--- Training FER-2013 Image Emotion Model ---")
    
    # 1. Load Image Datasets directly from folders
    # Grayscale, 48x48 size, batches of 64
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels='inferred', class_names=CLASS_NAMES, 
        color_mode='grayscale', image_size=(48, 48), batch_size=64
    )
    
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, labels='inferred', class_names=CLASS_NAMES, 
        color_mode='grayscale', image_size=(48, 48), batch_size=64
    )

    # Normalize pixel values from 0-255 to 0-1 for the neural network
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # 2. Build and Train
    model = build_fer_cnn()
    
    # Train for 30 epochs (FER dataset requires more epochs to learn faces well)
    print("Beginning training... This might take a while on CPU.")
    model.fit(train_ds, validation_data=test_ds, epochs=30)
    
    # 3. Save as .keras file replacing the old tabular one
    model.save(MODELS_DIR / 'emotion_classifier.keras')
    print("\n✓ Image Emotion Model Trained and Saved successfully as 'emotion_classifier.keras'.")

if __name__ == "__main__":
    train_emotion_model()