# src/train_new_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *
from models import build_emotion_classifier, build_cognitive_load_classifier

def train_emotion_model():
    print("\n--- Training Emotion Classifier (VREED) ---")
    try:
        df = pd.read_csv(VREED_DATA_PATH)
        print(f"Data loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find VREED data at {VREED_DATA_PATH}")
        print("Please ensure you extracted the '04 Eye Tracking Data' folder into data/raw/vreed/")
        return

    # In VREED, 'Quad_Cat' is the target column
    if 'Quad_Cat' in df.columns:
        X = df.drop(columns=['Quad_Cat']).values
        y = df['Quad_Cat'].values
    else:
        # Fallback: assume the first column is the target as per standard VREED format
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

    # Clean data: fill NaNs with 0 if any exist
    X = np.nan_to_num(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_emotion_classifier(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    model.save(MODELS_DIR / 'emotion_classifier.h5')
    print("✓ Emotion Model Saved successfully.")


def train_cognitive_load_model():
    print("\n--- Training Cognitive Load Classifier ---")
    try:
        df = pd.read_csv(COGLOAD_DATA_PATH)
        print(f"Data loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find Cognitive Load data at {COGLOAD_DATA_PATH}")
        return

    # FIX: Dynamically separate features and target using the LAST column as the target
    # This prevents the KeyError: "['Target Variable'] not found in axis"
    target_col_name = df.columns[-1]
    print(f"Auto-detected Target Column name: '{target_col_name}'")
    
    X = df.drop(columns=[target_col_name]).values
    y = df[target_col_name].values

    # Clean data: fill NaNs with 0 if any exist
    X = np.nan_to_num(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_cognitive_load_classifier(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    model.save(MODELS_DIR / 'cognitive_load_classifier.h5')
    print("✓ Cognitive Load Model Saved successfully.")

if __name__ == "__main__":
    train_emotion_model()
    train_cognitive_load_model()