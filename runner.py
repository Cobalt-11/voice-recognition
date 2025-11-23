import os
import sys
import warnings
import logging
import csv
import numpy as np
import joblib
import librosa
from tensorflow import keras
from single_file_features import extract_single_audio_features, N_MFCC

# ========== CONFIGURATION ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

base_dir = os.path.expanduser("~/Documents/voice-recognition")
model_path = os.path.join(base_dir, "Models/dnn.keras")
audio_dir = os.path.join(base_dir, "test_audio")
csv_output_path = os.path.join(base_dir, "predictions.csv")

# ========== MODEL LOADING ==========
def load_models():
    """Load all models and preprocessing artifacts"""
    models = {
        'dnn': keras.models.load_model(model_path),
        'gender_rf': joblib.load("Models/best_random_forest_gender.pkl"),
        'age_rf': joblib.load("Models/best_random_forest_age.pkl"),
        'scaler': joblib.load("scaler.pkl"),
        'gender_features': np.load("top_30_feature_indices_gender.npy"),
        'age_features': np.load("top_30_feature_indices_age.npy"),
        # Cascaded models
        'cascaded_gender': joblib.load("Models/gender_model.pkl"),
        'cascaded_age': joblib.load("Models/age_model_with_gender_input.pkl"),
        'gender_encoder': joblib.load("Models/gender_encoder.pkl")
    }
    print(f"[INFO] DNN input shape: {models['dnn'].input_shape}")
    return models

# ========== AUDIO PROCESSING ==========
def preprocess_audio(audio_path):
    """Process audio for DNN model"""
    y, sr = librosa.load(audio_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    processed = mfcc.flatten()[:51]
    if len(processed) < 51:
        processed = np.pad(processed, (0, 51 - len(processed)))
    return np.expand_dims(processed, axis=0)

# ========== PREDICTION FUNCTIONS ==========
def predict_rf_models(models, features):
    """Get predictions from RF models"""
    scaled = models['scaler'].transform(features.reshape(1, -1))
    # Gender prediction
    gender_features = scaled[:, models['gender_features']]
    gender_pred = models['gender_rf'].predict(gender_features)[0]
    # Age prediction
    age_features = scaled[:, models['age_features']]
    age_pred = models['age_rf'].predict(age_features)[0]
    return gender_pred, age_pred

def predict_cascaded(models, features):
    """Predict using cascaded gender→age approach"""
    scaled = models['scaler'].transform(features.reshape(1, -1))
    # Predict gender
    gender_pred = models['cascaded_gender'].predict(scaled)
    # Encode gender prediction
    gender_encoded = models['gender_encoder'].transform(np.array(gender_pred).reshape(-1, 1))
    # Prepare age model input
    age_input = np.hstack([scaled, gender_encoded])
    # Predict age
    age_pred = models['cascaded_age'].predict(age_input)
    return gender_pred[0], age_pred[0]

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    try:
        models = load_models()
    except Exception as e:
        print(f"[ERROR] Model loading failed: {str(e)}")
        sys.exit(1)

    # Class labels for DNN (must match your training order)
    dnn_class_labels = ["Female Adult", "Female Kid", "Male Adult", "Male Kid"]

    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'audio_filename', 
            'dnn_prediction',
            'dnn_confidence',
            'rf_gender',
            'rf_age',
            'cascaded_gender',
            'cascaded_age',
            'cascaded_combined'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]

        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            print(f"\n[INFO] Processing: {audio_file}")

            try:
                # DNN prediction
                dnn_input = preprocess_audio(audio_path)
                dnn_probs = models['dnn'].predict(dnn_input, verbose=0)[0]
                dnn_class = dnn_class_labels[np.argmax(dnn_probs)]
                dnn_confidence = np.max(dnn_probs)

                # Feature extraction for RF models
                features = extract_single_audio_features(audio_path)
                if features is None:
                    raise ValueError("Feature extraction failed")

                # RF predictions
                gender_pred, age_pred = predict_rf_models(models, features)

                # Cascaded predictions
                cascaded_gender, cascaded_age = predict_cascaded(models, features)
                cascaded_combined = f"{cascaded_gender}_{cascaded_age}"

                # Write results
                writer.writerow({
                    'audio_filename': audio_file,
                    'dnn_prediction': dnn_class,
                    'dnn_confidence': f"{dnn_confidence:.2f}",
                    'rf_gender': gender_pred,
                    'rf_age': age_pred,
                    'cascaded_gender': cascaded_gender,
                    'cascaded_age': cascaded_age,
                    'cascaded_combined': cascaded_combined
                })

                print(f"[RESULT] DNN: {dnn_class} ({dnn_confidence:.2f})")
                print(f"[RESULT] RF: {gender_pred}/{age_pred}")
                print(f"[RESULT] Cascaded: {cascaded_combined}")

            except Exception as e:
                print(f"[ERROR] Processing failed: {str(e)}")
                writer.writerow({
                    'audio_filename': audio_file,
                    'dnn_prediction': 'error',
                    'dnn_confidence': 'error',
                    'rf_gender': 'error',
                    'rf_age': 'error',
                    'cascaded_gender': 'error',
                    'cascaded_age': 'error',
                    'cascaded_combined': 'error'
                })

    print(f"\n[SUCCESS] Predictions saved to {csv_output_path}")
