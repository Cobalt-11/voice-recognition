import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from single_file_features import extract_single_audio_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== CASCADED MODEL LOADING ==========
model_dnn = keras.models.load_model("Models/dnn.keras")
cascaded_gender = joblib.load("Models/gender_model.pkl")
cascaded_age = joblib.load("Models/age_model_with_gender_input.pkl")
gender_encoder = joblib.load("Models/gender_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# ========== LABEL MAPPING ==========
LABEL_MAPPING = {
    'dnn': {
        0: ("Female", "Adult"),
        1: ("Female", "Kid"),
        2: ("Male", "Adult"),
        3: ("Male", "Kid")
    },
    'age': {
        0: "Kid", 1: "Adult",
        '0': "Kid", '1': "Adult",
        'kid': "Kid", 'adult': "Adult",
        'Kid': "Kid", 'Adult': "Adult"
    },
    'gender': {
        0: "Female", 1: "Male",
        '0': "Female", '1': "Male",
        'female': "Female", 'male': "Male",
        'Female': "Female", 'Male': "Male"
    }
}


# ========== AUDIO PROCESSING ==========
def preprocess_audio(audio_path):
    """Process audio specifically for DNN model"""
    y, sr = librosa.load(audio_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y, top_db=60)
    
    mfcc = librosa.feature.mfcc(
        y=y_trimmed,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )
    mfcc_flat = mfcc.flatten()
    
    processed_dnn = mfcc_flat[:51]
    if len(processed_dnn) < 51:
        processed_dnn = np.pad(processed_dnn, (0, 51 - len(processed_dnn)))
    
    return np.expand_dims(processed_dnn, axis=0)

# ========== FLASK ROUTES ==========
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify(error="No audio file uploaded"), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify(error="No file selected"), 400

    save_path = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # ========== DNN PREDICTION ==========
        dnn_input = preprocess_audio(save_path)
        dnn_probs = model_dnn.predict(dnn_input, verbose=0)[0]
        dnn_class = int(np.argmax(dnn_probs))
        dnn_gender, dnn_age = LABEL_MAPPING['dnn'][dnn_class]

        # ========== CASCADED PREDICTIONS ==========
        features = extract_single_audio_features(save_path)
        if features is None:
            raise ValueError("Feature extraction failed")
            
        # Scale features
        scaled = scaler.transform(features.reshape(1, -1))
        
        # Gender prediction
        gender_pred = cascaded_gender.predict(scaled)[0]
        gender_probs = cascaded_gender.predict_proba(scaled)[0]
        # Convert to native Python type
        if isinstance(gender_pred, np.generic):
            gender_pred = gender_pred.item()
        gender_pred_key = int(gender_pred) if isinstance(gender_pred, (int, np.integer)) else str(gender_pred)

        # Prepare age model input
        gender_encoded = gender_encoder.transform([[gender_pred]])
        age_input = np.hstack([scaled, gender_encoded])
        
        # Age prediction
        age_pred = cascaded_age.predict(age_input)[0]
        age_probs = cascaded_age.predict_proba(age_input)[0]
        if isinstance(age_pred, np.generic):
            age_pred = age_pred.item()
        age_pred_key = int(age_pred) if isinstance(age_pred, (int, np.integer)) else str(age_pred)

        # Prepare probabilities as floats
        gender_probs_dict = {
            "Female": float(gender_probs[0]),
            "Male": float(gender_probs[1])
        }
        age_probs_dict = {
            "Kid": float(age_probs[0]),
            "Adult": float(age_probs[1])
        }

        return jsonify(
            dnn={
                'gender': dnn_gender,
                'age': dnn_age,
                'confidence': float(np.max(dnn_probs))
            },
            cascaded_gender={
                'prediction': LABEL_MAPPING['gender'][gender_pred_key],
                'probabilities': gender_probs_dict
            },
            cascaded_age={
                'prediction': LABEL_MAPPING['age'][age_pred_key],
                'probabilities': age_probs_dict
            }
        )

    except Exception as e:
        return jsonify(error=f"Prediction failed: {str(e)}"), 500
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
