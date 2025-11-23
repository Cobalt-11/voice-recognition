import os
import numpy as np
import joblib
import sys  # Import the sys module to access command-line arguments
from sklearn.preprocessing import StandardScaler
from single_file_features import extract_single_audio_features, N_MFCC  # Import the feature extraction function

def load_and_predict(audio_file_path):
    """
    Loads a .wav audio file, extracts features, loads pre-trained models,
    and makes gender and age predictions.

    Args:
        audio_file_path (str): The path to the .wav audio file.
    """
    print(f"Processing audio file: {audio_file_path}")
    features = extract_single_audio_features(audio_file_path)

    if features is None:
        print("Feature extraction failed. Cannot make predictions.")
        return

    # Load the pre-trained scaler
    try:
        scaler = joblib.load("scaler.pkl")
        expected_feature_length = (N_MFCC * 3) + 12
        if len(features) != expected_feature_length:
            print(f"Warning: Feature vector length mismatch. Expected {expected_feature_length}, got {len(features)}.")
            if len(features) < expected_feature_length:
                features = np.pad(features, (0, expected_feature_length - len(features)), mode='constant')
            else:
                features = features[:expected_feature_length]
        scaled_features = scaler.transform(features.reshape(1, -1))
    except FileNotFoundError:
        print("Error: 'scaler.pkl' not found. Make sure to train the model first.")
        return
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return

    # Load the pre-trained gender model
    try:
        gender_model = joblib.load("models/best_random_forest_gender.pkl")
        # Assuming the gender model was trained on the top 30 gender features
        top_30_gender_indices = np.load("top_30_feature_indices_gender.npy")
        scaled_features_gender = scaled_features[:, top_30_gender_indices]
        gender_prediction = gender_model.predict(scaled_features_gender)
        print("Predicted Gender:", gender_prediction)
    except FileNotFoundError:
        print("Error: 'models/gender_model.pkl' not found.")
    except Exception as e:
        print(f"Error loading gender model or making prediction: {e}")

    # Load the pre-trained age model
    try:
        age_model = joblib.load("models/best_random_forest_age.pkl")
        # Assuming the age model was trained on the top 30 age features
        top_30_age_indices = np.load("top_30_feature_indices_age.npy")
        scaled_features_age = scaled_features[:, top_30_age_indices]
        age_prediction = age_model.predict(scaled_features_age)
        print("Predicted Age:", age_prediction)
    except FileNotFoundError:
        print("Error: 'models/age_model.pkl' not found.")
    except Exception as e:
        print(f"Error loading age model or making prediction: {e}")

if __name__ == "__main__":
    test_samples_dir = "test_samples"

    if len(sys.argv) > 1:
        audio_file_name = sys.argv[1]
        audio_file_path = os.path.join(test_samples_dir, audio_file_name)
        if os.path.exists(audio_file_path) and audio_file_path.endswith(".mp3"):
            load_and_predict(audio_file_path)
        else:
            print(f"Error: Audio file '{audio_file_name}' not found in '{test_samples_dir}' or is not a .wav file.")
    else:
        audio_files = [f for f in os.listdir(test_samples_dir) if f.endswith(".wav")]
        if not audio_files:
            print(f"No .mp3 files found in '{test_samples_dir}'. Please add a .mp3 audio file for testing.")
        else:
            # Load the first .wav audio file found in the test_samples directory if no argument is provided
            first_audio_file = os.path.join(test_samples_dir, audio_files[0])
            print(f"No audio file specified in the command. Processing the first one found: {first_audio_file}")
            load_and_predict(first_audio_file)