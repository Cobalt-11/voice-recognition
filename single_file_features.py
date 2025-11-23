import os
import numpy as np
import librosa
import parselmouth
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.preprocessing import StandardScaler
import joblib

N_MFCC = 13

def extract_formants(file_path):
    """Extracts the first three formants from an audio file."""
    try:
        sound = parselmouth.Sound(file_path)
        formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=3, maximum_formant=4000)
        f1 = np.nanmean([formants.get_value_at_time(1, t) for t in np.arange(0.0, sound.duration, 0.02)])
        f2 = np.nanmean([formants.get_value_at_time(2, t) for t in np.arange(0.0, sound.duration, 0.02)])
        f3 = np.nanmean([formants.get_value_at_time(3, t) for t in np.arange(0.0, sound.duration, 0.02)])
        return f1, f2, f3
    except Exception as e:
        print(f"Error extracting formants from {file_path}: {e}")
        return 0, 0, 0

def extract_jitter_shimmer(file_path):
    """Extracts jitter and shimmer from an audio file."""
    try:
        [Fs, x] = audioBasicIO.read_audio_file(file_path)
        F, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
        jitter = np.mean(F[6, :])
        shimmer = np.mean(F[7, :])
        return jitter, shimmer
    except Exception as e:
        print(f"Error extracting jitter/shimmer from {file_path}: {e}")
        return 0, 0

def extract_hnr(file_path):
    """Extracts Harmonic-to-Noise Ratio (HNR) from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        hnr = np.mean(harmonic) / np.mean(noise) if np.mean(noise) != 0 else 0
        return hnr
    except Exception as e:
        print(f"Error extracting HNR from {file_path}: {e}")
        return 0

def extract_single_audio_features(file_path):

    try:
        y, sr = librosa.load(file_path, sr=None)

        # Trim silence
        y, _ = librosa.effects.trim(y)

        # MFCC and derivatives
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        mfccs_mean = np.mean(mfccs, axis=1)
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta2_mean = np.mean(delta2_mfccs, axis=1)

        # Pitch (pyin)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]
        pitch_mean = np.mean(f0) if f0.size > 0 else 0
        pitch_std = np.std(f0) if f0.size > 0 else 0

        # Energy
        energy = np.array([
            np.sum(np.square(frame))
            for frame in librosa.util.frame(y, frame_length=2048, hop_length=512)
        ])
        energy_mean = np.mean(energy) if energy.size > 0 else 0

        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # Formants, jitter, shimmer, and HNR
        f1, f2, f3 = extract_formants(file_path)
        jitter, shimmer = extract_jitter_shimmer(file_path)
        hnr = extract_hnr(file_path)

        stats = [
            pitch_mean, pitch_std, energy_mean,
            spectral_centroid, spectral_bandwidth, spectral_rolloff,
            f1, f2, f3, jitter, shimmer, hnr
        ]

        # Final feature vector
        features = np.concatenate([
            mfccs_mean, delta_mean, delta2_mean,
            np.array(stats)
        ])

        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def prepare_features_for_prediction(audio_file_path):
    features = extract_single_audio_features(audio_file_path)
    if features is None:
        return None

    # Load the pre-trained scaler
    try:
        scaler = joblib.load("scaler.pkl")
        # Ensure the feature vector has the expected number of features before scaling
        expected_feature_length = (N_MFCC * 3) + 12  # 12 is the length of the 'stats' list
        if len(features) != expected_feature_length:
            print(f"Warning: Feature vector length mismatch. Expected {expected_feature_length}, got {len(features)}. Prediction might be unreliable.")
            # You might want to handle this differently, e.g., padding or raising an error
            if len(features) < expected_feature_length:
                features = np.pad(features, (0, expected_feature_length - len(features)), mode='constant')
            else:
                features = features[:expected_feature_length]

        scaled_features = scaler.transform(features.reshape(1, -1))
        return scaled_features
    except FileNotFoundError:
        print("Error: 'scaler.pkl' not found. Make sure to train the model first to save the scaler.")
        return None
    except Exception as e:
        print(f"Error scaling features: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    audio_file = "test_samples/audio.mp3"  # Replace with the actual path to your .mp3 file

    # Ensure the test_samples directory exists and contains an audio file for testing
    if not os.path.exists("test_samples"):
        os.makedirs("test_samples")
        # Create a dummy audio file for testing if one doesn't exist
        if not os.path.exists(audio_file):
            import soundfile as sf
            dummy_audio = np.random.rand(44100 * 5) * 0.1  # 5 seconds of random audio
            sf.write(audio_file, dummy_audio, 44100, format='MP3')
            print(f"Created a dummy audio file at {audio_file} for testing.")

    scaled_features = prepare_features_for_prediction(audio_file)

    if scaled_features is not None:
        print("\nExtracted and scaled features:")
        print(scaled_features)
        print(f"Shape of scaled features: {scaled_features.shape}")
        # Now you can use 'scaled_features' to make a prediction with your trained model.
        # For example, if you have a loaded model called 'gender_model':
        # prediction = gender_model.predict(scaled_features)
        # print("Predicted gender:", prediction)