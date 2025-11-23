import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import joblib

N_MFCC = 13
MAX_LEN = 300  # max number of frames after padding (tune this!)

def extract_mfcc_sequence(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)  # trim silence
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = mfcc.T  # shape (time_steps, n_mfcc)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(audio_directory, tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    df['gender'] = df['gender'].map({
        'male_masculine': 'male',
        'female_feminine': 'female'
    })
    rows = [(os.path.join(audio_directory, row['path']), row['gender'], row['age_group']) for _, row in df.iterrows()]

    print(f"\n🔄 Starting parallel MFCC extraction using {cpu_count()} cores...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_mfcc_sequence, [row[0] for row in rows]), total=len(rows)))

    # Filter out None results
    mfcc_sequences = [res for res in results if res is not None]
    labels = [rows[i][1:] for i, res in enumerate(results) if res is not None]  # gender, age_group

    print(f"Extracted {len(mfcc_sequences)} MFCC sequences.")

    return mfcc_sequences, labels

def pad_mfcc_sequences(mfcc_sequences, max_len=MAX_LEN):
    # Pad sequences to max_len frames, pad with zeros
    padded = pad_sequences(mfcc_sequences, maxlen=max_len, padding='post', dtype='float32')
    return padded

if __name__ == "__main__":
    audio_dir = "processed_audio/"

    train_tsv = "processed/updated_balanced_train.tsv"
    val_tsv = "processed/updated_balanced_validated.tsv"

    # Extract raw MFCC sequences
    X_train_seq, labels_train = process_dataset(audio_dir, train_tsv)
    X_val_seq, labels_val = process_dataset(audio_dir, val_tsv)

    # Pad sequences
    X_train_padded = pad_mfcc_sequences(X_train_seq, max_len=MAX_LEN)
    X_val_padded = pad_mfcc_sequences(X_val_seq, max_len=MAX_LEN)

    print(f"Train padded shape: {X_train_padded.shape}")  # (num_samples, MAX_LEN, N_MFCC)
    print(f"Val padded shape: {X_val_padded.shape}")

    # Example: convert categorical labels to one-hot encoding
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical

    # Combine gender + age group as one label for example (adjust to your case)
    combined_train_labels = [f"{g}_{a}" for g, a in labels_train]
    combined_val_labels = [f"{g}_{a}" for g, a in labels_val]

    le = LabelEncoder()
    y_train_int = le.fit_transform(combined_train_labels)
    y_val_int = le.transform(combined_val_labels)

    y_train_cat = to_categorical(y_train_int)
    y_val_cat = to_categorical(y_val_int)

    print(f"Number of classes: {len(le.classes_)}")

    # Save padded features and labels
    np.save("train_mfcc_padded.npy", X_train_padded)
    np.save("val_mfcc_padded.npy", X_val_padded)
    np.save("train_labels_encoded.npy", y_train_cat)
    np.save("val_labels_encoded.npy", y_val_cat)

    # Save label encoder for inference
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Saved label encoder to 'label_encoder.pkl'")
