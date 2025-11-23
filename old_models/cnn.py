import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========== 1. ENHANCED MFCC EXTRACTION WITH CACHING ==========
N_MFCC = 13
MAX_LEN = 300

def extract_augmented_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)
        # Data augmentation
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-2, 2))
        # Enhanced MFCC with delta features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2]).T
        if features.shape[0] > MAX_LEN:
            features = features[:MAX_LEN]
        else:
            features = np.pad(features, ((0, MAX_LEN - features.shape[0]), (0, 0)))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_cache_paths(tsv_path):
    base_name = os.path.splitext(os.path.basename(tsv_path))[0]
    return (
        f"{base_name}_mfcc.npy",
        f"{base_name}_labels.npy",
        f"{base_name}_gender_encoder.pkl",
        f"{base_name}_age_encoder.pkl"
    )

def process_dataset(audio_directory, tsv_file):
    # Check for cached features
    mfcc_path, labels_path, gender_enc_path, age_enc_path = get_cache_paths(tsv_file)
    if os.path.exists(mfcc_path) and os.path.exists(labels_path):
        print(f"Loading cached features from {mfcc_path} and {labels_path}")
        mfcc_sequences = np.load(mfcc_path)
        labels_dict = np.load(labels_path, allow_pickle=True).item()
        return mfcc_sequences, labels_dict

    # If not cached, process data
    df = pd.read_csv(tsv_file, sep='\t')
    df['gender'] = df['gender'].map({
        'male_masculine': 'male',
        'female_feminine': 'female'
    })
    rows = [(os.path.join(audio_directory, row['path']), row['gender'], row['age_group']) 
            for _, row in df.iterrows()]

    print(f"\n🔄 Starting parallel MFCC extraction using {cpu_count()} cores...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_augmented_mfcc, [row[0] for row in rows]), total=len(rows)))
    mfcc_sequences = [res for res in results if res is not None]
    labels_list = [rows[i][1:] for i, res in enumerate(results) if res is not None]

    print(f"Extracted {len(mfcc_sequences)} MFCC sequences.")

    # Convert labels_list to dictionary of lists
    genders = [g for g, a in labels_list]
    ages = [a for g, a in labels_list]

    # Encode separately
    gender_le = LabelEncoder()
    age_le = LabelEncoder()
    genders_encoded = gender_le.fit_transform(genders)
    ages_encoded = age_le.fit_transform(ages)

    # Save encoders and data for future use
    np.save(mfcc_path, np.array(mfcc_sequences))
    np.save(labels_path, {'gender': genders_encoded, 'age': ages_encoded})
    joblib.dump(gender_le, gender_enc_path)
    joblib.dump(age_le, age_enc_path)

    return np.array(mfcc_sequences), {'gender': genders_encoded, 'age': ages_encoded}

# ========== 2. ENHANCED CNN MODEL WITH ATTENTION ==========
def build_enhanced_cnn(input_shape, num_age_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    
    # Multi-scale Feature Extraction
    branch1 = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    branch2 = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.concatenate([branch1, branch2])
    
    # Channel Attention
    def channel_attention(input_tensor):
        avg = layers.GlobalAveragePooling1D()(input_tensor)
        dense = layers.Dense(input_tensor.shape[-1] // 8, activation='relu')(avg)
        dense = layers.Dense(input_tensor.shape[-1], activation='sigmoid')(dense)
        return layers.Multiply()([input_tensor, dense])
    
    x = layers.SeparableConv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = channel_attention(x)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.SeparableConv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = channel_attention(x)
    x = layers.SpatialDropout1D(0.4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Multi-task Output (positional, not named)
    gender_out = layers.Dense(1, activation='sigmoid')(x)
    age_out = layers.Dense(num_age_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=[gender_out, age_out])

# ========== 3. TRAINING CONFIGURATION ==========
def configure_training(model):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
        clipnorm=1.0
    )
    
    # Custom F1 metric for gender
    class F1Score(tf.keras.metrics.Metric):
        def __init__(self, name='f1_score', **kwargs):
            super().__init__(name=name, **kwargs)
            self.precision = tf.keras.metrics.Precision()
            self.recall = tf.keras.metrics.Recall()
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            self.precision.update_state(y_true, y_pred)
            self.recall.update_state(y_true, y_pred)
            
        def result(self):
            p = self.precision.result()
            r = self.recall.result()
            return 2 * ((p * r) / (p + r + 1e-6))
            
        def reset_state(self):
            self.precision.reset_state()
            self.recall.reset_state()
    
    model.compile(
        optimizer=optimizer,
        loss=['binary_crossentropy', 'categorical_crossentropy'],
        metrics=[['accuracy', F1Score()], ['accuracy']],
        loss_weights=[0.4, 0.6]
    )
    
    return [
        callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor='val_loss',
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            monitor='val_loss'
        ),
        callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss'
        )
    ]

# ========== 4. MAIN EXECUTION ==========
if __name__ == "__main__":
    # Define paths
    audio_dir = "processed_audio/"
    train_tsv = "processed/updated_balanced_train.tsv"
    val_tsv = "processed/updated_balanced_validated.tsv"

    # Process or load data
    X_train, labels_train = process_dataset(audio_dir, train_tsv)
    X_val, labels_val = process_dataset(audio_dir, val_tsv)

    y_train_gender = labels_train['gender']
    y_train_age = labels_train['age']
    y_val_gender = labels_val['gender']
    y_val_age = labels_val['age']

    # Convert age to categorical
    num_age_classes = len(np.unique(np.concatenate([y_train_age, y_val_age])))
    y_train_age_cat = utils.to_categorical(y_train_age, num_age_classes)
    y_val_age_cat = utils.to_categorical(y_val_age, num_age_classes)

    # Build model
    model = build_enhanced_cnn(X_train.shape[1:], num_age_classes)
    
    # Print model summary
    model.summary()

    # Compute sample weights for each task
    from sklearn.utils.class_weight import compute_class_weight
    gender_classes = np.unique(y_train_gender)
    gender_weights = compute_class_weight('balanced', classes=gender_classes, y=y_train_gender)
    gender_class_weight = dict(zip(gender_classes, gender_weights))
    
    age_classes = np.unique(y_train_age)
    age_weights = compute_class_weight('balanced', classes=age_classes, y=y_train_age)
    age_class_weight = dict(zip(age_classes, age_weights))

    gender_sample_weights = np.array([gender_class_weight[c] for c in y_train_gender])
    age_sample_weights = np.array([age_class_weight[c] for c in y_train_age])

    # Train with positional outputs and sample weights
    history = model.fit(
        X_train,
        [y_train_gender, y_train_age_cat],  # List format for outputs
        validation_data=(X_val, [y_val_gender, y_val_age_cat]),
        epochs=150,
        batch_size=64,
        callbacks=configure_training(model),
        sample_weight=[gender_sample_weights, age_sample_weights],  # List format
        verbose=2
    )

    # Evaluation
    results = model.evaluate(X_val, [y_val_gender, y_val_age_cat], verbose=0)
    print(f"\nValidation Results:")
    print(f"Overall Loss: {results[0]:.4f}")
    print(f"Gender Loss: {results[1]:.4f}")
    print(f"Age Loss: {results[2]:.4f}")
    print(f"Gender Accuracy: {results[3]*100:.2f}%")
    print(f"Gender F1: {results[4]:.4f}")
    print(f"Age Accuracy: {results[5]*100:.2f}%")
