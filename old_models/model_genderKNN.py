import os
import numpy as np
import pandas as pd
import librosa
import parselmouth
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import joblib

array_loaded = np.load('mfcc/val_dataset_features.npy')
print (array_loaded)

# Function to normalize features using StandardScaler
def normalize_features(features_train, features_val):
    # Normalize after padding to ensure feature vectors are consistent
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)
    
    # Save the scaler for later use
    joblib.dump(scaler, "scaler.pkl")
    print("Saved scaler as 'scaler.pkl'")

    return features_train_scaled, features_val_scaled

if __name__ == "__main__":
    audio_dir = "processed_audio/"

# Učitaj podatke
X_train = np.load("mfcc/train_dataset_features.npy")
y_train = np.load("mfcc/train_dataset_labels.npy", allow_pickle=True)
X_val = np.load("mfcc/val_dataset_features.npy")
y_val = np.load("mfcc/val_dataset_labels.npy", allow_pickle=True)

# Labela u 0 (female) ili 1 (male)
def is_male(label_tuple):
    return 1 if label_tuple[0] == 'male' else 1

y_train_la = np.array([is_male(lbl) for lbl in y_train])
y_val_la = np.array([is_male(lbl) for lbl in y_val])

features_train_scaled, features_val_scaled = normalize_features(X_train, X_val)

print("Jedinstvene klase u y_train:", np.unique(y_train))


    # Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train_scaled, y_train_la)



# Evaluate KNN Classifier
y_pred = knn.predict(features_val_scaled)
y_prob = knn.predict_proba(features_val_scaled)[:, 1]

    # Print Classification Report
print("\nClassification Report:")
print(classification_report(y_val_la, y_pred, target_names=["Male", "Female"]))

    # Confusion Matrix
cm = confusion_matrix(y_val_la, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

    # ROC AUC Score
auc = roc_auc_score(y_val_la, y_prob)
print(f"\nROC AUC Score: {auc:.4f}")
