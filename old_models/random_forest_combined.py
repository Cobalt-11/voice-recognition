import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
train_features = np.load('mfcc/train_dataset_features.npy')
train_labels = np.load('mfcc/train_dataset_labels.npy')
val_features = np.load('mfcc/val_dataset_features.npy')
val_labels = np.load('mfcc/val_dataset_labels.npy')

# Split gender and age labels
train_gender_labels = train_labels[:, 0]
train_age_labels = train_labels[:, 1]
val_gender_labels = val_labels[:, 0]
val_age_labels = val_labels[:, 1]

# --- Combine gender and age into a single class label ---
train_combined_labels_str = np.array([f"{g}_{a}" for g, a in zip(train_gender_labels, train_age_labels)])
val_combined_labels_str = np.array([f"{g}_{a}" for g, a in zip(val_gender_labels, val_age_labels)])

# Encode string labels to integers for classification
label_encoder = LabelEncoder()
train_combined_encoded = label_encoder.fit_transform(train_combined_labels_str)
val_combined_encoded = label_encoder.transform(val_combined_labels_str)

print("✅ Combined gender and age into single multi-class labels.")

# --- Train Random Forest Classifier ---
print("\n--- Training Random Forest Classifier on Combined Labels ---")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)

rf.fit(train_features, train_combined_encoded)

# Save the model
os.makedirs("Models", exist_ok=True)
joblib.dump(rf, 'Models/best_random_forest_combined_classes.pkl')
joblib.dump(label_encoder, "Models/combined_label_encoder.pkl")

# --- Evaluate Model ---
val_preds_encoded = rf.predict(val_features)
val_preds_labels = label_encoder.inverse_transform(val_preds_encoded)

# Classification report
print("\n--- Classification Report (Combined Classes) ---")
print(classification_report(val_combined_labels_str, val_preds_labels))

# --- Confusion Matrix ---
cm = confusion_matrix(val_combined_labels_str, val_preds_labels, labels=label_encoder.classes_)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Combined Gender and Age Classes')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix_combined_classes.png')
print("✅ Saved confusion matrix as 'confusion_matrix_combined_classes.png'")
