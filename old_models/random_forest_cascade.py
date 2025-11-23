import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Load data ---
train_features = np.load('mfcc/train_dataset_features.npy')
train_labels = np.load('mfcc/train_dataset_labels.npy')
val_features = np.load('mfcc/val_dataset_features.npy')
val_labels = np.load('mfcc/val_dataset_labels.npy')

# --- Split gender and age labels ---
train_gender_labels = train_labels[:, 0]
train_age_labels = train_labels[:, 1]
val_gender_labels = val_labels[:, 0]
val_age_labels = val_labels[:, 1]

# --- Train Gender Model ---
print("\n--- Training Gender Classifier ---")
gender_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
gender_model.fit(train_features, train_gender_labels)

# Save gender model
os.makedirs("Models", exist_ok=True)
joblib.dump(gender_model, 'Models/gender_model.pkl')
print("✅ Saved gender model as 'Models/gender_model.pkl'")

# --- Predict Gender for Age Model Input ---
train_pred_gender = gender_model.predict(train_features)
val_pred_gender = gender_model.predict(val_features)

# --- Encode gender as one-hot for use in age model ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_gender_onehot = encoder.fit_transform(train_pred_gender.reshape(-1, 1))
val_gender_onehot = encoder.transform(val_pred_gender.reshape(-1, 1))

# --- Append predicted gender to original features ---
train_features_with_gender = np.hstack((train_features, train_gender_onehot))
val_features_with_gender = np.hstack((val_features, val_gender_onehot))

# --- Train Age Model ---
print("\n--- Training Age Classifier (using predicted gender) ---")
age_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
age_model.fit(train_features_with_gender, train_age_labels)

# Save age model
joblib.dump(age_model, 'Models/age_model_with_gender_input.pkl')
print("✅ Saved age model as 'Models/age_model_with_gender_input.pkl'")

# --- Predict Age using predicted gender ---
val_age_pred = age_model.predict(val_features_with_gender)

# --- Combine predictions for evaluation ---
val_combined_pred = np.array([f"{g}_{a}" for g, a in zip(val_pred_gender, val_age_pred)])
val_combined_true = np.array([f"{g}_{a}" for g, a in zip(val_gender_labels, val_age_labels)])

# --- Encode combined labels for confusion matrix ---
label_encoder = LabelEncoder()
val_combined_true_enc = label_encoder.fit_transform(val_combined_true)
val_combined_pred_enc = label_encoder.transform(val_combined_pred)

# --- Report and Confusion Matrix ---
print("\n--- Classification Report (Predicted Gender + Predicted Age) ---")
print(classification_report(val_combined_true, val_combined_pred))
joblib.dump(encoder, 'Models/gender_encoder.pkl')
cm = confusion_matrix(val_combined_true, val_combined_pred, labels=label_encoder.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Cascaded Gender → Age')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix_cascaded.png')
print("✅ Saved confusion matrix as 'confusion_matrix_cascaded.png'")
