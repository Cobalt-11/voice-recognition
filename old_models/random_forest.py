import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and validation data
train_features = np.load('train_dataset_features.npy')
train_labels = np.load('train_dataset_labels.npy')
val_features = np.load('val_dataset_features.npy')
val_labels = np.load('val_dataset_labels.npy')

# Separate gender and age group labels
train_gender_labels = train_labels[:, 0]
train_age_labels = train_labels[:, 1]
val_gender_labels = val_labels[:, 0]
val_age_labels = val_labels[:, 1]

print("Loaded training and validation data.")
print(f"Training features shape: {train_features.shape}")
print(f"Training gender labels shape: {train_gender_labels.shape}")
print(f"Training age labels shape: {train_age_labels.shape}")
print(f"Validation features shape: {val_features.shape}")
print(f"Validation gender labels shape: {val_gender_labels.shape}")
print(f"Validation age labels shape: {val_age_labels.shape}")

# --- Model Training for Gender Classification ---
print("\n--- Training Random Forest for Gender Classification ---")

# Define the parameter grid for GridSearchCV
param_grid_gender = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'class_weight': ['balanced', None]
}

# Create a Random Forest Classifier
rf_gender = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search_gender = GridSearchCV(rf_gender, param_grid_gender, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_gender.fit(train_features, train_gender_labels)

# Get the best model and its parameters
best_rf_gender = grid_search_gender.best_estimator_
best_params_gender = grid_search_gender.best_params_

print(f"Best hyperparameters for gender classification: {best_params_gender}")

# Feature Importance Analysis for Gender
feature_importances_gender = best_rf_gender.feature_importances_
print("\nFeature Importances (Gender):", feature_importances_gender)

# Get indices of top 30 features
top_30_feature_indices_gender = np.argsort(feature_importances_gender)[-30:]

# Get the names or IDs of the top 30 features (if available)
top_30_feature_names_gender = ["feature_" + str(i) for i in top_30_feature_indices_gender]  # Replace with actual names if available

# Create a plot of feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_30_feature_names_gender, feature_importances_gender[top_30_feature_indices_gender])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 30 Feature Importances for Gender Classification')
plt.savefig('feature_importances_gender.png')  # Save the figure
print("✅ Saved feature importances plot for Gender as 'feature_importances_gender.png'")

# Save the top 30 feature indices
np.save('top_30_feature_indices_gender.npy', top_30_feature_indices_gender)
print("✅ Saved top 30 feature indices for Gender as 'top_30_feature_indices_gender.npy'")

# Select top 30 features
train_features_top30_gender = train_features[:, top_30_feature_indices_gender]
val_features_top30_gender = val_features[:, top_30_feature_indices_gender]

print("\nRetraining with top 30 features for Gender")

# Retrain model with top 30 features
best_rf_gender.fit(train_features_top30_gender, train_gender_labels)

# Evaluate the best model on the validation set
gender_predictions = best_rf_gender.predict(val_features_top30_gender)
accuracy_gender = accuracy_score(val_gender_labels, gender_predictions)
report_gender = classification_report(val_gender_labels, gender_predictions)

print(f"Validation accuracy (gender): {accuracy_gender:.4f}")
print("Classification Report (gender):\n", report_gender)

# Confusion Matrix for Gender
cm_gender = confusion_matrix(val_gender_labels, gender_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gender, annot=True, fmt='d', cmap='Blues',
    xticklabels=np.unique(val_gender_labels),
    yticklabels=np.unique(val_gender_labels))
plt.title('Confusion Matrix - Gender')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_gender.png')  # Save the confusion matrix plot
print("✅ Saved confusion matrix plot for Gender as 'confusion_matrix_gender.png'")

# Save the best gender classification model
joblib.dump(best_rf_gender, 'models/best_random_forest_gender.pkl')
print("✅ Saved the best gender classification model as 'models/best_random_forest_gender.pkl'")

# --- Model Training for Age Group Classification ---
print("\n--- Training Random Forest for Age Group Classification ---")

# Define the parameter grid for GridSearchCV
param_grid_age = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'class_weight': ['balanced', None]
}

# Create a Random Forest Classifier
rf_age = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search_age = GridSearchCV(rf_age, param_grid_age, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_age.fit(train_features, train_age_labels)

# Get the best model and its parameters
best_rf_age = grid_search_age.best_estimator_
best_params_age = grid_search_age.best_params_

print(f"Best hyperparameters for age group classification: {best_params_age}")

# Feature Importance Analysis for Age
feature_importances_age = best_rf_age.feature_importances_
print("\nFeature Importances (Age):", feature_importances_age)

# Get indices of top 30 features
top_30_feature_indices_age = np.argsort(feature_importances_age)[-30:]

# Get the names or IDs of the top 30 features (if available)
top_30_feature_names_age = ["feature_" + str(i) for i in top_30_feature_indices_age] # Replace with actual names if available

# Create a plot of feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_30_feature_names_age, feature_importances_age[top_30_feature_indices_age])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 30 Feature Importances for Age Group Classification')
plt.savefig('feature_importances_age.png')  # Save the figure
print("✅ Saved feature importances plot for Age as 'feature_importances_age.png'")

# Save the top 30 feature indices
np.save('top_30_feature_indices_age.npy', top_30_feature_indices_age)
print("✅ Saved top 30 feature indices for Age as 'top_30_feature_indices_age.npy'")

# Select top 30 features
train_features_top30_age = train_features[:, top_30_feature_indices_age]
val_features_top30_age = val_features[:, top_30_feature_indices_age]

print("\nRetraining with top 30 features for Age")

# Retrain model with top 30 features
best_rf_age.fit(train_features_top30_age, train_age_labels)

# Evaluate the best model on the validation set
age_predictions = best_rf_age.predict(val_features_top30_age)
accuracy_age = accuracy_score(val_age_labels, age_predictions)
report_age = classification_report(val_age_labels, age_predictions)

print(f"Validation accuracy (age): {accuracy_age:.4f}")
print("Classification Report (age):\n", report_age)

# Confusion Matrix for Age
cm_age = confusion_matrix(val_age_labels, age_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_age, annot=True, fmt='d', cmap='Blues',
    xticklabels=np.unique(val_age_labels),
    yticklabels=np.unique(val_age_labels))
plt.title('Confusion Matrix - Age')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_age.png')  # Save the confusion matrix plot
print("✅ Saved confusion matrix plot for Age as 'confusion_matrix_age.png'")

# Save the best age group classification model
joblib.dump(best_rf_age, 'models/best_random_forest_age.pkl')
print("✅ Saved the best age group classification model as 'models/best_random_forest_age.pkl'")
