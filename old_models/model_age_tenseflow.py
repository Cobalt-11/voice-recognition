import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
X_train = np.load("mfcc/train_dataset_features.npy")
y_train = np.load("mfcc/train_dataset_labels.npy", allow_pickle=True)
X_val = np.load("mfcc/val_dataset_features.npy")
y_val = np.load("mfcc/val_dataset_labels.npy", allow_pickle=True)

# Labela u 0 (child) ili 1 (adult)
def is_adult(label_tuple):
    return 1 if label_tuple[1] == 'adult' else 0

y_train_binary = np.array([is_adult(lbl) for lbl in y_train])
y_val_binary = np.array([is_adult(lbl) for lbl in y_val])

# Model
model = Sequential([
    Dense(500, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(350, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(200, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')  # Za binarnu klasifikaciju
])

# Kompajliraj model
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0006),
    metrics=['accuracy']
)

# Treniraj model
history = model.fit(X_train, y_train_binary, epochs=100, batch_size=32, validation_data=(X_val, y_val_binary))

# Evaluacija modela
def evaluate_model(model, X_val, y_val):
    # Predikcija
    y_prob = model.predict(X_val)
    y_pred = (y_prob > 0.5).astype(int)

    # Confusion Matrix
    labels = ["Child", "Adult"]
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=labels))

    auc = roc_auc_score(y_val, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")

# Pokreni evaluaciju
evaluate_model(model, X_val, y_val_binary)

model.save('Models/Age_model.keras')
