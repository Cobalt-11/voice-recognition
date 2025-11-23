import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ========== 1. DATA LOADING ==========
X_train = np.load("mfcc/train_dataset_features.npy")
X_val = np.load("mfcc/val_dataset_features.npy")

y_train_cat = np.load("Models/train_labels_encoded.npy", allow_pickle=True)
y_val_cat = np.load("Models/val_labels_encoded.npy", allow_pickle=True)

y_train_class = np.argmax(y_train_cat, axis=1)
y_val_class = np.argmax(y_val_cat, axis=1)

# ========== 2. CLASS WEIGHTS (for imbalance) ==========
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_class),
    y=y_train_class
)
class_weight_dict = dict(enumerate(class_weights))

# ========== 3. MODEL DEFINITION (Simpler & Stronger Regularization) ==========
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# ========== 4. CALLBACKS ==========
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ========== 5. TRAINING ==========
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=250,
    batch_size=32,
    callbacks=[tensorboard_callback, early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# ========== 6. EVALUATION ==========
def evaluate_model(model, X_val, y_val_class, labels):
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_val_class, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (4-Class Model)")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_val_class, y_pred, target_names=labels))

labels = ["Female Adult", "Female Kid", "Male Adult", "Male Kid"]
evaluate_model(model, X_val, y_val_class, labels)

# ========== 7. PLOTTING ==========
def plot_history(history, title):
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{title} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{title} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history, "4-Class Gender+Age Model")

# ========== 8. FINAL METRICS ==========
train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)

print(f"\n📊 Final Training Accuracy: {train_acc*100:.2f}%")
print(f"📊 Validation Accuracy: {val_acc*100:.2f}%")

# ========== 9. SAVE MODEL ==========
model.save('Models/dnn.keras')

# ========== 10. DATA AUGMENTATION PLACEHOLDER ==========
# If you have access to raw audio, consider augmenting your dataset before feature extraction.
# Example: Add noise, pitch shift, time stretch, etc. (use audiomentations or torchaudio)
