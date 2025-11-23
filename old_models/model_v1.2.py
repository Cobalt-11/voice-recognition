import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.regularizers import l2
# ========== 1. FUNKCIJA ZA PARSIRANJE LABELA ==========

def parse_labels(label_array):
    parsed = []
    for item in label_array:
        label_np = item[0] if isinstance(item[0], (np.ndarray, list)) else item
        try:
            gender = str(label_np[0]).lower()
            age = str(label_np[1]).lower()
            parsed.append((gender, age))
        except Exception as e:
            print(f"Greška pri parsiranju: {item} -> {e}")
    return np.array(parsed)

# ========== 2. UČITAVANJE PODATAKA ==========

X_train = np.load("train_dataset_features.npy")
X_val = np.load("val_dataset_features.npy")

y_train_raw = np.load("train_dataset_labels.npy", allow_pickle=True)
y_val_raw = np.load("val_dataset_labels.npy", allow_pickle=True)

y_train = parse_labels(y_train_raw)
y_val = parse_labels(y_val_raw)

# ========== 3. MAPIRANJE U 4 KLASE ==========

# 0: female kid, 1: female adult, 2: male kid, 3: male adult
def label_to_class(label_tuple):
    gender, age = label_tuple
    if gender == 'female' and age == 'kid':
        return 0
    elif gender == 'female' and age == 'adult':
        return 1
    elif gender == 'male' and age == 'kid':
        return 2
    elif gender == 'male' and age == 'adult':
        return 3
    else:
        raise ValueError(f"Neočekivana kombinacija: {label_tuple}")

y_train_class = np.array([label_to_class(lbl) for lbl in y_train])
y_val_class = np.array([label_to_class(lbl) for lbl in y_val])

y_train_cat = to_categorical(y_train_class, num_classes=4)
y_val_cat = to_categorical(y_val_class, num_classes=4)

# ========== 4. DEFINICIJA MODELA ==========

model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 klase
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

# ========== 5. TRENING ==========

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32
)

# ========== 6. EVALUACIJA ==========

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

labels = ["Female Kid", "Female Adult", "Male Kid", "Male Adult"]
evaluate_model(model, X_val, y_val_class, labels)

# ========== 7. CRTANJE GRAFOVA ==========

def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Loss')

    plt.tight_layout()
    plt.show()

plot_history(history, "4-Class Gender+Age Model")

# ========== 8. TOČNOST NA TRENING I VALIDACIJI ==========

train_loss, train_accuracy = model.evaluate(X_train, y_train_cat, verbose=0)
val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)

print(f"\n📊 Točnost na trening skupu: {train_accuracy * 100:.2f}%")
print(f"📊 Točnost na validacijskom skupu: {val_accuracy * 100:.2f}%")
