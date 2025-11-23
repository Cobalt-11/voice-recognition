import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

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

# ========== 4. STANDARDIZACIJA PODATAKA ==========

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ========== 5. GRID SEARCH ZA NAJBOLJI k ==========

param_grid = {'n_neighbors': np.arange(1, 21)}  # k od 1 do 20
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5)
grid.fit(X_train_scaled, y_train_class)

best_k = grid.best_params_['n_neighbors']
print(f"🔎 Najbolji pronađeni broj susjeda (k): {best_k}")
print(f"🔎 Najbolja cross-val točnost: {grid.best_score_ * 100:.2f}%")

# ========== 6. TRENING NAJBOLJEG KNN MODELA ==========

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train_class)

# ========== 7. EVALUACIJA MODELA ==========

def evaluate_model(model, X_val, y_val_class, labels):
    y_pred = model.predict(X_val)

    cm = confusion_matrix(y_val_class, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (KNN with k={best_k})")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_val_class, y_pred, target_names=labels))

labels = ["Female Kid", "Female Adult", "Male Kid", "Male Adult"]
evaluate_model(best_knn, X_val_scaled, y_val_class, labels)

# ========== 8. TOČNOST NA TRENING I VALIDACIJI ==========

train_accuracy = best_knn.score(X_train_scaled, y_train_class)
val_accuracy = best_knn.score(X_val_scaled, y_val_class)

print(f"\n📊 Točnost na trening skupu: {train_accuracy * 100:.2f}%")
print(f"📊 Točnost na validacijskom skupu: {val_accuracy * 100:.2f}%")

# ========== 9. SPREMANJE MODELA I SCALERA ==========

joblib.dump(best_knn, 'Models/knn_best_model.pkl')
joblib.dump(scaler, 'Models/scaler_knn.pkl')
print("\n✅ KNN model i scaler su spremljeni!")
