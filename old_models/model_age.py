import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

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

# PyTorch dataset
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AudioDataset(X_train, y_train_binary)
val_dataset = AudioDataset(X_val, y_val_binary)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
class AgeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 350),
            nn.ReLU(),
            nn.BatchNorm1d(350),
            nn.Dropout(0.2),
            nn.Linear(350, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.2),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeClassifier(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035, weight_decay=1e-4)

# Trening (bez early stopping)
best_acc = 0.0
epochs = 100

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validacija
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch}, Loss: {running_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Spremi najbolji model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

# Učitaj najbolji model
model.load_state_dict(torch.load("best_model.pt"))
model.save("Model/age")
# Evaluacija
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    # Confusion Matrix
    labels = ["Child", "Adult"]
    cm = confusion_matrix(y_true, y_pred)
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
    print(classification_report(y_true, y_pred, target_names=labels))

    auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")

# Pokreni evaluaciju
evaluate_model(model, val_loader, device)
