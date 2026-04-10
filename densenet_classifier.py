import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import os
import time
import json

# =========================
# TRANSFORMS + DATA
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data_path = "data_samples"
dataset = datasets.ImageFolder(root=data_path, transform=transform)
class_names = dataset.classes  # folder names = class labels

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =========================
# MODEL
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: newer torchvision uses weights=... instead of pretrained=True
# Keeping your style but making it robust:
try:
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
except Exception:
    model = models.densenet121(pretrained=True)

num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# =========================
# TRAINING
# =========================
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# =========================
# EVALUATION + METRICS
# =========================
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for data, targets in val_loader:
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

accuracy  = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
recall    = recall_score(all_targets, all_preds, average="macro", zero_division=0)
f1        = f1_score(all_targets, all_preds, average="macro", zero_division=0)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_targets, all_preds)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", ax=ax)
plt.title("DenseNet121 Confusion Matrix")
plt.tight_layout()

plt.savefig("densenet_cm.png")
plt.show()

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "densenet_classifier.pth")
print("✅ DenseNet model saved successfully: densenet_classifier.pth")

# =========================
# SAVE METRICS (JSON)
# =========================
densenet_metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "class_names": class_names
}

with open("metrics/densenet_metrics.json", "w") as f:
    json.dump(densenet_metrics, f, indent=2)

print("✅ DenseNet metrics saved to metrics/densenet_metrics.json")
print("✅ Confusion matrix image saved to metrics/densenet_cm.png")
