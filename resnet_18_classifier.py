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
import json
import time
import os

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset & Dataloader
# -----------------------------
data_path = "data_samples"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

class_names = dataset.classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Model (ResNet18)
# -----------------------------
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
num_epochs = 10
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Step [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

end_time = time.time()
print(f"\nTraining Time: {end_time - start_time:.2f} seconds")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average="weighted")
recall = recall_score(all_targets, all_preds, average="weighted")
f1 = f1_score(all_targets, all_preds, average="weighted")

print("\nValidation Metrics:")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# -----------------------------
# Save Metrics
# -----------------------------
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics/resnet18_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved to resnet18_metrics.json")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_targets, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("ResNet18 Confusion Matrix")
plt.savefig("resnet18_confusion_matrix.png")
plt.close()

print("Confusion matrix saved as resnet18_confusion_matrix.png")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "resnet18_classifier.pth")
print("\nModel saved as resnet18_classifier.pth")
