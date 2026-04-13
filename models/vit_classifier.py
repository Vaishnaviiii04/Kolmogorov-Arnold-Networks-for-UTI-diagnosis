import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import timm
import time
import json
import os


# -----------------------------
# Device (CPU ONLY)
# -----------------------------
DEVICE = "cpu"
print("Using device:", DEVICE)


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data_samples"
NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LR = 3e-4


# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Dataset
# -----------------------------
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Model (Pretrained ViT)
# -----------------------------
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
).to(DEVICE)


# -----------------------------
# Freeze Backbone
# -----------------------------
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

print("✅ Backbone frozen. Training classification head only.")


# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# -----------------------------
# Training
# -----------------------------
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Step [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] Avg Loss: {avg_loss:.4f}")

end_time = time.time()
print(f"\n⏱ Training Time: {end_time - start_time:.2f} seconds")


# -----------------------------
# Evaluation
# -----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")

print("\n📊 Validation Metrics")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
class_names = dataset.classes

os.makedirs("metrics", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", ax=ax)
plt.title("ViT Confusion Matrix")
plt.tight_layout()
plt.savefig("vit_confusion_matrix.png")
plt.show()


# -----------------------------
# Save Metrics (JSON)
# -----------------------------
vit_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm.tolist()
}

with open("metrics/vit_metrics.json", "w") as f:
    json.dump(vit_metrics, f, indent=2)

print("✅ Metrics saved to metrics/vit_metrics.json")


# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "vit_classifier_fast_cpu.pth")
print("✅ Model saved as vit_classifier_fast_cpu.pth")
