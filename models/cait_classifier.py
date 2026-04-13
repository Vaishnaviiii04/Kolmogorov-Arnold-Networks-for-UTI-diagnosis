import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
class_names = dataset.classes  # <-- class labels from folder names

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =========================
# MODEL
# =========================
class CaiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12):
        super(CaiT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # NOTE: default TransformerEncoderLayer uses (seq_len, batch, dim) unless batch_first=True.
        # We'll set batch_first=True so input is (batch, seq_len, dim) which matches your code.
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)       # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)               # (B, 1+num_patches, dim)

        x = x + self.pos_embed
        x = self.transformer(x)
        return self.fc(x[:, 0])  # CLS token output

# =========================
# SETUP
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CaiT(img_size=224, patch_size=16, num_classes=3, dim=768, depth=12, heads=12).to(device)

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
plt.title("CaiT Confusion Matrix")
plt.tight_layout()

plt.savefig("cait_cm.png")
plt.show()

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "cait_classifier.pth")
print("✅ CaiT model saved successfully: cait_classifier.pth")

# =========================
# SAVE METRICS (JSON)
# =========================
cait_metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "class_names": class_names
}

with open("metrics/cait_metrics.json", "w") as f:
    json.dump(cait_metrics, f, indent=2)

print("✅ CaiT metrics saved to metrics/cait_metrics.json")
print("✅ Confusion matrix image saved to metrics/cait_cm.png")
