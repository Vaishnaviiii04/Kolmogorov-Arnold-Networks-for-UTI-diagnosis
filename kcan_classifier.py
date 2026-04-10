import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json

from kan_convolutional.KANConv import KAN_Convolutional_Layer


# =========================
# DATASET CLASS (SAFE)
# =========================
class UrineCultureData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.class_names = []

        for label, subfolder in enumerate(os.listdir(root_dir)):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                self.class_names.append(subfolder)
                for img_file in os.listdir(subfolder_path):
                    if img_file.endswith(".jpg"):
                        self.image_files.append(os.path.join(subfolder_path, img_file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def get_class_names(self):
        return self.class_names


# =========================
# MODEL (SAFE TO IMPORT)
# =========================
class KANC_MLP(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(5, (3, 3), device=device)
        self.conv2 = KAN_Convolutional_Layer(5, (3, 3), device=device)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(625, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# =========================
# TRAINING (RUN ONLY MANUALLY)
# =========================
if __name__ == "__main__":

    root = "data_samples"

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = UrineCultureData(root, transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = KANC_MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =========================
    # TRAINING
    # =========================
    for epoch in range(30):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    # =========================
    # EVALUATION
    # =========================
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(all_labels, all_preds)
    class_names = dataset.get_class_names()

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax)
    plt.title("KAN-C-MLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig("kanc_mlp_cm.png")
    plt.show()

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "kan_c_mlp.pth")
    print("✅ KAN-C-MLP model saved successfully")

    # =========================
    # SAVE METRICS
    # =========================
    os.makedirs("metrics", exist_ok=True)

    kan_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm.tolist()
}

    with open("metrics/kan_metrics.json", "w") as f:
        json.dump(kan_metrics, f, indent=2)


    print("✅ KAN metrics saved to metrics/kan_metrics.json")
