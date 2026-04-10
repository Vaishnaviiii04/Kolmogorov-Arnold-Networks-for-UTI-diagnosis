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
import time
import numpy as np
import os
import json


# =========================
# MODEL FACTORY (SAFE TO IMPORT)
# =========================
def build_googlenet(num_classes, device):
    model = models.googlenet(pretrained=False, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)


# =========================
# TRAINING (RUN ONLY WHEN EXECUTED DIRECTLY)
# =========================
if __name__ == "__main__":

    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # TRANSFORMS
    # =========================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # DATASET
    # =========================
    data_path = "data_samples"
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    class_names = dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # =========================
    # MODEL
    # =========================
    model = build_googlenet(num_classes, device)

    # =========================
    # LOSS & OPTIMIZER
    # =========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =========================
    # TRAINING
    # =========================
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
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        print(f"Epoch {epoch+1} Average Loss: {running_loss / len(train_loader):.4f}")

    print(f"\nTraining time: {time.time() - start_time:.2f} seconds")

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # =========================
    # METRICS
    # =========================
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="weighted")
    rec = recall_score(all_targets, all_preds, average="weighted")
    f1 = f1_score(all_targets, all_preds, average="weighted")

    print(f"Validation Accuracy : {acc*100:.2f}%")
    print(f"Precision           : {prec:.4f}")
    print(f"Recall              : {rec:.4f}")
    print(f"F1 Score            : {f1:.4f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(all_targets, all_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - GoogLeNet")
    plt.tight_layout()
    plt.savefig("googlenet.png")
    plt.show()

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "googlenet_classifier.pth")
    print("✅ GoogLeNet model saved successfully")

    # =========================
    # SAVE METRICS
    # =========================
    os.makedirs("metrics", exist_ok=True)

    googlenet_metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "confusion_matrix": cm.tolist()
    }

    with open("metrics/googlenet_metrics.json", "w") as f:
        json.dump(googlenet_metrics, f, indent=2)

    print("✅ GoogLeNet metrics saved to metrics/googlenet_metrics.json")


