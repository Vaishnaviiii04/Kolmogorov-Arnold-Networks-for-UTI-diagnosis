import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F

from models.load_kan_classifier import load_kan_model
from models.load_googlenet import load_googlenet

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODELS
# =========================
try:
    kan_model = load_kan_model("kan_c_mlp.pth", device=DEVICE)
    kan_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load KAN-C-MLP model: {e}")

try:
    googlenet_model = load_googlenet(
        weight_path="googlenet_classifier.pth",
        num_classes=3,
        device=DEVICE
    )
    googlenet_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load GoogLeNet model: {e}")

# =========================
# TRANSFORMS
# =========================
kan_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

gnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = ["Negative", "Positive", "Uncertain"]

# =========================
# HELPER: RUN INFERENCE
# =========================
def run_inference(image_paths):
    results = []

    # counters
    kan_counts = {c: 0 for c in CLASS_NAMES}
    gnet_counts = {c: 0 for c in CLASS_NAMES}

    for path in image_paths:
        try:
            img_rgb = Image.open(path).convert("RGB")
            img_gray = img_rgb.convert("L")

            # -----------------
            # KAN-C-MLP
            # -----------------
            kan_tensor = kan_transform(img_gray).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                try:
                    kan_logits = kan_model(kan_tensor)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"KAN-C-MLP forward pass failed on {path}. "
                        f"Check model architecture vs weights. Original error: {e}"
                    )
                kan_probs = F.softmax(kan_logits, dim=1)
                kan_pred = torch.argmax(kan_probs, dim=1).item()
                kan_conf = kan_probs[0, kan_pred].item() * 100

            kan_label = CLASS_NAMES[kan_pred]
            kan_counts[kan_label] += 1

            # -----------------
            # GoogLeNet
            # -----------------
            gnet_tensor = gnet_transform(img_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                gnet_logits = googlenet_model(gnet_tensor)
                gnet_probs = F.softmax(gnet_logits, dim=1)
                gnet_pred = torch.argmax(gnet_probs, dim=1).item()
                gnet_conf = gnet_probs[0, gnet_pred].item() * 100

            gnet_label = CLASS_NAMES[gnet_pred]
            gnet_counts[gnet_label] += 1

            results.append({
                "image": os.path.basename(path),
                "kan": kan_label,
                "kan_confidence": round(kan_conf, 2),
                "googlenet": gnet_label,
                "googlenet_confidence": round(gnet_conf, 2)
            })

        except Exception as e:
            results.append({
                "image": os.path.basename(path),
                "error": str(e)
            })

    # -----------------
    # Distribution (%)
    # -----------------
    total = len(image_paths) if len(image_paths) > 0 else 1
    distribution = {
        "kan": {k: round((v / total) * 100, 2) for k, v in kan_counts.items()},
        "googlenet": {k: round((v / total) * 100, 2) for k, v in gnet_counts.items()}
    }

    return {
        "predictions": results,
        "class_distribution": distribution
    }
