import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": round(acc * 100, 2),
        "confusion_matrix": cm.tolist()
    }
