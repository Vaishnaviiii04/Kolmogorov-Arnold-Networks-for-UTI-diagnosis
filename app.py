from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import json

from inference import run_inference
from metrics.metrics_loader import load_all_metrics

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# Helper: Convert 3x3 confusion matrix → Binary
# Negative vs (Positive + Uncertain)
# -------------------------------------------------
def multiclass_to_binary(confusion_matrix):
    cm = np.array(confusion_matrix)

    if cm.shape != (3, 3):
        raise ValueError("Confusion matrix must be 3x3")

    TN = cm[0, 0]
    FP = cm[0, 1] + cm[0, 2]
    FN = cm[1, 0] + cm[2, 0]
    TP = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]

    return {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN)
    }

# -------------------------------------------------
# Load stored metrics ONCE at startup
# -------------------------------------------------
try:
    STORED_METRICS = load_all_metrics()

    # ----- KAN -----
    kan_raw = STORED_METRICS["kan"]
    kan_fixed = {
        "accuracy": kan_raw["accuracy"],
        "precision": kan_raw["precision"],
        "recall": kan_raw["recall"],
        "f1": kan_raw["f1"],
        "confusion_matrix": multiclass_to_binary(
            kan_raw["confusion_matrix"]
        )
    }

    # ----- GoogLeNet -----
    goog_raw = STORED_METRICS["googlenet"]
    goog_fixed = {
        "accuracy": goog_raw["accuracy"],
        "precision": goog_raw["precision"],
        "recall": goog_raw["recall"],
        "f1": goog_raw["f1"],
        "confusion_matrix": multiclass_to_binary(
            goog_raw["confusion_matrix"]
        )
    }

    FIXED_METRICS = {
        "kan": kan_fixed,
        "googlenet": goog_fixed
    }

    print("✅ Metrics loaded successfully")
    print(json.dumps(FIXED_METRICS, indent=2))

except Exception as e:
    print("❌ Failed to load metrics:", e)
    FIXED_METRICS = None

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/evaluation.html")
def evaluation_page():
    return render_template("evaluation.html")

@app.route("/about.html")
def about_page():
    return render_template("about.html")

# -------------------------------------------------
# Evaluation API (pretrained inference only)
# -------------------------------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():
    if FIXED_METRICS is None:
        return jsonify({"error": "Metrics not loaded"}), 500

    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    image_paths = []
    for file in files:
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        image_paths.append(save_path)

    # Run pretrained inference
    inference_output = run_inference(image_paths)

    response = {
        "predictions": inference_output.get("predictions", []),
        "class_distribution": inference_output.get("class_distribution", {}),
        "metrics": FIXED_METRICS
    }

    return jsonify(response)

# -------------------------------------------------
# Run server
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
