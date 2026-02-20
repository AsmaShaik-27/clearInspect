import os
import sys
import json
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory

# ---------------------------------------------------
# Base Project Path (clearInspect)
# ---------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_FOLDER = os.path.join(BASE_DIR, "data", "web_uploads")
CAM_FOLDER = os.path.join(BASE_DIR, "outputs", "cams")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CAM_FOLDER, exist_ok=True)

# ---------------------------------------------------
# Flask App
# ---------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------
# API: Analyze Image
# ---------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"})

    # Save uploaded image
    image_path = os.path.join(DATA_FOLDER, file.filename)
    file.save(image_path)

    # ---------------------------------------------------
    # Run PyTorch inference script (using SAME venv)
    # ---------------------------------------------------
    command = [
        sys.executable,
        os.path.join(BASE_DIR, "src", "infer_with_gradcam.py"),
        image_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        return jsonify({
            "success": False,
            "error": result.stderr
        })

    # ---------------------------------------------------
    # Get latest GradCAM image
    # ---------------------------------------------------

    cam_files = sorted(
        [f for f in os.listdir(CAM_FOLDER) if f.endswith(".jpg") or f.endswith(".png")],
        key=lambda x: os.path.getmtime(os.path.join(CAM_FOLDER, x)),
        reverse=True
    )
    if not cam_files:
        return jsonify({"success": False, "error": "GradCAM output not found"})

    gradcam_file = cam_files[0]

 
    # Find latest result JSON file
    result_files = sorted(
        [f for f in os.listdir(CAM_FOLDER) if f.endswith("_result.json")],
        key=lambda x: os.path.getmtime(os.path.join(CAM_FOLDER, x)),
        reverse=True
    )

    if not result_files:
        return jsonify({"success": False, "error": "Result JSON not found"})

    latest_result = os.path.join(CAM_FOLDER, result_files[0])

    with open(latest_result, "r") as f:
        model_output = json.load(f)

    predicted_class = model_output["pred_class"]
    confidence = model_output["confidence"]
    final_label = model_output["final_label"]

    is_defective = predicted_class != "good"

    # Simple severity logic based on confidence
    severity_score = int(confidence * 100)

    if severity_score < 40:
        severity_level = "LOW"
    elif severity_score < 70:
        severity_level = "MEDIUM"
    else:
        severity_level = "HIGH"

    return jsonify({
        "success": True,
        "original_url": f"/data/web_uploads/{file.filename}",
        "heatmap_url": f"/outputs/cams/{gradcam_file}",
        "predicted_class": predicted_class,
        "is_defective": is_defective,
        "severity_score": severity_score,
        "severity_level": severity_level
    })


# ---------------------------------------------------
# Serve uploaded images
# ---------------------------------------------------
@app.route("/data/web_uploads/<path:filename>")
def serve_uploaded(filename):
    return send_from_directory(DATA_FOLDER, filename)


# ---------------------------------------------------
# Serve GradCAM outputs
# ---------------------------------------------------
@app.route("/outputs/cams/<path:filename>")
def serve_cam(filename):
    return send_from_directory(CAM_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
