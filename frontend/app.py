import os
import json
from flask import Flask, jsonify, render_template, Response
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models

# ---------------------------------------------------
# Flask App
# ---------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------
# Device
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------------------------------------------------
# Load Model ONCE
# ---------------------------------------------------
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CKPT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
    "best_mobilenetv2_multiclass.pth"
)
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=device)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    return model, classes

model, classes = load_model()

# ---------------------------------------------------
# Transform
# ---------------------------------------------------
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------------------------------------------------
# Global Live Status Variables
# ---------------------------------------------------
latest_class = "good"
latest_score = 0
latest_severity = "LOW"
latest_is_defective = False

# ---------------------------------------------------
# Live Camera Generator
# ---------------------------------------------------
def generate_frames():
    global latest_class, latest_score, latest_severity, latest_is_defective

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        x = tfm(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        pred_idx = int(probs.argmax())
        conf = float(probs[0][pred_idx])

        latest_class = classes[pred_idx]
        latest_score = int(conf * 100)
        latest_is_defective = latest_class != "good"

        # Severity logic
        if latest_score < 40:
            latest_severity = "LOW"
        elif latest_score < 70:
            latest_severity = "MEDIUM"
        else:
            latest_severity = "HIGH"

        label = f"{latest_class} ({conf:.2f})"

        color = (0,255,0) if not latest_is_defective else (0,0,255)

        cv2.putText(frame, label, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/live_status")
def live_status():
    return jsonify({
        "success": True,
        "predicted_class": latest_class,
        "severity_score": latest_score,
        "severity_level": latest_severity,
        "is_defective": latest_is_defective
    })

# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)