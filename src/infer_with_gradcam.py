# src/infer_with_gradcam.py
# Predict + Grad-CAM with "GOOD -> blue heatmap" behavior.
#
# What you asked:
# - For GOOD images, you want the grad image to be fully blue (no scary red).
# This script does that by forcing a BLANK CAM (all zeros) when:
#   pred_class == "good" AND confidence high AND CAM is diffuse (low peakiness)
#
# Run:
#   python src/infer_with_gradcam.py "data/raw/bottle/test/good/000.png"
#
# Output:
#   outputs/cams/<name>_overlay.jpg   (overlay on original image)
#   outputs/cams/<name>_heatmap.jpg   (heatmap-only; GOOD should be fully blue)
#   outputs/cams/<name>_result.json

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

from gradcam_utils import (
    GradCAM,
    overlay_cam_on_image,
    cam_to_colormap,
    cam_peakiness,
    bottle_zone_explanation,
)

CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")
OUT_DIR = Path("outputs/cams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320  # must match training/eval resize
UNCERTAIN_THRESH = 0.60  # if max prob below -> MANUAL_CHECK

# GOOD heatmap suppression settings (tune if needed)
GOOD_CONF_THRESH = 0.75      # if good confidence >= this
GOOD_PEAK_THRESH = 0.20      # and CAM is diffuse (peakiness < this)
# then we force cam_to_show = zeros -> fully blue heatmap

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_model():
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    ckpt = torch.load(CKPT, map_location=device)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, classes

def pick_target_layer_mobilenetv2(model):
    # last conv in MobileNetV2
    return model.features[-1][0]

def main(img_path: str):
    model, classes = load_model()

    target_layer = pick_target_layer_mobilenetv2(model)
    cam_engine = GradCAM(model, target_layer)

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    conf = float(probs[pred_idx])

    # decision (industrial)
    if conf < UNCERTAIN_THRESH:
        final_label = "MANUAL_CHECK"
        decision = "MANUAL_CHECK"
    else:
        final_label = pred_class
        decision = "ACCEPT" if pred_class == "good" else "REJECT"

    # Grad-CAM for predicted class
    cam = cam_engine(x, pred_idx)  # [h,w] in [0,1]
    peak = cam_peakiness(cam)

    # Force GOOD images to have blank heatmap (fully blue in heatmap-only view)
    if pred_class == "good" and conf >= GOOD_CONF_THRESH and peak < GOOD_PEAK_THRESH:
        cam_to_show = np.zeros_like(cam, dtype=np.float32)
        cam_note = f"GOOD suppression applied (conf={conf:.3f}, peak={peak:.3f})"
    else:
        cam_to_show = cam
        cam_note = f"Normal CAM shown (conf={conf:.3f}, peak={peak:.3f})"

    rgb = np.array(pil).astype(np.uint8)

    # Save overlay (on original image)
    overlay = overlay_cam_on_image(rgb, cam_to_show, alpha=0.45)
    out_overlay = OUT_DIR / (Path(img_path).stem + f"_pred-{pred_class}_overlay.jpg")
    cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


    # Explanation text (based on CAM actually shown)
    explanation = bottle_zone_explanation(cam_to_show) if final_label != "MANUAL_CHECK" else "Low confidence → manual inspection recommended."

    top3 = sorted([(classes[i], float(probs[i])) for i in range(len(classes))],
                  key=lambda t: t[1], reverse=True)[:3]

    result = {
        "image": img_path,
        "final_label": final_label,
        "decision": decision,
        "pred_class": pred_class,
        "confidence": conf,
        "top3": top3,
        "uncertain_threshold": UNCERTAIN_THRESH,
        "cam_peakiness": float(peak),
        "cam_note": cam_note,
        "explanation": explanation,
        "overlay_path": str(out_overlay),
    }

    out_json = OUT_DIR / (Path(img_path).stem + "_result.json")
    out_json.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print("\n✅ Saved overlay :", out_overlay)
    print("✅ Saved JSON    :", out_json)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/infer_with_gradcam.py "path/to/image.png"')
        raise SystemExit(1)
    main(sys.argv[1])
