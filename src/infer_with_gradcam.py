# src/infer_with_gradcam.py
# Predict + Grad-CAM overlay + simple textual explanation.
#
# Run:
#   python src/infer_with_gradcam.py "data/raw/bottle/test/contamination/000.png"
#
# Output:
#   outputs/cams/<image>_pred-<class>_cam.jpg
#   outputs/cams/<image>_result.json

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

from gradcam_utils import GradCAM, overlay_cam_on_image, bottle_zone_explanation , cam_peakiness

CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")
OUT_DIR = Path("outputs/cams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320  # must match what you used during training/eval
UNCERTAIN_THRESH = 0.60  # industrial-style safety net (tune later)

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
    # Good default target layer: last Conv2d in features
    # model.features[-1] is ConvBNReLU; [0] is Conv2d
    return model.features[-1][0]

def main(img_path: str):
    model, classes = load_model()
    target_layer = pick_target_layer_mobilenetv2(model)
    cam_engine = GradCAM(model, target_layer)

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(device)

    # forward for probs
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    conf = float(probs[pred_idx])

    # uncertainty gating (optional but recommended for production)
    final_label = pred_class if conf >= UNCERTAIN_THRESH else "UNCERTAIN"

    # Grad-CAM requires backward; run cam for predicted class
    cam = cam_engine(x, pred_idx)
    peak = cam_peakiness(cam)

    # If predicted good + high confidence + diffuse CAM => don't show heatmap
    SHOW_CAM = not (pred_class == "good" and conf >= 0.80 and peak < 0.15)
    


    rgb = np.array(pil).astype(np.uint8)
    overlay = overlay_cam_on_image(rgb, cam, alpha=0.45)
    if SHOW_CAM:
        overlay = overlay_cam_on_image(rgb, cam, alpha=0.45)
    else:
        overlay = rgb  # plain image, no heatmap


    out_img = OUT_DIR / (Path(img_path).stem + f"_pred-{pred_class}_cam.jpg")
    cv2.imwrite(str(out_img), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # top-3 classes
    top3 = sorted([(classes[i], float(probs[i])) for i in range(len(classes))],
                  key=lambda t: t[1], reverse=True)[:3]

    explanation = bottle_zone_explanation(cam)


    result = {
        "image": img_path,
        "final_label": final_label,
        "pred_class": pred_class,
        "confidence": conf,
        "top3": top3,
        "explanation": explanation,
        "cam_overlay_path": str(out_img),
        "uncertain_threshold": UNCERTAIN_THRESH,
    }

    out_json = OUT_DIR / (Path(img_path).stem + "_result.json")
    out_json.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print("\n✅ Saved CAM overlay:", out_img)
    print("✅ Saved result JSON :", out_json)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/infer_with_gradcam.py "path/to/image.png"')
        raise SystemExit(1)
    main(sys.argv[1])
