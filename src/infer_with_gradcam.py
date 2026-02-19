from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

from gradcam_utils import GradCAM, overlay_cam_on_image

CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")
OUT_DIR = Path("outputs/cams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_model():
    ckpt = torch.load(CKPT, map_location=device)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, classes

def zone_explanation(cam: np.ndarray):
    # simple 3x3 grid zone explanation
    h, w = cam.shape
    gh, gw = h // 3, w // 3
    best = (-1, None)
    for r in range(3):
        for c in range(3):
            patch = cam[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
            score = float(patch.mean())
            if score > best[0]:
                best = (score, (r, c))
    r, c = best[1]
    return f"Highest activation in grid zone (row={r+1}, col={c+1})"

def main(img_path: str):
    model, classes = load_model()

    # pick a good target layer: last conv in MobileNetV2 features
    target_layer = model.features[-1][0]  # Conv2d inside last block
    cam_engine = GradCAM(model, target_layer)

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    conf = float(probs[pred_idx])

    # Grad-CAM needs backward, so run cam separately
    cam = cam_engine(x, pred_idx)

    # create overlay on original RGB image
    rgb = np.array(pil).astype(np.uint8)
    overlay = overlay_cam_on_image(rgb, cam, alpha=0.45)

    out_img = OUT_DIR / (Path(img_path).stem + f"_pred-{pred_class}_cam.jpg")
    cv2.imwrite(str(out_img), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    explanation = zone_explanation(cam)
    out_json = OUT_DIR / (Path(img_path).stem + "_result.json")
    out = {
        "image": img_path,
        "pred_class": pred_class,
        "confidence": conf,
        "top3": sorted([(classes[i], float(probs[i])) for i in range(len(classes))],
                       key=lambda x: x[1], reverse=True)[:3],
        "explanation": explanation,
        "cam_image": str(out_img)
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nSaved CAM overlay:", out_img)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer_with_gradcam.py <path_to_image>")
        raise SystemExit(1)
    main(sys.argv[1])
