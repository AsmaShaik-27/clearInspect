from pathlib import Path
import csv
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

from gradcam_utils import GradCAM, overlay_cam_on_image, cam_to_colormap, cam_peakiness, bottle_zone_explanation

CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")
TEST_ROOT = Path("data/processed/bottle_splits/test")   # use processed split test
OUT_DIR = Path("outputs/batch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320
UNCERTAIN_THRESH = 0.60
GOOD_CONF_THRESH = 0.75
GOOD_PEAK_THRESH = 0.20

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

def pick_target_layer_mobilenetv2(model):
    return model.features[-1][0]

def iter_images(root: Path):
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        true_label = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield true_label, p

def main():
    model, classes = load_model()
    cam_engine = GradCAM(model, pick_target_layer_mobilenetv2(model))

    rows = []
    (OUT_DIR / "overlays").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "json").mkdir(parents=True, exist_ok=True)

    for true_label, img_path in iter_images(TEST_ROOT):
        pil = Image.open(img_path).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_class = classes[pred_idx]
        conf = float(probs[pred_idx])

        # decision
        if conf < UNCERTAIN_THRESH:
            final_label = "MANUAL_CHECK"
            decision = "MANUAL_CHECK"
        else:
            final_label = pred_class
            decision = "ACCEPT" if pred_class == "good" else "REJECT"

        # CAM
        cam = cam_engine(x, pred_idx)
        peak = cam_peakiness(cam)

        # GOOD suppression (make good heatmap blue)
        if pred_class == "good" and conf >= GOOD_CONF_THRESH and peak < GOOD_PEAK_THRESH:
            cam_to_show = np.zeros_like(cam, dtype=np.float32)
        else:
            cam_to_show = cam

        rgb = np.array(pil).astype(np.uint8)

        overlay = overlay_cam_on_image(rgb, cam_to_show, alpha=0.45)
        heatmap = cam_to_colormap(cam_to_show)

        out_base = f"{img_path.stem}_true-{true_label}_pred-{pred_class}"
        out_overlay = OUT_DIR / "overlays" / f"{out_base}.jpg"
        out_heat = OUT_DIR / "heatmaps" / f"{out_base}.jpg"
        out_json = OUT_DIR / "json" / f"{out_base}.json"

        cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_heat), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        explanation = bottle_zone_explanation(cam_to_show) if final_label != "MANUAL_CHECK" else "Low confidence → manual check."

        rec = {
            "image": str(img_path),
            "true_label": true_label,
            "pred_class": pred_class,
            "confidence": conf,
            "final_label": final_label,
            "decision": decision,
            "cam_peakiness": float(peak),
            "explanation": explanation,
            "overlay_path": str(out_overlay),
            "heatmap_path": str(out_heat),
        }
        out_json.write_text(json.dumps(rec, indent=2))

        rows.append([str(img_path), true_label, pred_class, f"{conf:.4f}", final_label, decision, f"{peak:.4f}"])

    # write CSV log
    csv_path = OUT_DIR / "inspection_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "pred_class", "confidence", "final_label", "decision", "cam_peakiness"])
        w.writerows(rows)

    print("✅ Batch complete")
    print("CSV log:", csv_path)
    print("Overlays:", OUT_DIR / "overlays")
    print("Heatmaps:", OUT_DIR / "heatmaps")
    print("JSON:", OUT_DIR / "json")

if __name__ == "__main__":
    main()
