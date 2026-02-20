from pathlib import Path
import time
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

from gradcam_utils import (
    GradCAM, overlay_cam_on_image, cam_peakiness,
    bottle_zone_explanation, cam_to_colormap
)

# ------------------------
# Config
# ------------------------
CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")
OUT_DIR = Path("outputs/live")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "events").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "cams").mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320

# Thresholds
UNCERTAIN_THRESH = 0.60
GOOD_CONF_THRESH = 0.75
GOOD_PEAK_THRESH = 0.20

# Video settings
CAMERA_INDEX = 0          # 0 = default webcam
FRAME_SKIP = 3            # process every Nth frame (CPU friendly)
ROI_SCALE = 0.70          # central crop % of frame width/height

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

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

def center_crop(frame_bgr, scale=0.7):
    h, w = frame_bgr.shape[:2]
    ch, cw = int(h * scale), int(w * scale)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    roi = frame_bgr[y0:y0+ch, x0:x0+cw]
    return roi, (x0, y0, cw, ch)

def put_label(frame, text, y=30):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)

def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    model, classes = load_model()
    cam_engine = GradCAM(model, pick_target_layer_mobilenetv2(model))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam/video source.")

    frame_id = 0
    last_save_time = 0.0

    print("✅ Live inspection started. प्रेस 'q' to quit. Press 's' to save a snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        display = frame.copy()

        # ROI (Phase-1: center crop)
        roi_bgr, (x0, y0, cw, ch) = center_crop(frame, scale=ROI_SCALE)
        cv2.rectangle(display, (x0,y0), (x0+cw, y0+ch), (0,255,255), 2)

        if frame_id % FRAME_SKIP == 0:
            # BGR -> RGB PIL
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(roi_rgb)

            x = tfm(pil).unsqueeze(0).to(device)

            # predict
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

            # GOOD suppression -> blue heatmap
            if pred_class == "good" and conf >= GOOD_CONF_THRESH and peak < GOOD_PEAK_THRESH:
                cam_to_show = np.zeros_like(cam, dtype=np.float32)
                cam_note = "Good attention suppressed"
            else:
                cam_to_show = cam
                cam_note = "CAM shown"

            # Create overlay for ROI preview
            overlay_rgb = overlay_cam_on_image(roi_rgb, cam_to_show, alpha=0.45)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            # Put overlay back into display
            overlay_resized = cv2.resize(overlay_bgr, (cw, ch))
            display[y0:y0+ch, x0:x0+cw] = overlay_resized

            # Text explanation
            expl = bottle_zone_explanation(cam_to_show) if final_label != "MANUAL_CHECK" else "Low confidence → manual check."
            put_label(display, f"Pred: {pred_class}  Conf: {conf:.2f}  Decision: {decision}", y=30)
            put_label(display, f"Peak: {peak:.2f}  {cam_note}", y=60)
            put_label(display, f"Explain: {expl}", y=90)

            # Auto-save defect/uncertain events (cool demo)
            now = time.time()
            if decision in ("REJECT", "MANUAL_CHECK") and (now - last_save_time) > 1.0:
                ts = int(now)
                out_img = OUT_DIR / "events" / f"{ts}_{decision}_{pred_class}.jpg"
                cv2.imwrite(str(out_img), display)

                heat = cam_to_colormap(cam_to_show)
                out_heat = OUT_DIR / "cams" / f"{ts}_{decision}_{pred_class}_heat.jpg"
                cv2.imwrite(str(out_heat), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))

                out_json = OUT_DIR / "events" / f"{ts}_{decision}_{pred_class}.json"
                out_json.write_text(json.dumps({
                    "timestamp": ts,
                    "pred_class": pred_class,
                    "confidence": conf,
                    "final_label": final_label,
                    "decision": decision,
                    "cam_peakiness": float(peak),
                    "explanation": expl
                }, indent=2))

                last_save_time = now

        cv2.imshow("ClearInspect - Live Inspection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            ts = int(time.time())
            snap = OUT_DIR / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(snap), display)
            print("Saved snapshot:", snap)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stopped.")

if __name__ == "__main__":
    main()