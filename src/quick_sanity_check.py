from pathlib import Path
import random
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image

# ---- paths ----
CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")
TEST_ROOT = Path("data/processed/bottle_splits/test")  # use your split test
N_SAMPLES = 20  # increase if you want

IMG_SIZE = 320
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

def collect_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    items = []
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((cls, p))
    return items

def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    if not TEST_ROOT.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_ROOT}")

    model, classes = load_model()
    items = collect_images(TEST_ROOT)
    random.shuffle(items)
    items = items[:min(N_SAMPLES, len(items))]

    correct = 0
    conf_correct = []
    conf_wrong = []

    print("Device:", device)
    print("Classes:", classes)
    print(f"Sampling {len(items)} images from:", TEST_ROOT)
    print("-" * 70)

    for true_cls, img_path in items:
        pil = Image.open(img_path).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_cls = classes[pred_idx]
        conf = float(probs[pred_idx])

        ok = (pred_cls == true_cls)
        correct += int(ok)

        if ok:
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)

        print(f"{img_path.name:25s}  true={true_cls:15s}  pred={pred_cls:15s}  conf={conf:.3f}  {'✅' if ok else '❌'}")

    acc = correct / len(items) if items else 0.0
    avg_c = float(np.mean(conf_correct)) if conf_correct else float("nan")
    avg_w = float(np.mean(conf_wrong)) if conf_wrong else float("nan")

    print("-" * 70)
    print(f"Sanity accuracy: {acc:.3f} ({correct}/{len(items)})")
    print(f"Avg confidence (correct): {avg_c:.3f}")
    print(f"Avg confidence (wrong)  : {avg_w:.3f}")
    print("\n✅ If confidence(correct) > confidence(wrong) and accuracy is decent, model is working.")

if __name__ == "__main__":
    main()