# src/train_multiclass.py
# Train a 4-class MobileNetV2 on your splits:
# data/processed/bottle_splits/{train,val}/{good,broken_large,broken_small,contamination}/
#
# Run:
#   python src/train_multiclass.py
#
# Output:
#   outputs/models/best_mobilenetv2_multiclass.pth  (best val accuracy checkpoint)

from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.utils.class_weight import compute_class_weight

# -------------------------
# Config
# -------------------------
DATA = Path("data/processed/bottle_splits")
OUT  = Path("outputs/models")
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16          # small dataset -> smaller batch is safer
EPOCHS_HEAD = 12         # train classifier head
EPOCHS_FINE = 18         # fine-tune last blocks
LR_HEAD = 1e-3
LR_FINE = 2e-4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------------
# Transforms (conveyor-like)
# -------------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15)],
        p=0.85
    ),
    transforms.RandomRotation(7),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# -------------------------
# Datasets / Loaders
# -------------------------
train_dir = DATA / "train"
val_dir   = DATA / "val"

if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(
        "Expected splits at data/processed/bottle_splits. "
        "Run: python src/make_splits.py"
    )

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = datasets.ImageFolder(val_dir, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

# -------------------------
# Class weights (imbalance)
# -------------------------
y = np.array([lbl for _, lbl in train_ds.samples])
cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y)
cw = torch.tensor(cw, dtype=torch.float32).to(device)
print("Class weights:", cw)

# -------------------------
# Model
# -------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=cw)

# -------------------------
# Eval
# -------------------------
def evaluate():
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, yb in val_loader:
            x, yb = x.to(device), yb.to(device)
            logits = model(x)
            loss = criterion(logits, yb)

            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += x.size(0)

    return (loss_sum / total) if total else 0.0, (correct / total) if total else 0.0

# -------------------------
# Train loop
# -------------------------
def train_loop(optimizer, epochs, best_path: Path):
    best_acc = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for x, yb in pbar:
            x, yb = x.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

        val_loss, val_acc = evaluate()
        print(f"Epoch {ep:02d}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": class_names}, best_path)
            print("  ✅ saved:", best_path)

# -------------------------
# Phase 1: Train head only
# -------------------------
best_path = OUT / "best_mobilenetv2_multiclass.pth"

for p in model.features.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD)
print("\nPhase 1: training classifier head")
train_loop(optimizer, EPOCHS_HEAD, best_path)

# -------------------------
# Phase 2: Fine-tune last blocks
# -------------------------
for p in model.features.parameters():
    p.requires_grad = True

# freeze early blocks; keep last ~4 trainable
for i, block in enumerate(model.features):
    if i < len(model.features) - 4:
        for p in block.parameters():
            p.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE)
print("\nPhase 2: fine-tuning last blocks")
train_loop(optimizer, EPOCHS_FINE, best_path)

print("\n✅ Done. Best checkpoint:", best_path)
