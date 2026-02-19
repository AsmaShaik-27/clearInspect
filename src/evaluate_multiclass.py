from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

DATA = Path("data/processed/bottle_splits")
CKPT = Path("outputs/models/best_mobilenetv2_multiclass.pth")

IMG_SIZE = 224
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_ds = datasets.ImageFolder(DATA / "test", transform=tfm)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

ckpt = torch.load(CKPT, map_location=device)
classes = ckpt["classes"]
print("Classes:", classes)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())

print("\nClassification report (macro-F1 matters most here):")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))
