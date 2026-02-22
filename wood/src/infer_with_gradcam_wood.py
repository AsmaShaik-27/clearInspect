import sys
import os
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image

# =========================
# CHECK INPUT
# =========================
if len(sys.argv) != 2:
    print("Usage: python src/infer_with_gradcam_wood.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print("Image not found:", IMAGE_PATH)
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
class_names = ['color', 'combined', 'good', 'hole', 'liquid', 'scratch']

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/wood_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Target layer for GradCAM
target_layer = model.layer4[-1]

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# =========================
# HOOKS FOR GRADCAM
# =========================
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# =========================
# FORWARD PASS
# =========================
output = model(input_tensor)
probs = F.softmax(output, dim=1)
confidence, pred_class = torch.max(probs, 1)

predicted_label = class_names[pred_class.item()]
confidence_score = float(confidence.item())

# =========================
# BACKWARD FOR GRADCAM
# =========================
model.zero_grad()
output[0, pred_class].backward()

# Compute weights
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# Weight activations
for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = torch.relu(heatmap)
heatmap = heatmap.cpu().detach().numpy()

# Normalize
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
heatmap = cv2.resize(heatmap, (224, 224))

# =========================
# OVERLAY
# =========================
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

original = cv2.imread(IMAGE_PATH)
original = cv2.resize(original, (224, 224))

overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

# =========================
# SAVE OUTPUT
# =========================
os.makedirs("output", exist_ok=True)

filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_image_path = f"output/{filename}_gradcam.jpg"

cv2.imwrite(output_image_path, overlay)

# =========================
# JSON OUTPUT
# =========================
result = {
    "input_image": IMAGE_PATH,
    "output_image": output_image_path,
    "predicted_class": predicted_label,
    "confidence": confidence_score
}

json_path = f"output/{filename}_result.json"

with open(json_path, "w") as f:
    json.dump(result, f, indent=4)

print("Prediction:", predicted_label)
print("Confidence:", confidence_score)
print("Saved image:", output_image_path)
print("Saved JSON:", json_path)