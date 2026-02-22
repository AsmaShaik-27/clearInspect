import io
import base64
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
from PIL import Image

app = FastAPI()

# 🔥 Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['color', 'combined', 'good', 'hole', 'liquid', 'scratch']

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/wood_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

target_layer = model.layer4[-1]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_gradcam(input_tensor):
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    confidence, pred_class = torch.max(probs, 1)

    model.zero_grad()
    output[0, pred_class].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.cpu().detach().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (224, 224))

    handle_fwd.remove()
    handle_bwd.remove()

    return heatmap, class_names[pred_class.item()], float(confidence.item())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    heatmap, predicted_class, confidence = generate_gradcam(input_tensor)

    image_np = np.array(image.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "heatmap_image": heatmap_base64
    }