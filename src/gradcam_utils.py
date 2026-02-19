import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx: int):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        grads = self.gradients          # [B,C,H,W]
        acts  = self.activations        # [B,C,H,W]

        weights = grads.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam_on_image(rgb_img: np.ndarray, cam: np.ndarray, alpha=0.45):
    """
    rgb_img: uint8 RGB image HxWx3
    cam: float heatmap HxW in [0,1]
    """
    h, w = rgb_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay

def bottle_zone_explanation(cam: np.ndarray) -> str:
    """
    Simple bottle-specific zones using vertical bands:
      - cap/neck: top 0-25%
      - body/label: 25-80%
      - base: 80-100%
    Also reports whether hotspot is centered or edge.
    """
    h, w = cam.shape

    def band_mean(y0, y1):
        return float(cam[int(y0*h):int(y1*h), :].mean())

    cap = band_mean(0.00, 0.25)
    body = band_mean(0.25, 0.80)
    base = band_mean(0.80, 1.00)

    # left/center/right
    left = float(cam[:, :w//3].mean())
    center = float(cam[:, w//3:2*w//3].mean())
    right = float(cam[:, 2*w//3:].mean())

    zones = {"cap/neck": cap, "body/label": body, "base": base}
    horiz = {"left": left, "center": center, "right": right}

    top_zone = max(zones, key=zones.get)
    top_side = max(horiz, key=horiz.get)

    return (
        f"Hotspot concentrated in **{top_zone}** region and **{top_side}** area "
        f"(cap={cap:.3f}, body={body:.3f}, base={base:.3f})."
    )
def cam_peakiness(cam: np.ndarray) -> float:
    # Higher when activation is concentrated (defect-like), lower when diffuse (good-like)
    p95 = float(np.percentile(cam, 95))
    mean = float(cam.mean())
    return p95 - mean
