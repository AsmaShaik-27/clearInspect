# src/gradcam_utils.py
# Grad-CAM + helpers:
# - overlay_cam_on_image: standard overlay
# - cam_peakiness: measure how "concentrated" the CAM is
# - cam_to_colormap: heatmap-only image (useful to make GOOD images fully blue by passing zeros)
# - bottle_zone_explanation: operator-friendly zone explanation

from __future__ import annotations
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Minimal Grad-CAM for CNNs.

    Example:
      cam_engine = GradCAM(model, target_layer)
      cam = cam_engine(x, class_idx)  # returns HxW in [0,1]
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
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

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        grads = self.gradients   # [1,C,h,w]
        acts = self.activations  # [1,C,h,w]

        # channel importance
        weights = grads.mean(dim=(2, 3), keepdim=True)      # [1,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)     # [1,1,h,w]
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()          # [h,w]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam_on_image(rgb_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Standard overlay of CAM on RGB image.
    rgb_img: HxWx3 uint8 RGB
    cam: HxW float [0,1]
    """
    h, w = rgb_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heat = (cam_resized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay


def cam_to_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Heatmap-only image (no overlay). If cam is all zeros, output will be fully blue (JET colormap).
    cam: HxW float [0,1]
    returns: HxWx3 uint8 RGB heatmap
    """
    heat = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def cam_peakiness(cam: np.ndarray) -> float:
    """
    Returns how "concentrated" CAM is.
    Higher => sharper hotspot (more defect-like), lower => diffuse attention (often good-like).
    """
    p95 = float(np.percentile(cam, 95))
    mean = float(cam.mean())
    return p95 - mean


def bottle_zone_explanation(cam: np.ndarray) -> str:
    """
    Simple bottle-specific zones using vertical bands + left/center/right.
    """
    h, w = cam.shape

    def band_mean(y0, y1):
        return float(cam[int(y0*h):int(y1*h), :].mean())

    cap = band_mean(0.00, 0.25)
    body = band_mean(0.25, 0.80)
    base = band_mean(0.80, 1.00)

    left = float(cam[:, :w//3].mean())
    center = float(cam[:, w//3:2*w//3].mean())
    right = float(cam[:, 2*w//3:].mean())

    zones = {"cap/neck": cap, "body/label": body, "base": base}
    horiz = {"left": left, "center": center, "right": right}

    top_zone = max(zones, key=zones.get)
    top_side = max(horiz, key=horiz.get)

    return (
        f"Hotspot concentrated in {top_zone} region and {top_side} area "
        f"(cap={cap:.3f}, body={body:.3f}, base={base:.3f})."
    )
