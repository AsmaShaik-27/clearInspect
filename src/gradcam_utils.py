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
