from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.thermal_model import build_thermal_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "thermal" / "models" / "thermal_efficientnet_b0_best.pth"


class ThermalGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_index=None):
        self.model.zero_grad()

        output = self.model(image_tensor)

        if class_index is None:
            class_index = int(torch.argmax(output, dim=1).item())

        score = output[:, class_index]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probability = float(torch.softmax(output, dim=1)[0, 1].item())

        return {
            "heatmap": cam,
            "predicted_class": class_index,
            "dm_probability": probability,
        }


def load_thermal_gradcam(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_thermal_model(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Last convolutional feature layer in EfficientNet-B0.
    target_layer = model.features[-1]

    gradcam = ThermalGradCAM(model, target_layer)

    return model, gradcam, checkpoint, device


def prepare_thermal_image(image_path, image_size=224):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size))

    tensor = (
        torch.tensor(image_rgb, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    return image_rgb, tensor


def create_gradcam_overlay(image_rgb, heatmap, alpha=0.42):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image_rgb,
        1 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    return heatmap_color, overlay
