import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os


class FaceImageDataset(Dataset):
    """
    Pure vision dataset. Loads ONLY the image tensor and the binary label.

    Metadata columns (gender, age_group, image_quality, detection_difficulty,
    feat_* one-hot columns, etc.) are intentionally NOT read here. They must
    never enter the training/inference forward pass -- the original
    architecture's shortcut-learning bug came from exactly that path.
    If you need metadata for subgroup analysis, read it separately in
    evaluate.py directly from the dataframe (see `raw_df` / `attribute_cols`
    there), never through this Dataset class.

    Only `local_image_path` and `label_numeric` are copied out of `df`, into a
    plain list and a float32 array; the DataFrame itself is not kept. On
    Windows each DataLoader worker receives a pickled copy of this object, so
    holding the full metadata table here multiplied its size per worker.
    """

    def __init__(self, df: pd.DataFrame, base_dir: str, transform=None):
        self.base_dir = base_dir
        self.paths = df["local_image_path"].astype(str).tolist()
        # VeriFace AI binary label: 1 = FAKE (positive class), 0 = REAL
        self.labels = df["label_numeric"].to_numpy(dtype=np.float32, copy=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.paths[idx])

        # Context manager closes the file handle as soon as the pixels are decoded.
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)

        return image, label


class VisionFaceClassifier(nn.Module):
    """
    Pure image-based binary classifier. No tabular/metadata branch, no
    fusion head -- a single CNN backbone with a linear classification head
    producing one raw logit per image (for BCEWithLogitsLoss).

    backbone: "efficientnet_b0" (default) or "xception"
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        import timm

        valid_backbones = {"efficientnet_b0", "xception"}
        if backbone not in valid_backbones:
            raise ValueError(f"backbone must be one of {valid_backbones}, got '{backbone}'")

        # num_classes=0 -> backbone returns pooled feature embedding, no head
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),  # raw logit, no sigmoid (BCEWithLogitsLoss applies it internally)
        )

    def forward(self, img):
        features = self.backbone(img)
        logits = self.classifier(features).squeeze(1)
        return logits


class FrequencyBranch(nn.Module):
    """
    Small CNN over the 2D FFT log-magnitude spectrum of the input image.

    Diffusion/GAN synthesis leaves characteristic periodic artifacts
    (checkerboard patterns from transposed convolutions, unnatural spectral
    falloff, upsampling grid signatures) that are often much more visible in
    the frequency domain than in raw RGB. This branch gives the network an
    explicit view of that signal instead of hoping the RGB backbone learns
    it incidentally.

    Operates only on pixel data derived from the same image tensor the RGB
    branch sees -- no metadata, no extra inputs required from the Dataset.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # Always fp32: under AMP autocast, cuFFT half precision only accepts
        # power-of-two sizes (224 is not) and log-magnitudes lose precision.
        with torch.autocast(device_type=x_rgb.device.type, enabled=False):
            return self._forward_fp32(x_rgb.float())

    def _forward_fp32(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # Collapse to grayscale in whatever (normalized) pixel space the RGB
        # branch already receives -- FFT magnitude patterns from synthesis
        # artifacts survive a fixed per-channel affine normalization fine.
        gray = x_rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        spectrum = torch.fft.fftshift(torch.fft.fft2(gray))
        magnitude = torch.log1p(torch.abs(spectrum))

        # Per-image standardization so spectrum scale doesn't vary with
        # exposure/contrast -- keeps the branch focused on spectral *shape*.
        b = magnitude.shape[0]
        flat = magnitude.view(b, -1)
        mean = flat.mean(dim=1).view(b, 1, 1, 1)
        std = flat.std(dim=1).view(b, 1, 1, 1) + 1e-6
        magnitude = (magnitude - mean) / std

        feats = self.conv(magnitude).flatten(1)
        return self.proj(feats)


class DualDomainVisionClassifier(nn.Module):
    """
    RGB backbone + frequency-domain branch, fused before the classification
    head. Still a single-image input/single-logit-output model -- FaceImageDataset,
    train.py's loop, and BCEWithLogitsLoss all work unchanged.

    backbone: "efficientnet_b0" (default) or "xception"
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freq_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        import timm

        valid_backbones = {"efficientnet_b0", "xception"}
        if backbone not in valid_backbones:
            raise ValueError(f"backbone must be one of {valid_backbones}, got '{backbone}'")

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.freq_branch = FrequencyBranch(out_dim=freq_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + freq_dim, 1),
        )

    def forward(self, img):
        rgb_feats = self.backbone(img)
        freq_feats = self.freq_branch(img)
        combined = torch.cat([rgb_feats, freq_feats], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits


def build_model(model_type: str = "vision", backbone: str = "efficientnet_b0", pretrained: bool = True):
    """
    Single place that maps a checkpoint's saved `model_type` string to the
    right class, so train.py / evaluate.py / app.py never duplicate this
    branching logic and can't drift out of sync with each other.
    """
    if model_type == "vision":
        return VisionFaceClassifier(backbone=backbone, pretrained=pretrained)
    elif model_type == "dual_domain":
        return DualDomainVisionClassifier(backbone=backbone, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'vision' or 'dual_domain'.")


def group_params_for_discriminative_lr(
    model: nn.Module,
    head_lr: float,
    backbone_lr_mult: float = 0.1,
    freeze_stem: bool = True,
):
    """
    Alternative/complementary strategy to the frequency branch: instead of
    (or in addition to) adding a frequency branch, preserve the low-level
    edge/texture detectors that ImageNet pretraining already gives the early
    conv layers -- those generalize to synthesis artifacts reasonably well
    on their own -- and let only the deeper, more task-specific layers and
    the classification head adapt quickly to this dataset.

    Returns a list of param groups for torch.optim, e.g.:
        optimizer = torch.optim.AdamW(group_params_for_discriminative_lr(model, 1e-4))

    - `freeze_stem=True` sets requires_grad=False on the stem/first block
      (the earliest, most generic edge/texture filters) so they don't drift
      away from their pretrained values at all.
    - Remaining backbone layers get `head_lr * backbone_lr_mult`.
    - The classification head (and frequency branch, if present) gets the
      full `head_lr`, since those are randomly initialized and need to move
      the most.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Model has no `.backbone` attribute -- expected a timm feature extractor.")

    stem_param_names = []
    if freeze_stem:
        # timm's EfficientNet/Xception both expose an early "stem" conv +
        # norm before the main stage blocks; name prefixes below cover both.
        for name, param in backbone.named_parameters():
            if name.startswith(("conv_stem", "bn1", "stem", "conv1", "bn0")):
                param.requires_grad = False
                stem_param_names.append(name)

    backbone_params = [
        p for n, p in backbone.named_parameters()
        if p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("backbone.") and p.requires_grad
    ]

    if stem_param_names:
        print(f"Froze {len(stem_param_names)} stem parameter tensors (low-level edge/texture filters).")

    return [
        {"params": backbone_params, "lr": head_lr * backbone_lr_mult},
        {"params": head_params, "lr": head_lr},
    ]


if __name__ == "__main__":
    print("Testing vision-only model instantiation...")
    test_model = VisionFaceClassifier(backbone="efficientnet_b0", pretrained=False)
    dummy_img = torch.randn(2, 3, 224, 224)
    out = test_model(dummy_img)
    print("Forward pass successful! Output shape:", out.shape)
    assert out.shape == (2,), "Expected one logit per image, no metadata dependency."
    print("No metadata input required -- model is vision-only.")

    print("\nTesting dual-domain (RGB + frequency) model instantiation...")
    dual_model = DualDomainVisionClassifier(backbone="efficientnet_b0", pretrained=False)
    out_dual = dual_model(dummy_img)
    print("Forward pass successful! Output shape:", out_dual.shape)
    assert out_dual.shape == (2,), "Expected one logit per image."
