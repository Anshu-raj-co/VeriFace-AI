"""
VeriFace AI -- single-image inference shared by app.py and tests.

Reproduces the evaluation pipeline exactly:
  PIL RGB -> face_utils.crop_face_square (256 px square face crop, the function preprocess_data.py used)
  -> train.get_eval_transforms (CenterCrop 224, ToTensor, ImageNet normalisation) -> model -> logit.
Label convention: FAKE = 1, sigmoid(logit) = model score for FAKE.

Decision policy (validation-derived, see research_runs/stage1_20261016/reports/app_policy.json):
  score >= FAKE_AT_OR_ABOVE  -> "LIKELY MANIPULATED"  (val: 1.0 % of genuine faces land here)
  score <  REAL_BELOW        -> "LIKELY AUTHENTIC"    (val: 0.7 % of fakes land here)
  otherwise                  -> "INCONCLUSIVE"        (val: 4.8 % of images)
The displayed score is temperature-scaled (T fitted on validation) -- a calibrated *model score* on
data like the training sources, not a guaranteed probability that a given photo is fake.
"""
import os

import numpy as np
import torch
from PIL import Image

from face_utils import crop_face_square
from model import build_model
from train import get_eval_transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_face_model.pth")
POSITIVE_CLASS = "FAKE"

# Baseline epoch-7 model, fitted on the validation split only (stage-1 research report).
REAL_BELOW = 0.30
FAKE_AT_OR_ABOVE = 0.90
TEMPERATURE = 1.5523

_EVAL_TRANSFORM = get_eval_transforms()


def load_checkpoint(path: str = MODEL_PATH, device: torch.device = torch.device("cpu")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at '{path}'. Run train.py first.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError("Unsupported checkpoint format (expected a dict with 'state_dict').")
    if ckpt.get("positive_class") != POSITIVE_CLASS:
        # Scoring a REAL=1 checkpoint as P(FAKE) would silently invert every verdict.
        raise ValueError(f"Checkpoint positive_class={ckpt.get('positive_class')!r}, expected {POSITIVE_CLASS!r}.")
    model = build_model(model_type=ckpt.get("model_type", "vision"), backbone=ckpt.get("backbone", "efficientnet_b0"),
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: ckpt.get(k) for k in ("backbone", "model_type", "epoch", "positive_class", "input_size")}
    return model, meta


def preprocess(image: Image.Image):
    """Returns (tensor 1x3x224x224, model_view PIL 224x224, face_crop PIL 256x256, crop_info)."""
    face_crop, info = crop_face_square(image.convert("RGB"), out_size=256)
    tensor = _EVAL_TRANSFORM(face_crop).unsqueeze(0)
    w, h = face_crop.size
    left, top = (w - 224) // 2, (h - 224) // 2
    model_view = face_crop.crop((left, top, left + 224, top + 224))
    return tensor, model_view, face_crop, info


def decide(raw_score: float) -> str:
    if raw_score >= FAKE_AT_OR_ABOVE:
        return "LIKELY MANIPULATED"
    if raw_score < REAL_BELOW:
        return "LIKELY AUTHENTIC"
    return "INCONCLUSIVE"


def calibrated_score(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-logit / TEMPERATURE)))


@torch.no_grad()
def predict(model, image: Image.Image, device: torch.device = torch.device("cpu")) -> dict:
    tensor, model_view, face_crop, info = preprocess(image)
    logit = float(model(tensor.to(device)).float().item())
    raw = float(1.0 / (1.0 + np.exp(-logit)))
    return {"logit": logit, "raw_score_fake": raw, "calibrated_score_fake": calibrated_score(logit),
            "verdict": decide(raw), "crop_info": info, "model_view": model_view, "face_crop": face_crop,
            "tensor": tensor}
