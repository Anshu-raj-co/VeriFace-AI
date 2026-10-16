"""
VeriFace AI research -- inference with the exact evaluation preprocessing
(train.get_eval_transforms: CenterCrop 224 of the 256 face crop, ImageNet norm),
optionally after a controlled counterfactual degradation of the processed image.

Never touches the test split unless --split test is given explicitly (it is not,
in this study, after the locked baseline evaluation). Writes only to --out.

Perturbation arms (applied to the 256x256 processed JPEG before eval transforms):
  none        -- the stored file as-is
  reencode    -- decode -> JPEG at the image's own pipeline quality -> decode (control for double compression)
  down120     -- INTER_AREA down to 120 px, INTER_CUBIC back to 256 (mimics FF++ small-face upscaling), then reencode
  down160     -- same with 160 px
  jpeg35      -- JPEG q=35 (heavy, codec-like blocking), then reencode at pipeline quality
"""
import argparse
import io
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE = r"D:\PROJECT\VeriFace AI"
sys.path.insert(0, BASE)
from model import build_model  # noqa: E402
from train import get_eval_transforms  # noqa: E402

ARMS = ["none", "reencode", "down120", "down160", "jpeg35"]


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q), subsampling=2)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def perturb(img, arm, q):
    if arm == "none":
        return img
    if arm in ("down120", "down160"):
        side = 120 if arm == "down120" else 160
        a = np.asarray(img)
        a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(a)
    elif arm == "jpeg35":
        img = jpeg(img, 35)
    return jpeg(img, q)


class DS(Dataset):
    def __init__(self, df, arm):
        self.p = df["local_image_path"].tolist()
        self.q = df["jpeg_quality"].tolist()
        self.arm = arm
        self.t = get_eval_transforms()

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(os.path.join(BASE, self.p[i])) as im:
            img = im.convert("RGB")
        return self.t(perturb(img, self.arm, self.q[i])), i


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ck.get("positive_class") != "FAKE":
        raise SystemExit(f"{ckpt_path}: positive_class={ck.get('positive_class')!r}, expected 'FAKE'")
    m = build_model(model_type=ck.get("model_type", "vision"), backbone=ck.get("backbone", "efficientnet_b0"),
                    pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--rows", default=None, help="optional CSV of image_id to restrict to (and its order)")
    ap.add_argument("--arms", default="none", help=f"comma list from {ARMS}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    if a.split == "test" and os.environ.get("VERIFACE_ALLOW_TEST") != "1":
        raise SystemExit("test split is locked for this study (set VERIFACE_ALLOW_TEST=1 only for the final confirmation).")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"),
                     usecols=["image_id", "local_image_path", "label_numeric", "dataset_split", "generator",
                              "generator_family", "source_dataset", "group_id", "jpeg_quality", "upscaled", "crop_side"])
    df = md[md.dataset_split == a.split]
    if a.rows:
        ids = pd.read_csv(a.rows)["image_id"]
        df = df.set_index("image_id").loc[ids].reset_index()
    df = df.reset_index(drop=True)
    model, ck = load(a.checkpoint, dev)
    print(f"[infer] {a.checkpoint} epoch={ck.get('epoch')} split={a.split} n={len(df)} arms={arms} device={dev}", flush=True)
    outs = []
    t0 = time.time()
    for arm in arms:
        dl = DataLoader(DS(df, arm), batch_size=a.batch_size, shuffle=False, num_workers=0)
        probs = np.zeros(len(df), dtype=np.float64)
        logits = np.zeros(len(df), dtype=np.float64)
        with torch.no_grad():
            for x, idx in dl:
                lg = model(x.to(dev)).float().cpu().numpy()
                logits[idx.numpy()] = lg
                probs[idx.numpy()] = 1 / (1 + np.exp(-lg))
        o = df.copy()
        o["arm"] = arm
        o["logit"] = logits
        o["prob_fake"] = probs
        outs.append(o)
        print(f"[infer] arm {arm} done ({time.time() - t0:.0f}s) mean p={probs.mean():.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.concat(outs).to_csv(a.out, index=False)
    if dev.type == "cuda":
        print(f"[infer] peak VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"[infer] wrote {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
