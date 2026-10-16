"""
Frozen CLIP ViT-B/16 (OpenAI weights via timm, cached locally) image embeddings for TRAIN and VAL only.
Framing identical to the EfficientNet pipeline: processed 256 px face crop -> CenterCrop 224; CLIP's own
mean/std normalisation. Optional --jpeg Q re-encodes each image at quality Q first (robustness arm).
Writes features (float16) + row index to OUT/data/clip/. Resumable per split (skips finished files).
"""
import argparse
import io
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\stage1_20261016\data\clip")
MODEL = "vit_base_patch16_clip_224.openai"


class DS(Dataset):
    def __init__(self, paths, tf, jpeg):
        self.p, self.tf, self.jpeg = paths, tf, jpeg

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(os.path.join(BASE, self.p[i])) as im:
            img = im.convert("RGB")
        if self.jpeg:
            b = io.BytesIO(); img.save(b, format="JPEG", quality=self.jpeg, subsampling=2); b.seek(0)
            with Image.open(b) as im2:
                img = im2.convert("RGB")
        return self.tf(img), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--jpeg", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    m = timm.create_model(MODEL, pretrained=True, num_classes=0).to(dev).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "local_image_path", "dataset_split"])
    for split in a.splits.split(","):
        assert split in ("train", "val"), "test split is locked"
        tag = f"{split}" + (f"_jpeg{a.jpeg}" if a.jpeg else "")
        f_out = os.path.join(OUT, f"clip_b16_{tag}.npy")
        if os.path.exists(f_out):
            print(f"[clip] {f_out} exists, skipping", flush=True)
            continue
        d = md[md.dataset_split == split].reset_index(drop=True)
        feats = np.zeros((len(d), m.num_features), dtype=np.float16)
        dl = DataLoader(DS(d.local_image_path.tolist(), tf, a.jpeg), batch_size=a.batch_size, num_workers=0)
        t0 = time.time()
        with torch.no_grad():
            for k, (x, i) in enumerate(dl):
                with torch.autocast("cuda", dtype=torch.float16):
                    f = m(x.to(dev))
                feats[i.numpy()] = f.float().cpu().numpy().astype(np.float16)
                if k % 100 == 0:
                    print(f"[clip] {tag} {k * a.batch_size}/{len(d)} {time.time() - t0:.0f}s", flush=True)
        np.save(f_out + ".tmp.npy", feats)
        os.replace(f_out + ".tmp.npy", f_out)
        d[["image_id"]].to_csv(os.path.join(OUT, f"clip_b16_{tag}_ids.csv"), index=False)
        print(f"[clip] wrote {f_out} {feats.shape} in {time.time() - t0:.0f}s; peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
