"""
S1-B evaluation: full VALIDATION split, clean and re-encoded at JPEG q50 / q35 (decode -> JPEG Q -> decode,
then the standard eval transform). Usage: eval_jpeg_val.py NAME=CHECKPOINT [NAME=CHECKPOINT ...]
Writes data/jpeg_eval/<NAME>_<arm>.csv (resumable: existing files are skipped).
"""
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\stage1_20261016\data\jpeg_eval")
sys.path.insert(0, BASE)
from model import build_model  # noqa: E402
from train import get_eval_transforms  # noqa: E402

ARMS = {"clean": 0, "q50": 50, "q35": 35}


class DS(Dataset):
    def __init__(self, paths, q):
        self.p, self.q, self.t = paths, q, get_eval_transforms()

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(os.path.join(BASE, self.p[i])) as im:
            img = im.convert("RGB")
        if self.q:
            b = io.BytesIO(); img.save(b, format="JPEG", quality=self.q, subsampling=2); b.seek(0)
            with Image.open(b) as im2:
                img = im2.convert("RGB")
        return self.t(img), i


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "local_image_path", "label_numeric",
                                                                            "dataset_split", "generator", "group_id"])
    va = md[md.dataset_split == "val"].reset_index(drop=True)
    for spec in sys.argv[1:]:
        name, ck_path = spec.split("=", 1)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        assert ck.get("positive_class") == "FAKE"
        m = build_model(model_type=ck["model_type"], backbone=ck["backbone"], pretrained=False)
        m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
        for arm, q in ARMS.items():
            f = os.path.join(OUT, f"{name}_{arm}.csv")
            if os.path.exists(f):
                continue
            t0 = time.time(); z = np.zeros(len(va))
            with torch.no_grad():
                for x, i in DataLoader(DS(va.local_image_path.tolist(), q), batch_size=64, num_workers=0):
                    with torch.autocast("cuda", dtype=torch.float16):
                        z[i.numpy()] = m(x.to(dev)).float().cpu().numpy()
            o = va[["image_id", "generator", "label_numeric", "group_id"]].copy(); o["logit"] = z; o["prob_fake"] = 1 / (1 + np.exp(-z))
            o.to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)
            print(f"[jpeg_eval] {name} {arm} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
