"""
Part 2 secondary comparison (pre-declared in PREDECLARATION.md): TEST split, one pass per model x arm,
clean and re-encoded at JPEG q50 / q35 (decode -> JPEG Q, 4:2:0 -> decode, then the standard eval transform).
fp32 like evaluate.py, so the baseline clean arm can be checked against the stored evaluate.py predictions.
Usage: eval_test_arms.py NAME=CHECKPOINT [NAME=CHECKPOINT ...]
Writes data/test_arms/<NAME>_<arm>.csv (resumable: existing files are skipped, never overwritten).
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
OUT = os.path.join(BASE, r"research_runs\part2_final_20261016\data\test_arms")
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
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"),
                     usecols=["image_id", "local_image_path", "label_numeric", "dataset_split", "source_dataset",
                              "generator", "generator_family", "group_id"])
    te = md[md.dataset_split == "test"].reset_index(drop=True)
    assert len(te) == 16635
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
            t0 = time.time(); z = np.zeros(len(te))
            with torch.no_grad():
                for x, i in DataLoader(DS(te.local_image_path.tolist(), q), batch_size=64, num_workers=0):
                    z[i.numpy()] = m(x.to(dev)).float().cpu().numpy().reshape(-1)
            o = te[["image_id", "source_dataset", "generator", "generator_family", "label_numeric", "group_id"]].copy()
            o["logit"] = z; o["prob_fake"] = 1 / (1 + np.exp(-z))
            o.to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)
            print(f"[test_arms] {name} {arm} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
