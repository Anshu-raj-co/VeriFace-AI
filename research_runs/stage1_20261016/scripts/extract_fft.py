"""
Spectral features for the FFT probe (S1-D), TRAIN + VAL only, CPU.
Processed 256 px face crop -> CenterCrop 224 -> grayscale float -> Hann window -> 2-D FFT -> log power ->
azimuthal average over 112 radial bins, plus 4 band-energy ratios (high-frequency share) = 116 features.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\stage1_20261016\data\fft")
N = 224
_y, _x = np.indices((N, N)) - N // 2
R = np.sqrt(_x ** 2 + _y ** 2).astype(np.int32)
R[R >= N // 2] = N // 2 - 1
COUNTS = np.bincount(R.ravel(), minlength=N // 2)
WIN = np.outer(np.hanning(N), np.hanning(N))


def feat(path):
    with Image.open(os.path.join(BASE, path)) as im:
        g = np.asarray(im.convert("L"), dtype=np.float32)
    o = (g.shape[0] - N) // 2
    g = g[o:o + N, o:o + N]
    g = (g - g.mean()) * WIN
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    radial = np.bincount(R.ravel(), weights=p.ravel(), minlength=N // 2) / COUNTS
    lr = np.log1p(radial)
    tot = radial.sum()
    bands = [radial[a:b].sum() / tot for a, b in [(28, 56), (56, 84), (84, 100), (100, 112)]]
    return np.r_[lr, np.log(np.asarray(bands) + 1e-12)].astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "local_image_path", "dataset_split"])
    for split in ["val", "train"]:
        f = os.path.join(OUT, f"fft_{split}.npy")
        if os.path.exists(f):
            continue
        d = md[md.dataset_split == split].reset_index(drop=True)
        t0 = time.time()
        with Pool(4) as pool:
            X = np.stack(pool.map(feat, d.local_image_path.tolist(), chunksize=256))
        np.save(f, X); d[["image_id"]].to_csv(os.path.join(OUT, f"fft_{split}_ids.csv"), index=False)
        print(f"[fft] {split} {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
