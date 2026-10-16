"""
VeriFace AI -- Phase 1 standardised preprocessing.

Every image (REAL and FAKE, Kaggle 140k and GRAVEX-200K) goes through the
identical pipeline so that resolution, framing, file format and compression
history stop being label-correlated:

  1. Face detection + square crop  (face_utils.crop_face_square)
  2. Resize to --size x --size (default 256), no padding
  3. Re-encode as baseline JPEG, 4:2:0 chroma, no EXIF, with a quality factor
     drawn uniformly from [60, 100] -- deterministic per image_id, so reruns
     reproduce byte-identical files.

Input : cleaned_metadata.csv as written by audit_dataset.py (raw paths).
Output: dataset_processed/<split>/<label>/<image_id>.jpg, and
        cleaned_metadata.csv rewritten with local_image_path pointing at the
        processed files plus per-image pipeline columns. A second shortcut
        probe then checks that the processed files carry no metadata signal.

Usage:
    python preprocess_data.py               # full run
    python preprocess_data.py --limit 2000  # smoke test; leaves cleaned_metadata.csv untouched
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import zlib
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from face_utils import crop_face_square

cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "dataset_processed")
REPORT_DIR = os.path.join(BASE_DIR, "audit_reports")

SEED = 42
JPEG_Q_RANGE = (60, 100)  # inclusive


def log(msg: str = ""):
    print(f"[VeriFace AI] {msg}", flush=True)


def jpeg_quality_for(image_id: str, seed: int) -> int:
    rng = np.random.default_rng([seed, zlib.crc32(image_id.encode())])
    return int(rng.integers(JPEG_Q_RANGE[0], JPEG_Q_RANGE[1] + 1))


def _process_one(task):
    image_id, src, dst, size, seed = task
    try:
        with Image.open(src) as im:
            face, info = crop_face_square(im, out_size=size)
        q = jpeg_quality_for(image_id, seed)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        face.save(dst, format="JPEG", quality=q, subsampling=2, optimize=False)
        return dict(image_id=image_id, face_detected=info["face_detected"], num_faces=info["num_faces"],
                    crop_side=info["crop_side"], upscaled=info["upscaled"], jpeg_quality=q,
                    proc_file_size=os.path.getsize(dst), proc_error="")
    except Exception as e:  # noqa: BLE001 -- record and keep going
        return dict(image_id=image_id, proc_error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="VeriFace AI -- standardised face-crop preprocessing")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="process a stratified sample only (smoke test)")
    args = ap.parse_args()
    t0 = time.time()

    log("=" * 72)
    log("VeriFace AI -- standardised preprocessing")
    log("=" * 72)
    df = pd.read_csv(CSV_PATH)
    if "raw_path" not in df.columns:
        raise SystemExit("cleaned_metadata.csv has no raw_path column -- run audit_dataset.py first.")

    if args.limit:
        rng = np.random.default_rng(args.seed)
        keep = [rng.choice(ix, size=min(len(ix), max(1, args.limit * len(ix) // len(df))), replace=False)
                for ix in df.groupby(["generator", "dataset_split"]).indices.values()]
        df = df.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
        log(f"--limit: processing a stratified sample of {len(df):,} images")

    df["local_image_path"] = [
        f"dataset_processed/{s}/{l.lower()}/{i}.jpg" for s, l, i in zip(df["dataset_split"], df["label"], df["image_id"])
    ]
    tasks = [(i, os.path.join(BASE_DIR, r), os.path.join(BASE_DIR, p), args.size, args.seed)
             for i, r, p in zip(df["image_id"], df["raw_path"], df["local_image_path"])]

    log(f"Processing {len(tasks):,} images -> {args.size}x{args.size} face crops, "
        f"JPEG q~U{list(JPEG_Q_RANGE)}, {args.workers} workers")
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(_process_one, tasks, chunksize=64), total=len(tasks),
                            desc="[VeriFace AI] preprocessing", mininterval=10))
    res = pd.DataFrame(results)
    df = df.drop(columns=[c for c in res.columns if c != "image_id" and c in df.columns]).merge(res, on="image_id")

    failed = df[df["proc_error"].fillna("") != ""]
    if len(failed):
        log(f"WARNING: {len(failed):,} images failed and are dropped (see audit_reports/preprocess_failures.csv)")
        failed[["image_id", "raw_path", "proc_error"]].to_csv(os.path.join(REPORT_DIR, "preprocess_failures.csv"), index=False)
    df = df[df["proc_error"].fillna("") == ""].drop(columns=["proc_error"]).reset_index(drop=True)
    df["proc_width"] = df["proc_height"] = args.size
    df["proc_aspect_ratio"] = 1.0

    # Face-detection coverage: a large REAL/FAKE gap here means framing still differs by class.
    det = (df.groupby(["generator", "label"])
             .agg(n=("image_id", "size"), face_detected=("face_detected", "mean"), upscaled=("upscaled", "mean"),
                  median_crop_side=("crop_side", "median"), median_kb=("proc_file_size", lambda s: s.median() / 1024))
             .reset_index())
    log("Per-generator pipeline stats:")
    for r in det.itertuples():
        log(f"  {r.generator:<24} {r.label:<4} n={r.n:>7,}  face={r.face_detected:6.1%}  "
            f"upscaled={r.upscaled:6.1%}  crop_side~{r.median_crop_side:>5.0f}px  {r.median_kb:5.1f} KB")

    # Post-preprocessing shortcut probe
    from audit_dataset import print_probe, run_shortcut_probe
    df["face_detected_f"] = df["face_detected"].astype(float)
    df["upscaled_f"] = df["upscaled"].astype(float)
    probe = run_shortcut_probe(df, {
        "processed file (w, h, size, aspect)": ["proc_width", "proc_height", "proc_file_size", "proc_aspect_ratio"],
        "JPEG quality factor": ["jpeg_quality"],
        "crop trace (face found, crop side, upscaled)": ["face_detected_f", "crop_side", "upscaled_f"],
    })
    log("Shortcut probe on PROCESSED images (train -> test):")
    print_probe(probe)
    df = df.drop(columns=["face_detected_f", "upscaled_f"])

    out_csv = os.path.join(REPORT_DIR, "preprocess_sample.csv") if args.limit else CSV_PATH
    df.to_csv(out_csv, index=False)
    with open(os.path.join(REPORT_DIR, "preprocess_report.json" if not args.limit else "preprocess_sample_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "jpeg_q_range": JPEG_Q_RANGE,
                   "n_processed": int(len(df)), "n_failed": int(len(failed)),
                   "per_generator": det.to_dict(orient="records"), "probe": probe}, fh, indent=2, default=str)
    log(f"Wrote {os.path.relpath(out_csv, BASE_DIR)} ({len(df):,} rows) in {time.time() - t0:.0f}s")
    if not args.limit:
        log("Next: python train.py")


if __name__ == "__main__":
    main()
