"""
VeriFace AI -- evaluation protocol.

Label convention: FAKE is the positive class (label_numeric = 1, model output
sigmoid(logit) = P(FAKE)); REAL = 0.

Reports, for the chosen split:
  - threshold-free metrics: ROC-AUC, PR-AUC (average precision, FAKE positive),
    Equal Error Rate (EER) and the threshold where it occurs
  - thresholded metrics at 0.5 and at the EER threshold
  - a per-dataset / per-generator breakdown. Every generator is single-class,
    so each one gets its own detection rate (FAKE generators: share flagged
    FAKE; REAL sources: share correctly passed as REAL) and, for FAKE
    generators, ROC-AUC / EER of that generator against ALL real images in
    the split -- i.e. "how separable is StyleGAN vs real", "Stable Diffusion
    vs real", and so on.

Usage:
    python evaluate.py                      # test split, best_face_model.pth
    python evaluate.py --split val --checkpoint other.pth
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FaceImageDataset, build_model
from train import get_eval_transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_metadata.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_face_model.pth")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation_reports")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSITIVE_CLASS = "FAKE"

# Columns used ONLY to slice predictions after inference; never model input.
BREAKDOWN_COLS = ["source_dataset", "generator_family", "generator"]


def log(msg: str = ""):
    print(f"[VeriFace AI] {msg}", flush=True)


def load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Support the current checkpoint format ({"backbone", "model_type", "state_dict", ...}),
    # the previous one ({"backbone", "state_dict"} -> vision-only), and a bare state_dict.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        backbone = checkpoint.get("backbone", "efficientnet_b0")
        model_type = checkpoint.get("model_type", "vision")
        state_dict = checkpoint["state_dict"]
        positive = checkpoint.get("positive_class")
    else:
        backbone, model_type, state_dict, positive = "efficientnet_b0", "vision", checkpoint, None

    if positive != POSITIVE_CLASS:
        # Pre-Phase-1 checkpoints were trained with REAL = 1; scoring them as
        # P(FAKE) silently inverts every metric.
        raise SystemExit(
            f"Checkpoint positive_class={positive!r}, expected {POSITIVE_CLASS!r}. "
            "It predates the VeriFace AI FAKE=1 convention -- retrain with train.py."
        )

    model = build_model(model_type=model_type, backbone=backbone, pretrained=False).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_type, backbone


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray):
    """EER = point where FPR (REAL flagged FAKE) == FNR (FAKE passed as REAL)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    # Linear interpolation between the two closest ROC points.
    if i + 1 < len(fpr) and (fpr[i] - fnr[i]) * (fpr[i + 1] - fnr[i + 1]) < 0:
        d0, d1 = fpr[i] - fnr[i], fpr[i + 1] - fnr[i + 1]
        t = d0 / (d0 - d1)
        eer = fpr[i] + t * (fpr[i + 1] - fpr[i])
    else:
        eer = (fpr[i] + fnr[i]) / 2
    return float(eer), float(min(thr[i], 1.0))


def thresholded(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision_fake": precision_score(y_true, pred, zero_division=0),
        "recall_fake (TPR)": recall_score(y_true, pred, zero_division=0),
        "f1_fake": f1_score(y_true, pred, zero_division=0),
        "false_positive_rate (REAL flagged)": fp / max(1, fp + tn),
        "confusion [tn, fp, fn, tp]": [int(tn), int(fp), int(fn), int(tp)],
    }


def breakdown(df: pd.DataFrame, col: str, eer_thr: float) -> pd.DataFrame:
    """Per-group detection rate; FAKE groups also get AUC/EER vs every REAL image."""
    real_scores = df.loc[df["label_numeric"] == 0, "prob_fake"].to_numpy()
    rows = []
    for (val, label), sub in df.groupby([col, "label"]):
        s = sub["prob_fake"].to_numpy()
        is_fake = label == "FAKE"
        row = {
            col: val, "label": label, "n": len(sub),
            "detection_rate@0.5": float((s >= 0.5).mean() if is_fake else (s < 0.5).mean()),
            "detection_rate@eer": float((s >= eer_thr).mean() if is_fake else (s < eer_thr).mean()),
            "mean_prob_fake": float(s.mean()),
            "auc_vs_all_real": np.nan, "eer_vs_all_real": np.nan,
        }
        if is_fake and len(real_scores):
            y = np.r_[np.zeros(len(real_scores)), np.ones(len(s))]
            sc = np.r_[real_scores, s]
            row["auc_vs_all_real"] = roc_auc_score(y, sc)
            row["eer_vs_all_real"] = equal_error_rate(y, sc)[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["label", col], ascending=[False, True])


def main():
    ap = argparse.ArgumentParser(description="VeriFace AI -- evaluation (FAKE = positive class)")
    ap.add_argument("--checkpoint", default=MODEL_PATH)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    log("=" * 72)
    log(f"VeriFace AI -- evaluation on '{args.split}' split ({DEVICE})")
    log("=" * 72)

    df = pd.read_csv(args.csv)
    eval_df = df[df["dataset_split"] == args.split].reset_index(drop=True)
    log(f"Samples: {len(eval_df):,} (REAL {int((eval_df.label_numeric == 0).sum()):,} | "
        f"FAKE {int((eval_df.label_numeric == 1).sum()):,})")

    # Identical framing to training/validation (center 224 crop of the 256 face crop).
    loader = DataLoader(FaceImageDataset(eval_df, BASE_DIR, transform=get_eval_transforms()),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, model_type, backbone = load_model(args.checkpoint)
    log(f"Model: {model_type} / {backbone} from {args.checkpoint}")

    probs = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="[VeriFace AI] inference"):
            probs.append(torch.sigmoid(model(imgs.to(DEVICE))).cpu().numpy())
    eval_df["prob_fake"] = np.concatenate(probs)
    y, s = eval_df["label_numeric"].to_numpy(), eval_df["prob_fake"].to_numpy()

    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)
    eer, eer_thr = equal_error_rate(y, s)
    at_05 = thresholded(y, s, 0.5)
    at_eer = thresholded(y, s, eer_thr)

    log("")
    log("PRIMARY METRICS (positive class = FAKE)")
    log(f"  ROC-AUC : {roc_auc:.4f}")
    log(f"  PR-AUC  : {pr_auc:.4f}   (FAKE prevalence {y.mean():.3f})")
    log(f"  EER     : {eer * 100:.2f}%  at threshold {eer_thr:.4f}")
    for name, m in (("@ 0.5", at_05), ("@ EER threshold", at_eer)):
        log(f"  {name}: " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    tables = {}
    for col in BREAKDOWN_COLS:
        t = breakdown(eval_df, col, eer_thr)
        tables[col] = t
        log("")
        log(f"BREAKDOWN BY {col.upper()}  (REAL rows: share passed as REAL; FAKE rows: share flagged FAKE)")
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
            for line in t.to_string(index=False).splitlines():
                log("  " + line)

    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    eval_df[["image_id", "local_image_path", "label", "label_numeric", "source_dataset", "generator",
             "generator_family", "prob_fake"]].to_csv(os.path.join(REPORT_DIR, f"predictions_{args.split}_{stamp}.csv"), index=False)
    report = {
        "project": "VeriFace AI", "generated": stamp, "split": args.split, "checkpoint": args.checkpoint,
        "model_type": model_type, "backbone": backbone, "positive_class": POSITIVE_CLASS, "n": int(len(eval_df)),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "eer": eer, "eer_threshold": eer_thr,
        "at_0.5": at_05, "at_eer_threshold": at_eer,
        "breakdown": {c: t.replace({np.nan: None}).to_dict(orient="records") for c, t in tables.items()},
    }
    out = os.path.join(REPORT_DIR, f"evaluation_{args.split}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    log("")
    log(f"Report + per-image predictions written to {os.path.relpath(REPORT_DIR, BASE_DIR)}/")


if __name__ == "__main__":
    main()
