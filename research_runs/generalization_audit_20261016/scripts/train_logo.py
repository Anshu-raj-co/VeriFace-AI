"""
VeriFace AI research -- leave-generator-out (LOGO) training for the Phase C unseen-generator test.
Derived from finetune.py; differences: --init imagenet trains from ImageNet weights (timm cache), and
--exclude-generators removes those generators from TRAIN (validation keeps them, so they are unseen at
evaluation). Model selection uses the final epoch only (no test, no held-out-generator tuning).

Original finetune.py docstring follows.

Starts from a frozen baseline checkpoint (never modified), trains on the TRAIN split
only, validates on the full VAL split, and writes everything under --out:
  config.json, train.log (stdout), history.csv, val_predictions.csv,
  checkpoint_last.pth (resumable, every --ckpt-every steps), candidate.pth (final weights,
  evaluate.py-compatible format), summary.json.

One controlled change per arm:
  --weighting none | source_class_balanced
  --sampler   uniform | video_x4
Reuses train.get_train_transforms / get_eval_transforms / warmup_cosine-free cosine,
model.build_model / group_params_for_discriminative_lr -- the reviewed project code.
FAKE = 1, REAL = 0.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

BASE = r"D:\PROJECT\VeriFace AI"
sys.path.insert(0, BASE)
from model import build_model, group_params_for_discriminative_lr  # noqa: E402
from train import get_eval_transforms, get_train_transforms, system_memory_status  # noqa: E402
from PIL import Image  # noqa: E402

SEED = 42
VIDEO_FAMILIES = ("ffpp", "celebdf", "dfdc")


def log(m):
    print(f"[finetune] {m}", flush=True)


def family(g):
    return "ffpp" if g.startswith("ffpp") else "celebdf" if g.startswith("celebdf") else "dfdc" if g.startswith("dfdc") else g


class DS(Dataset):
    def __init__(self, df, transform):
        self.p = df["local_image_path"].tolist()
        self.y = df["label_numeric"].to_numpy(np.float32)
        self.w = df["loss_w"].to_numpy(np.float32) if "loss_w" in df else np.ones(len(df), np.float32)
        self.t = transform

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(os.path.join(BASE, self.p[i])) as im:
            x = self.t(im.convert("RGB"))
        return x, torch.tensor(self.y[i]), torch.tensor(self.w[i]), i


class FixedOrderSampler(Sampler):
    """Precomputed, seeded index order; supports skipping already-done steps on resume."""

    def __init__(self, order, start=0):
        self.order, self.start = order, start

    def __iter__(self):
        return iter(self.order[self.start:].tolist())

    def __len__(self):
        return len(self.order) - self.start


def build_order(df, sampler_kind, steps, bs, seed):
    g = np.random.default_rng(seed)
    n = steps * bs
    if sampler_kind == "uniform":
        return g.permutation(len(df))[:n]
    w = np.where(df["family"].isin(VIDEO_FAMILIES), 4.0, 1.0)
    return g.choice(len(df), size=n, replace=True, p=w / w.sum())


def loss_weights(df, kind):
    if kind == "none":
        return np.ones(len(df))
    w = np.ones(len(df))
    for f, sub in df.groupby("family"):
        counts = sub["label_numeric"].value_counts()
        if len(counts) < 2:
            continue
        for lab, n in counts.items():
            w[sub.index[sub.label_numeric == lab]] = len(sub) / (2 * n)
    return w / w.mean()


@torch.no_grad()
def predict(model, df, device, bs=64):
    dl = DataLoader(DS(df, get_eval_transforms()), batch_size=bs, shuffle=False, num_workers=0)
    model.eval()
    out = np.zeros(len(df))
    for x, _, _, i in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(x.to(device))
        out[i.numpy()] = lg.float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weighting", default="none", choices=["none", "source_class_balanced"])
    ap.add_argument("--sampler", default="uniform", choices=["uniform", "video_x4"])
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps (no candidate saved)")
    ap.add_argument("--val-limit", type=int, default=0, help="smoke test: validate on first N val rows")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exclude-generators", default="", help="comma list removed from TRAIN only")
    ap.add_argument("--warmup-frac", type=float, default=0.0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    last = os.path.join(a.out, "checkpoint_last.pth")
    if not a.resume and os.path.exists(os.path.join(a.out, "config.json")):
        raise SystemExit(f"{a.out} already holds a run; use --resume or a new --out")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"),
                     usecols=["image_id", "local_image_path", "label_numeric", "dataset_split", "generator", "group_id"])
    md["family"] = md["generator"].map(family)
    excl = [g for g in a.exclude_generators.split(",") if g]
    tr = md[(md.dataset_split == "train") & ~md.generator.isin(excl)].reset_index(drop=True)
    assert not set(excl) & set(tr.generator), "exclusion failed"
    va = md[md.dataset_split == "val"].reset_index(drop=True)
    if a.val_limit:
        va = va.sample(a.val_limit, random_state=0).reset_index(drop=True)
    assert set(tr.dataset_split) == {"train"} and set(va.dataset_split) == {"val"}
    tr["loss_w"] = loss_weights(tr, a.weighting)
    steps = int(math.floor(len(tr) / a.batch_size * a.epochs))
    order = build_order(tr, a.sampler, steps, a.batch_size, SEED)

    if a.init == "imagenet":
        ck = {"model_type": "vision", "backbone": "efficientnet_b0", "positive_class": "FAKE", "epoch": 0,
              "input_size": 224, "eval_transform": "center_crop_224_of_256_face_crop"}
        model = build_model(model_type="vision", backbone="efficientnet_b0", pretrained=True)
    else:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        assert ck.get("positive_class") == "FAKE"
        model = build_model(model_type=ck["model_type"], backbone=ck["backbone"], pretrained=False)
        model.load_state_dict(ck["state_dict"])
    model.to(dev)
    params = group_params_for_discriminative_lr(model, head_lr=a.lr, backbone_lr_mult=0.3, freeze_stem=True)
    opt = torch.optim.AdamW(params, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, int(a.warmup_frac * steps)) if s < int(a.warmup_frac * steps) else 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    step, hist = 0, []
    if a.resume:
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); step = st["step"]; hist = st["hist"]
        torch.set_rng_state(st["rng_cpu"])
        log(f"resumed at step {step}/{steps}")
    cfg = {**vars(a), "excluded_from_train": excl, "steps": steps, "n_train": len(tr), "n_val": len(va), "seed": SEED,
           "init_epoch": ck.get("epoch"), "loss_weight_by_cell": tr.groupby(["family", "label_numeric"])["loss_w"].first().round(4).to_dict()}
    cfg["loss_weight_by_cell"] = {f"{k[0]}|{k[1]}": v for k, v in cfg["loss_weight_by_cell"].items()}
    if a.sampler != "uniform":
        cfg["sampled_family_share"] = tr.iloc[order]["family"].value_counts(normalize=True).round(4).to_dict()
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    log(json.dumps(cfg, default=str))

    dl = DataLoader(DS(tr, get_train_transforms()), batch_size=a.batch_size,
                    sampler=FixedOrderSampler(order, step * a.batch_size), drop_last=True, num_workers=0)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    t0, run_l, seen, mem_min = time.time(), 0.0, 0, 1e9
    for x, y, w, _ in dl:
        if step >= steps:
            break
        x, y, w = x.to(dev), y.to(dev), w.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(x)
        loss = (crit(lg.float(), y) * w).sum() / w.sum()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        step += 1; run_l += loss.item(); seen += 1
        if step % 500 == 0:
            sm = system_memory_status()
            if sm:
                mem_min = min(mem_min, sm["commit_avail_gb"])
                if sm["commit_avail_gb"] < 2.0:
                    raise SystemExit(f"commit headroom {sm['commit_avail_gb']:.2f} GB < 2 GB -- stopping")
            log(f"step {step}/{steps} loss {run_l / seen:.4f} lr {sched.get_last_lr()[-1]:.2e} "
                f"{(time.time() - t0) / 60:.1f} min commit_avail {sm['commit_avail_gb'] if sm else float('nan'):.1f} GB "
                f"vram_peak {torch.cuda.max_memory_reserved() / 2**30 if dev.type == 'cuda' else 0:.2f} GB")
            hist.append({"step": step, "train_loss": run_l / seen}); run_l, seen = 0.0, 0
        if a.ckpt_every and step % a.ckpt_every == 0:
            tmp = last + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "hist": hist, "rng_cpu": torch.get_rng_state()}, tmp)
            os.replace(tmp, last)
        if a.max_steps and step >= a.max_steps:
            log(f"--max-steps {a.max_steps} reached (smoke test)")
            break
    train_min = (time.time() - t0) / 60

    lg = predict(model, va, dev)
    vp = va[["image_id", "generator", "label_numeric", "group_id"]].copy()
    vp["logit"] = lg; vp["prob_fake"] = 1 / (1 + np.exp(-lg))
    vp.to_csv(os.path.join(a.out, "val_predictions.csv"), index=False)
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
    from sklearn.metrics import roc_auc_score
    summ = {"steps_done": step, "train_minutes": round(train_min, 1), "val_auc": roc_auc_score(vp.label_numeric, vp.prob_fake),
            "min_commit_avail_gb": None if mem_min > 1e8 else mem_min,
            "vram_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else None}
    if not a.max_steps:
        torch.save({"backbone": ck["backbone"], "model_type": ck["model_type"], "positive_class": "FAKE",
                    "input_size": ck.get("input_size", 224), "eval_transform": ck.get("eval_transform"),
                    "research_arm": {"weighting": a.weighting, "sampler": a.sampler}, "init_checkpoint": a.init,
                    "config": cfg, "state_dict": model.state_dict()}, os.path.join(a.out, "candidate.pth"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    log(f"done: {json.dumps(summ)}")


if __name__ == "__main__":
    main()
