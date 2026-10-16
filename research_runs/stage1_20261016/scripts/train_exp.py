"""
VeriFace AI stage 1 -- corrected experiment trainer (replaces the previous study's finetune.py /
train_logo.py for new runs; those files are left untouched as historical artifacts).

Fixes vs the previous study
  * build_order: one fresh seeded permutation PER EPOCH, concatenated -> the requested number of
    epochs/steps really runs (previously capped at one epoch). The run asserts steps_done == steps.
  * per-epoch validation predictions are written (val_predictions_epochNN.csv); the final epoch's
    predictions are also written as val_predictions.csv. Model selection = final epoch (pre-declared).

Options (one controlled change per experiment)
  --init imagenet | <checkpoint.pth>
  --exclude-generators g1,g2   removed from TRAIN only (leave-generator-out)
  --aug-jpeg LO,HI,P           with prob P re-encode the training image as JPEG q~U[LO,HI] before the
                               standard train transforms (compression-robustness arm)
Train split for fitting, val split for monitoring; the test split is never read. FAKE = 1, REAL = 0.
"""
import argparse
import io
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

BASE = r"D:\PROJECT\VeriFace AI"
sys.path.insert(0, BASE)
from model import build_model, group_params_for_discriminative_lr  # noqa: E402
from train import get_eval_transforms, get_train_transforms, system_memory_status  # noqa: E402

SEED = 42


def log(m):
    print(f"[train_exp] {m}", flush=True)


class JpegAug:
    def __init__(self, lo, hi, p, seed):
        self.lo, self.hi, self.p = lo, hi, p
        self.rng = random.Random(seed)

    def __call__(self, img):
        if self.rng.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.rng.randint(self.lo, self.hi), subsampling=2)
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("RGB")


class DS(Dataset):
    def __init__(self, df, transform, pre=None):
        self.p = df["local_image_path"].tolist()
        self.y = df["label_numeric"].to_numpy(np.float32)
        self.t, self.pre = transform, pre

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(os.path.join(BASE, self.p[i])) as im:
            img = im.convert("RGB")
        if self.pre is not None:
            img = self.pre(img)
        return self.t(img), torch.tensor(self.y[i]), i


class FixedOrderSampler(Sampler):
    def __init__(self, order, start=0):
        self.order, self.start = order, start

    def __iter__(self):
        return iter(self.order[self.start:].tolist())

    def __len__(self):
        return len(self.order) - self.start


def build_order(n_rows, epochs, bs, seed):
    """Per-epoch permutations, each truncated to whole batches (drop_last semantics)."""
    g = np.random.default_rng(seed)
    spe = n_rows // bs
    full = int(math.floor(epochs))
    parts = [g.permutation(n_rows)[: spe * bs] for _ in range(full)]
    frac = epochs - full
    if frac > 1e-9:
        parts.append(g.permutation(n_rows)[: int(round(spe * frac)) * bs])
    order = np.concatenate(parts)
    return order, len(order) // bs, spe


@torch.no_grad()
def predict(model, df, device, bs=64):
    dl = DataLoader(DS(df, get_eval_transforms()), batch_size=bs, shuffle=False, num_workers=0)
    was = model.training
    model.eval()
    out = np.zeros(len(df))
    for x, _, i in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(x.to(device))
        out[i.numpy()] = lg.float().cpu().numpy()
    model.train(was)
    return out


def save_val(model, va, dev, path):
    lg = predict(model, va, dev)
    vp = va[["image_id", "generator", "label_numeric", "group_id"]].copy()
    vp["logit"] = lg
    vp["prob_fake"] = 1 / (1 + np.exp(-lg))
    vp.to_csv(path, index=False)
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(vp.label_numeric, vp.prob_fake))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--exclude-generators", default="")
    ap.add_argument("--aug-jpeg", default="")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test only")
    ap.add_argument("--val-limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    last = os.path.join(a.out, "checkpoint_last.pth")
    if not a.resume and os.path.exists(os.path.join(a.out, "config.json")):
        raise SystemExit(f"{a.out} already holds a run; use --resume or a new --out")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"),
                     usecols=["image_id", "local_image_path", "label_numeric", "dataset_split", "generator", "group_id"])
    excl = [g for g in a.exclude_generators.split(",") if g]
    unknown = set(excl) - set(md.generator)
    if unknown:
        raise SystemExit(f"unknown generators: {unknown}")
    tr = md[(md.dataset_split == "train") & ~md.generator.isin(excl)].reset_index(drop=True)
    va = md[md.dataset_split == "val"].reset_index(drop=True)
    if a.val_limit:
        va = va.sample(a.val_limit, random_state=0).reset_index(drop=True)
    assert set(tr.dataset_split) == {"train"} and set(va.dataset_split) == {"val"} and not set(excl) & set(tr.generator)

    order, steps, spe = build_order(len(tr), a.epochs, a.batch_size, SEED)
    pre = None
    if a.aug_jpeg:
        lo, hi, p = a.aug_jpeg.split(",")
        pre = JpegAug(int(lo), int(hi), float(p), SEED)

    if a.init == "imagenet":
        ck = {"model_type": "vision", "backbone": "efficientnet_b0", "positive_class": "FAKE", "epoch": 0,
              "input_size": 224, "eval_transform": "center_crop_224_of_256_face_crop"}
        model = build_model(model_type="vision", backbone="efficientnet_b0", pretrained=True)
    else:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        if ck.get("positive_class") != "FAKE":
            raise SystemExit("init checkpoint is not FAKE-positive")
        model = build_model(model_type=ck["model_type"], backbone=ck["backbone"], pretrained=False)
        model.load_state_dict(ck["state_dict"])
    model.to(dev)
    params = group_params_for_discriminative_lr(model, head_lr=a.lr, backbone_lr_mult=0.3, freeze_stem=True)
    opt = torch.optim.AdamW(params, weight_decay=1e-2)
    wu = int(a.warmup_frac * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * min(s - wu, steps - wu) / max(1, steps - wu))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    step, hist, epoch_val = 0, [], []
    if a.resume:
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); step = st["step"]; hist = st["hist"]; epoch_val = st.get("epoch_val", [])
        torch.set_rng_state(st["rng_cpu"])
        if dev.type == "cuda" and "rng_cuda" in st:
            torch.cuda.set_rng_state(st["rng_cuda"])
        log(f"resumed at step {step}/{steps}")
    cfg = {**vars(a), "excluded_from_train": excl, "steps": steps, "steps_per_epoch": spe, "n_train": len(tr),
           "n_val": len(va), "seed": SEED, "init_epoch": ck.get("epoch"),
           "train_generators": sorted(tr.generator.unique().tolist())}
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    log(json.dumps({k: v for k, v in cfg.items() if k != "train_generators"}, default=str))

    dl = DataLoader(DS(tr, get_train_transforms(), pre), batch_size=a.batch_size,
                    sampler=FixedOrderSampler(order, step * a.batch_size), drop_last=True, num_workers=0)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    t0, run_l, seen, mem_min = time.time(), 0.0, 0, 1e9
    for x, y, _ in dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(x)
        loss = crit(lg.float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        step += 1; run_l += loss.item(); seen += 1
        if step % 500 == 0:
            sm = system_memory_status()
            ca = sm["commit_avail_gb"] if sm else float("nan")
            mem_min = min(mem_min, ca) if sm else mem_min
            if sm and ca < 2.0:
                raise SystemExit(f"commit headroom {ca:.2f} GB < 2 GB -- stopping (resume with --resume)")
            log(f"step {step}/{steps} loss {run_l / seen:.4f} lr {sched.get_last_lr()[-1]:.2e} "
                f"{(time.time() - t0) / 60:.1f} min commit_avail {ca:.1f} GB "
                f"vram_peak {torch.cuda.max_memory_reserved() / 2**30 if dev.type == 'cuda' else 0:.2f} GB")
            hist.append({"step": step, "train_loss": run_l / seen}); run_l, seen = 0.0, 0
        if step % spe == 0 and not a.max_steps:
            ep = step // spe
            auc = save_val(model, va, dev, os.path.join(a.out, f"val_predictions_epoch{ep:02d}.csv"))
            epoch_val.append({"epoch": ep, "step": step, "val_auc": auc})
            log(f"epoch {ep} done at step {step}: val AUC {auc:.4f}")
        if a.ckpt_every and step % a.ckpt_every == 0:
            st = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "scaler": scaler.state_dict(), "step": step, "hist": hist, "epoch_val": epoch_val,
                  "rng_cpu": torch.get_rng_state()}
            if dev.type == "cuda":
                st["rng_cuda"] = torch.cuda.get_rng_state()
            torch.save(st, last + ".tmp"); os.replace(last + ".tmp", last)
        if a.max_steps and step >= a.max_steps:
            log(f"--max-steps {a.max_steps} reached (smoke test)")
            break
    train_min = (time.time() - t0) / 60
    if not a.max_steps and step != steps:
        raise SystemExit(f"step count mismatch: ran {step}, planned {steps}")

    auc = save_val(model, va, dev, os.path.join(a.out, "val_predictions.csv"))
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
    summ = {"steps_done": step, "steps_planned": steps, "epochs": a.epochs, "train_minutes_this_session": round(train_min, 1),
            "val_auc_final": auc, "epoch_val": epoch_val, "min_commit_avail_gb": None if mem_min > 1e8 else mem_min,
            "vram_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else None,
            "smoke_test": bool(a.max_steps)}
    if not a.max_steps:
        torch.save({"backbone": ck["backbone"], "model_type": ck["model_type"], "positive_class": "FAKE",
                    "input_size": 224, "eval_transform": "center_crop_224_of_256_face_crop",
                    "research_config": cfg, "init_checkpoint": a.init, "state_dict": model.state_dict()},
                   os.path.join(a.out, "candidate.pth"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    log(f"done: {json.dumps(summ)}")


if __name__ == "__main__":
    main()
