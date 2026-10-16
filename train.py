"""
VeriFace AI -- training.

Trains on the standardised face crops (dataset_processed/, via
cleaned_metadata.csv). Label convention: FAKE = 1 (positive class), so
sigmoid(logit) = P(FAKE). Only the train and val splits are read here; the
test split stays untouched until `python evaluate.py --split test`.

  - RGB EfficientNet-B0 baseline (MODEL_TYPE="vision"); dual_domain is opt-in
  - AdamW, discriminative LR (frozen stem, reduced backbone LR)
  - linear warmup (WARMUP_EPOCHS) -> cosine decay, stepped per batch
  - mixed precision (AMP) on CUDA; no channels_last
  - model selection + early stopping on validation EER (ties: val loss)
  - resumable: atomic last_checkpoint.pth (every --checkpoint-every steps,
    before validation and after each epoch) restored with --resume; every
    resume re-checks a fingerprint of the restored state against the one
    stored at save time

Usage:
    python train.py
    python train.py --resume
    python train.py --epochs 20 --batch-size 32 --num-workers 1
    # short isolated test run (outputs only under smoke_runs/demo):
    python train.py --output-dir smoke_runs/demo --epochs 2 --limit-train-batches 100 \
        --max-val-batches 20 --max-steps 150

Exit codes: 0 done, 3 stopped for low commit (checkpoint saved), 4 stopped at
--max-steps (checkpoint saved); anything else is an error.

Memory on Windows -- three different budgets:
  - System commit (commit charge vs commit limit = physical RAM + page file).
    Every allocation and every loaded DLL reserves commit, even if it is never
    touched. When it runs out, Windows fails allocations with WinError 1455 or
    DefaultCPUAllocator errors although physical RAM may still look free.
    This is what killed the earlier runs: each DataLoader worker is a separate
    process that re-imports this module and loads torch's CUDA DLLs
    (~1.5-2 GB commit each). Hence --num-workers 0 by default.
  - Physical RAM ("available" memory): what is resident now. Low RAM makes
    Windows page, which is slow but does not by itself fail allocations.
  - GPU VRAM: on the device, separate from both. Running out raises a CUDA
    out-of-memory error, not WinError 1455.
The [mem] log lines report all three.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import transforms
from tqdm import tqdm

from model import FaceImageDataset, build_model, group_params_for_discriminative_lr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Module level holds constants only. Spawned DataLoader workers re-import this
# module, so nothing here may touch CUDA (device selection lives in main()).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_metadata.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_face_model.pth")
LAST_CHECKPOINT_PATH = os.path.join(BASE_DIR, "last_checkpoint.pth")
REPORT_DIR = os.path.join(BASE_DIR, "training_reports")
CSV_COLUMNS = ["local_image_path", "label_numeric", "dataset_split"]
# --output-dir may not point at (or inside) these: they hold real data/results.
PROTECTED_DIRS = ["dataset_images", "dataset_processed", "audit_reports", "training_reports",
                  "evaluation_reports", "__pycache__"]

BACKBONE = "efficientnet_b0"  # or "xception"
MODEL_TYPE = "vision"  # "vision" (RGB only) or "dual_domain" (RGB + FFT branch)
INPUT_SIZE = 224  # crops of the 256x256 preprocessed faces -- no resampling
BATCH_SIZE = 16
NUM_EPOCHS = 15
WARMUP_EPOCHS = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 3  # epochs without val-EER improvement before stopping

# Discriminative fine-tuning: freeze the backbone's earliest conv layers
# (generic edge/texture detectors from ImageNet pretraining, which already
# transfer reasonably well to artifact detection) and train the deeper
# backbone layers at a reduced LR, while the head (and frequency branch, if
# MODEL_TYPE="dual_domain") trains at the full LEARNING_RATE.
USE_DISCRIMINATIVE_LR = True
BACKBONE_LR_MULT = 0.3
FREEZE_STEM = True

# Each worker re-imports torch + CUDA DLLs (~1.5-2 GB commit on Windows); 6
# and then 2 workers exhausted the commit limit on a 16 GB machine. Default to
# loading in-process; raise only after checking commit headroom.
NUM_WORKERS = 0
PREFETCH_FACTOR = 2  # batches per worker; only used when num_workers > 0
CHECKPOINT_EVERY = 500  # optimizer steps between mid-epoch last_checkpoint.pth saves
LOG_MEMORY_EVERY = 200  # optimizer steps between [mem] lines / commit checks
MIN_COMMIT_HEADROOM_GB = 1.0  # checkpoint + stop below this (0 disables)
EXIT_LOW_COMMIT = 3
EXIT_MAX_STEPS = 4
CHECKPOINT_FORMAT = 2
SEED = 42

# Changing any of these mid-run would make the restored optimizer/scheduler
# state, the data order or the validation subset meaningless, so --resume
# refuses when they differ.
RESUME_LOCKED_KEYS = [
    "model_type", "backbone", "input_size", "batch_size", "epochs", "lr", "weight_decay",
    "warmup_epochs", "use_discriminative_lr", "backbone_lr_mult", "freeze_stem", "amp", "seed",
    "n_train", "n_val", "train_fingerprint", "val_fingerprint",
    "limit_train_batches", "steps_per_epoch", "max_val_batches", "val_subset_crc", "cudnn_benchmark",
]

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def log(msg: str = ""):
    print(f"[VeriFace AI] {msg}", flush=True)


def get_train_transforms():
    # Tailored for artifact detection: every augmentation is gentle so it
    # perturbs pose/lighting/framing without destroying the high-frequency
    # cues (GAN checkerboarding, diffusion smoothing, blending seams) the
    # model needs. 224 crops of the 256 inputs instead of a resize, so pixels
    # are not resampled. No JPEG augmentation here: preprocess_data.py already
    # re-encodes every image (REAL and FAKE) at a random quality in [60, 100].
    return transforms.Compose([
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)], p=0.5,
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2),
        transforms.RandomAffine(
            degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        NORMALIZE,
        transforms.RandomErasing(p=0.15, scale=(0.01, 0.03), ratio=(0.5, 2.0), value="random"),
    ])


def get_eval_transforms():
    """Shared with evaluate.py so train/val/test inputs are framed identically."""
    return transforms.Compose([transforms.CenterCrop(INPUT_SIZE), transforms.ToTensor(), NORMALIZE])


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Memory diagnostics (stdlib ctypes only -- no extra packages)
# ---------------------------------------------------------------------------
def system_memory_status():
    """System-wide commit and physical RAM in GB via GlobalMemoryStatusEx; None off Windows.

    commit_avail is what the process can still commit before WinError 1455 /
    DefaultCPUAllocator failures; it is NOT the same as free physical RAM.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    gb = 1024 ** 3
    return {
        "commit_avail_gb": stat.ullAvailPageFile / gb,
        "commit_limit_gb": stat.ullTotalPageFile / gb,
        "ram_avail_gb": stat.ullAvailPhys / gb,
        "ram_total_gb": stat.ullTotalPhys / gb,
    }


def gpu_memory_status(device):
    if device.type != "cuda":
        return None
    gb = 1024 ** 3
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_used_gb": (total - free) / gb,  # every process on the GPU, incl. the desktop
        "device_total_gb": total / gb,
        "torch_alloc_gb": torch.cuda.memory_allocated(device) / gb,
        "torch_reserved_gb": torch.cuda.memory_reserved(device) / gb,
        "torch_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / gb,
    }


def log_memory(tag: str, device):
    sysm, gpu = system_memory_status(), gpu_memory_status(device)
    parts = [f"[mem] {tag}"]
    if sysm:
        parts.append(f"commit avail {sysm['commit_avail_gb']:.2f} / limit {sysm['commit_limit_gb']:.2f} GB")
        parts.append(f"RAM avail {sysm['ram_avail_gb']:.2f} / {sysm['ram_total_gb']:.2f} GB")
    if gpu:
        parts.append(f"VRAM device {gpu['device_used_gb']:.2f} / {gpu['device_total_gb']:.2f} GB")
        parts.append(f"torch alloc {gpu['torch_alloc_gb']:.2f} reserved {gpu['torch_reserved_gb']:.2f} "
                     f"peak {gpu['torch_peak_reserved_gb']:.2f} GB")
    log(" | ".join(parts))
    return sysm


def commit_too_low(sysm, threshold_gb: float) -> bool:
    return threshold_gb > 0 and sysm is not None and sysm["commit_avail_gb"] < threshold_gb


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def atomic_torch_save(obj, path: str):
    """Write to <path>.tmp, fsync, then os.replace -- the previous file stays
    intact until the new one is complete. Errors propagate (training stops)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)


def capture_rng_state(device) -> dict:
    # Stored as tensors/ints/lists only, so torch.load(weights_only=True) can read it.
    name, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
    state = {
        "python": random.getstate(),
        "numpy": {"name": name, "keys": keys.tolist(), "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device):
    version, internal, gauss_next = state["python"]
    random.setstate((version, tuple(internal), gauss_next))
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["pos"],
                         n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        if len(state["torch_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        else:
            log("WARNING: GPU count changed since the checkpoint; CUDA RNG state not restored.")


def _crc_tensor(crc: int, t) -> int:
    return zlib.crc32(t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes(), crc)


def state_fingerprint(model, optimizer, scheduler, scaler, device) -> dict:
    """CRC32s of the exact bytes of every piece of resumable state. Stored in the
    checkpoint at save time and recomputed after --resume restores it; any
    difference means the restore was incomplete, and resuming is refused.
    Reads state only -- consumes no random numbers."""
    model_crc = 0
    for name, t in model.state_dict().items():
        model_crc = _crc_tensor(zlib.crc32(name.encode(), model_crc), t)
    optim_crc = 0
    for pid, st in sorted(optimizer.state_dict()["state"].items()):
        for key in sorted(st):
            v = st[key]
            optim_crc = zlib.crc32(f"{pid}:{key}".encode(), optim_crc)
            optim_crc = _crc_tensor(optim_crc, v) if torch.is_tensor(v) else zlib.crc32(repr(v).encode(), optim_crc)
    rng = capture_rng_state(device)
    rng_crc = zlib.crc32(repr(rng["python"]).encode())
    rng_crc = zlib.crc32(repr((rng["numpy"]["keys"], rng["numpy"]["pos"])).encode(), rng_crc)
    rng_crc = _crc_tensor(rng_crc, rng["torch_cpu"])
    for t in rng.get("torch_cuda", []):
        rng_crc = _crc_tensor(rng_crc, t)
    return {
        "model_crc": model_crc, "optimizer_crc": optim_crc, "rng_crc": rng_crc,
        "scheduler_step": int(scheduler.last_epoch),
        "lrs": [float(lr) for lr in scheduler.get_last_lr()],
        "scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
    }


def format_fingerprint(fp: dict) -> str:
    return (f"model {fp['model_crc']:08x} | optimizer {fp['optimizer_crc']:08x} | rng {fp['rng_crc']:08x} | "
            f"scheduler step {fp['scheduler_step']} lr {', '.join(f'{lr:.3e}' for lr in fp['lrs'])} | "
            f"scaler {fp['scaler_scale']}")


def dataset_fingerprint(df: pd.DataFrame) -> int:
    """CRC32 of the ordered (path, label) pairs -- detects a changed split on --resume."""
    payload = "\n".join(f"{p}\t{int(l)}" for p, l in zip(df["local_image_path"], df["label_numeric"]))
    return zlib.crc32(payload.encode("utf-8"))


class EpochShuffleSampler(Sampler):
    """Deterministic per-epoch shuffle (seeded by SEED + epoch, independent of
    the global RNG). Each epoch is the first `steps_per_epoch` full batches of
    that permutation; `start_batch` lets --resume continue with exactly the
    remaining batches instead of re-shuffling or re-reading skipped images."""

    def __init__(self, num_samples: int, batch_size: int, steps_per_epoch: int, seed: int):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 1
        self.start_batch = 0

    def set_epoch(self, epoch: int, start_batch: int = 0):
        if not 0 <= start_batch <= self.steps_per_epoch:
            raise ValueError(f"start_batch {start_batch} outside [0, {self.steps_per_epoch}]")
        self.epoch = epoch
        self.start_batch = start_batch

    def epoch_order(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.num_samples, generator=g)[: self.steps_per_epoch * self.batch_size]

    def order_crc(self, first_batch: int, end_batch: int) -> int:
        part = self.epoch_order()[first_batch * self.batch_size: end_batch * self.batch_size]
        return zlib.crc32(part.to(torch.int64).numpy().tobytes())

    def __iter__(self):
        return iter(self.epoch_order()[self.start_batch * self.batch_size:].tolist())

    def __len__(self):
        return (self.steps_per_epoch - self.start_batch) * self.batch_size


@torch.no_grad()
def validate(model, loader, criterion, use_amp, device):
    model.eval()
    total_loss, probs, labels = 0.0, [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(imgs)
        total_loss += criterion(logits.float(), y).item() * len(y)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.cpu().numpy())
    p, y = np.concatenate(probs), np.concatenate(labels)
    if len(np.unique(y)) < 2:
        # Unreachable with the full split or the stratified-by-chance random subset,
        # but never let a one-class subset crash the run after training work.
        log("WARNING: validation set has a single class; AUC/EER undefined (NaN).")
        auc = eer = float("nan")
    else:
        auc, eer = float(roc_auc_score(y, p)), eer_from_scores(y, p)
    return {
        "loss": total_loss / len(y),
        "acc": float(((p >= 0.5) == y).mean()),
        "auc": auc,
        "eer": eer,
    }


def loader_kwargs(num_workers: int, prefetch_factor: int, persistent: bool) -> dict:
    # pin_memory off: page-locked buffers add host commit, which ran out on this machine.
    kw = dict(num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        kw.update(persistent_workers=persistent, prefetch_factor=prefetch_factor)
    return kw


def resolve_output_paths(output_dir):
    """Default: the historical project-root locations. With --output-dir every
    run artifact (last/best checkpoint, history, summary) goes under that dir."""
    if output_dir is None:
        return {"dir": None, "last": LAST_CHECKPOINT_PATH, "best": MODEL_SAVE_PATH,
                "history": os.path.join(REPORT_DIR, "history.csv"),
                "summary": os.path.join(REPORT_DIR, "training_summary.json")}
    out = os.path.realpath(output_dir if os.path.isabs(output_dir) else os.path.join(BASE_DIR, output_dir))
    base = os.path.realpath(BASE_DIR)
    norm = os.path.normcase
    if norm(out) == norm(base):
        raise SystemExit("--output-dir must not be the project root (that is where the real checkpoints live).")
    for d in PROTECTED_DIRS:
        p = os.path.realpath(os.path.join(BASE_DIR, d))
        if norm(out) == norm(p) or norm(out).startswith(norm(p) + os.sep):
            raise SystemExit(f"--output-dir must not be inside {d}/.")
    if os.path.exists(out) and not os.path.isdir(out):
        raise SystemExit(f"--output-dir {out} exists and is not a directory.")
    return {"dir": out, "last": os.path.join(out, "last_checkpoint.pth"),
            "best": os.path.join(out, "best_face_model.pth"),
            "history": os.path.join(out, "history.csv"),
            "summary": os.path.join(out, "training_summary.json")}


def validate_args(ap, args):
    positive = {"--epochs": args.epochs, "--batch-size": args.batch_size, "--patience": args.patience,
                "--prefetch-factor": args.prefetch_factor}
    non_negative = {"--num-workers": args.num_workers, "--val-num-workers": args.val_num_workers,
                    "--checkpoint-every": args.checkpoint_every, "--log-memory-every": args.log_memory_every,
                    "--limit-train-batches": args.limit_train_batches, "--max-val-batches": args.max_val_batches,
                    "--max-steps": args.max_steps, "--min-commit-gb": args.min_commit_gb}
    for flag, v in positive.items():
        if v <= 0:
            ap.error(f"{flag} must be > 0 (got {v})")
    for flag, v in non_negative.items():
        if v < 0:
            ap.error(f"{flag} must be >= 0 (got {v})")
    if not args.lr > 0:
        ap.error(f"--lr must be > 0 (got {args.lr})")
    if args.resume and args.allow_overwrite:
        ap.error("--resume and --allow-overwrite are mutually exclusive")


def main():
    ap = argparse.ArgumentParser(description="VeriFace AI -- training (FAKE = positive class)")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--model-type", default=MODEL_TYPE, choices=["vision", "dual_domain"])
    ap.add_argument("--backbone", default=BACKBONE, choices=["efficientnet_b0", "xception"])
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                    help="training loader worker processes (each costs ~1.5-2 GB commit on Windows)")
    ap.add_argument("--val-num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR,
                    help="batches prefetched per worker; ignored when workers = 0")
    ap.add_argument("--no-persistent-workers", action="store_true",
                    help="re-spawn loader workers every epoch instead of keeping them alive")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--no-cudnn-benchmark", action="store_true",
                    help="disable cuDNN autotuning (fixed algorithm choice for reproducibility checks)")
    ap.add_argument("--output-dir", default=None,
                    help="put last/best checkpoints, history.csv and training_summary.json here "
                         "(relative paths resolve against the project folder); default: project root + training_reports/")
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="use only the first N batches of each epoch's shuffled order (0 = full epoch)")
    ap.add_argument("--max-val-batches", type=int, default=0,
                    help="validate on a fixed seeded random subset of N val batches (0 = full split)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help=f"stop this session after N optimizer steps: save a checkpoint, exit code {EXIT_MAX_STEPS}")
    ap.add_argument("--resume", action="store_true", help="continue from last_checkpoint.pth in the output location")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="start a fresh run even though outputs from an earlier run exist in the output location")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                    help="optimizer steps between mid-epoch checkpoints (0 = epoch boundaries only)")
    ap.add_argument("--log-memory-every", type=int, default=LOG_MEMORY_EVERY)
    ap.add_argument("--min-commit-gb", type=float, default=MIN_COMMIT_HEADROOM_GB,
                    help="save a checkpoint and stop if available commit drops below this (0 disables)")
    args = ap.parse_args()
    validate_args(ap, args)
    paths = resolve_output_paths(args.output_dir)

    # Fresh runs must not silently replace an earlier run's outputs.
    existing = [p for k, p in paths.items() if k != "dir" and p and os.path.exists(p)]
    if args.resume and not os.path.exists(paths["last"]):
        raise SystemExit(f"--resume given but {paths['last']} does not exist.")
    if not args.resume and existing and not args.allow_overwrite:
        raise SystemExit("Outputs from an earlier run exist: " + ", ".join(existing)
                         + ". Use --resume to continue, or --allow-overwrite to start over (they will be replaced).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    cudnn_benchmark = device.type == "cuda" and not args.no_cudnn_benchmark
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cudnn_benchmark
    log("=" * 72)
    log(f"VeriFace AI -- training | device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f" | AMP: {use_amp} | cuDNN benchmark: {cudnn_benchmark}")
    log(f"Outputs: {paths['dir'] or 'project root + training_reports/'}")
    log("=" * 72)
    sysm = log_memory("startup", device)
    if commit_too_low(sysm, args.min_commit_gb):
        raise SystemExit(f"Only {sysm['commit_avail_gb']:.2f} GB commit available (< --min-commit-gb "
                         f"{args.min_commit_gb}). Close applications or enlarge the page file, then retry.")
    if args.num_workers > 0:
        log(f"Note: {args.num_workers} loader worker(s) -- each re-imports torch/CUDA (~1.5-2 GB commit).")

    # 1. Data -- image path + label only; metadata columns never reach the model.
    df = pd.read_csv(CSV_PATH, usecols=CSV_COLUMNS)
    if not df["local_image_path"].str.startswith("dataset_processed/").all():
        raise SystemExit("cleaned_metadata.csv does not point at dataset_processed/ -- run preprocess_data.py first.")
    if not df["label_numeric"].isin([0, 1]).all():
        raise SystemExit("label_numeric must be 0 (REAL) or 1 (FAKE).")
    train_df = df.loc[df["dataset_split"] == "train", ["local_image_path", "label_numeric"]].reset_index(drop=True)
    val_df = df.loc[df["dataset_split"] == "val", ["local_image_path", "label_numeric"]].reset_index(drop=True)
    del df  # test rows are never kept in memory here
    log(f"Train: {len(train_df):,} (FAKE {int(train_df.label_numeric.sum()):,}) | "
        f"Val: {len(val_df):,} (FAKE {int(val_df.label_numeric.sum()):,})")

    full_steps = len(train_df) // args.batch_size  # drop_last semantics
    if full_steps == 0:
        raise SystemExit(f"--batch-size {args.batch_size} exceeds the {len(train_df)} training images.")
    steps_per_epoch = min(full_steps, args.limit_train_batches) if args.limit_train_batches else full_steps
    if args.limit_train_batches > full_steps:
        log(f"--limit-train-batches {args.limit_train_batches} > {full_steps} batches available; using the full epoch.")

    train_ds = FaceImageDataset(train_df, BASE_DIR, transform=get_train_transforms())
    val_ds = FaceImageDataset(val_df, BASE_DIR, transform=get_eval_transforms())
    val_batch_size = args.batch_size * 2
    val_subset_crc = None
    if args.max_val_batches:
        # The CSV lists each split class by class, so "the first N batches"
        # would be all REAL. A fixed seeded subset is identical across runs
        # and contains both classes.
        n_val_used = min(len(val_ds), args.max_val_batches * val_batch_size)
        g = torch.Generator()
        g.manual_seed(SEED)
        val_idx = sorted(torch.randperm(len(val_ds), generator=g)[:n_val_used].tolist())
        val_subset_crc = zlib.crc32(np.asarray(val_idx, dtype=np.int64).tobytes())
        n_fake = int(val_ds.labels[val_idx].sum())
        if n_fake in (0, n_val_used):
            raise SystemExit("--max-val-batches subset contains a single class; use a larger value.")
        val_eval_ds = Subset(val_ds, val_idx)
        log(f"Validation subset: {n_val_used:,} of {len(val_ds):,} images (FAKE {n_fake:,}), "
            f"seeded indices CRC {val_subset_crc:08x}")
    else:
        val_eval_ds = val_ds

    sampler = EpochShuffleSampler(len(train_ds), args.batch_size, steps_per_epoch, SEED)
    persistent = not args.no_persistent_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                              **loader_kwargs(args.num_workers, args.prefetch_factor, persistent))
    # Validation loads in-process by default: extra worker processes (each
    # re-importing torch + CUDA DLLs) exhausted the Windows commit limit.
    val_loader = DataLoader(val_eval_ds, batch_size=val_batch_size, shuffle=False,
                            **loader_kwargs(args.val_num_workers, args.prefetch_factor, False))

    config = {
        "model_type": args.model_type, "backbone": args.backbone, "input_size": INPUT_SIZE,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr, "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS, "use_discriminative_lr": USE_DISCRIMINATIVE_LR,
        "backbone_lr_mult": BACKBONE_LR_MULT, "freeze_stem": FREEZE_STEM, "amp": use_amp, "seed": SEED,
        "cudnn_benchmark": cudnn_benchmark, "patience": args.patience,
        "num_workers": args.num_workers, "val_num_workers": args.val_num_workers,
        "positive_class": "FAKE", "n_train": len(train_df), "n_val": len(val_df),
        "train_fingerprint": dataset_fingerprint(train_df), "val_fingerprint": dataset_fingerprint(val_df),
        "limit_train_batches": args.limit_train_batches, "steps_per_epoch": steps_per_epoch,
        "max_val_batches": args.max_val_batches, "val_subset_crc": val_subset_crc,
        "output_dir": paths["dir"],
    }
    del train_df, val_df

    # 2. Model / optimiser / schedule
    model = build_model(model_type=args.model_type, backbone=args.backbone, pretrained=True)
    # No channels_last: benchmarked 9x slower for EfficientNet depthwise convs on this torch/cuDNN build.
    model = model.to(device)
    log_memory("model on device", device)
    criterion = nn.BCEWithLogitsLoss()
    if USE_DISCRIMINATIVE_LR:
        params = group_params_for_discriminative_lr(model, head_lr=args.lr, backbone_lr_mult=BACKBONE_LR_MULT,
                                                    freeze_stem=FREEZE_STEM)
    else:
        params = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_cosine(s, steps_per_epoch * WARMUP_EPOCHS, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if paths["dir"]:
        os.makedirs(paths["dir"], exist_ok=True)
    else:
        os.makedirs(REPORT_DIR, exist_ok=True)
    history = []
    best = {"eer": float("inf"), "loss": float("inf"), "epoch": 0}
    stale = 0
    global_step = 0
    start_epoch, start_batch = 1, 0
    run_loss, run_correct, seen, epoch_elapsed = 0.0, 0, 0, 0.0
    early_stopped = False

    # 3. Resume
    if args.resume:
        ckpt = torch.load(paths["last"], map_location="cpu", weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise SystemExit(f"{paths['last']} has format {ckpt.get('format')!r}; this train.py expects "
                             f"{CHECKPOINT_FORMAT}.")
        changed = [k for k in RESUME_LOCKED_KEYS if ckpt["config"].get(k) != config[k]]
        if changed:
            raise SystemExit("Cannot resume -- settings/data differ from the checkpoint: "
                             + ", ".join(f"{k} ({ckpt['config'].get(k)!r} -> {config[k]!r})" for k in changed))
        if ckpt["config"].get("num_workers") != config["num_workers"]:
            log(f"Note: num_workers changed {ckpt['config'].get('num_workers')} -> {config['num_workers']}; "
                "state restores exactly, but augmentation random streams will differ from an uninterrupted run.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history, best, stale = ckpt["history"], ckpt["best"], ckpt["stale"]
        global_step, early_stopped = ckpt["global_step"], ckpt["early_stopped"]
        if ckpt["epoch_complete"]:
            start_epoch, start_batch = ckpt["epoch"] + 1, 0
        else:
            start_epoch, start_batch = ckpt["epoch"], ckpt["batches_done"]
            run_loss, run_correct, seen = ckpt["run_loss"], ckpt["run_correct"], ckpt["seen"]
            epoch_elapsed = ckpt["epoch_elapsed"]
        # Last, so model construction above cannot consume the restored RNG streams.
        restore_rng_state(ckpt["rng"], device)
        restored = state_fingerprint(model, optimizer, scheduler, scaler, device)
        if restored != ckpt["fingerprint"]:
            raise SystemExit("Restored state does not match the checkpoint fingerprint:\n  saved    "
                             + format_fingerprint(ckpt["fingerprint"]) + "\n  restored " + format_fingerprint(restored))
        if start_epoch <= args.epochs:
            sampler.set_epoch(start_epoch, start_batch)
            order = sampler.epoch_order()
            nxt = int(order[start_batch * args.batch_size]) if start_batch < steps_per_epoch else None
            log(f"Resumed from {paths['last']}: epoch {start_epoch}, batch {start_batch}/{steps_per_epoch} "
                f"(samples {start_batch * args.batch_size}..{steps_per_epoch * args.batch_size - 1} of this epoch's "
                f"order remain), global step {global_step}, next sample index {nxt}, "
                f"order CRC full {sampler.order_crc(0, steps_per_epoch):08x} done "
                f"{sampler.order_crc(0, start_batch):08x} remaining {sampler.order_crc(start_batch, steps_per_epoch):08x}, "
                f"best val EER {best['eer'] * 100:.2f}% (epoch {best['epoch']})")
        log("[state] restored (matches saved fingerprint): " + format_fingerprint(restored))
        del ckpt

    def save_last(epoch, batches_done, epoch_complete, elapsed):
        fp = state_fingerprint(model, optimizer, scheduler, scaler, device)
        atomic_torch_save({
            "format": CHECKPOINT_FORMAT, "config": config,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batches_done": batches_done, "epoch_complete": epoch_complete,
            "global_step": global_step, "run_loss": run_loss, "run_correct": run_correct, "seen": seen,
            "epoch_elapsed": elapsed, "history": history, "best": best, "stale": stale,
            "early_stopped": early_stopped, "rng": capture_rng_state(device), "fingerprint": fp,
        }, paths["last"])
        return fp

    def stop_session(reason, exit_code, epoch, batches_done, elapsed):
        # The checkpoint must succeed before stopping; if it raises, the error propagates.
        fp = save_last(epoch, batches_done, False, elapsed)
        log(f"{reason} Saved {paths['last']} at epoch {epoch}, batch {batches_done}/{steps_per_epoch}, "
            f"global step {global_step}; epoch order CRC done {sampler.order_crc(0, batches_done):08x}.")
        log("[state] saved: " + format_fingerprint(fp))
        log(f"Continue with the same arguments plus --resume. Exit code {exit_code}.")
        sys.exit(exit_code)

    def write_history():
        atomic_write_text(paths["history"], pd.DataFrame(history).to_csv(index=False))

    t_start = time.time()
    session_steps = 0

    # 4. Training loop
    log(f"Training {args.model_type}/{args.backbone} for up to {args.epochs} epochs "
        f"({steps_per_epoch:,} steps/epoch{' (limited)' if steps_per_epoch < full_steps else ''}, "
        f"batch {args.batch_size}, workers {args.num_workers}, warmup {WARMUP_EPOCHS} epoch, cosine decay, "
        f"patience {args.patience}" + (f", session cap {args.max_steps} steps" if args.max_steps else "") + ")")
    if early_stopped or start_epoch > args.epochs:
        log("Checkpoint already reached the end of training; nothing left to run.")
    for epoch in range(start_epoch, args.epochs + 1):
        if early_stopped:
            break
        first = start_batch if epoch == start_epoch else 0
        if first == 0:
            run_loss, run_correct, seen, epoch_elapsed = 0.0, 0, 0, 0.0
        sampler.set_epoch(epoch, first)
        log(f"epoch {epoch}: batches {first}..{steps_per_epoch - 1}, order CRC full "
            f"{sampler.order_crc(0, steps_per_epoch):08x} this session {sampler.order_crc(first, steps_per_epoch):08x}")
        model.train()
        t0 = time.time() - epoch_elapsed
        bar = tqdm(train_loader, desc=f"[VeriFace AI] epoch {epoch}/{args.epochs}",
                   total=steps_per_epoch, initial=first, mininterval=30)
        for batch_idx, (imgs, y) in enumerate(bar, start=first + 1):
            imgs = imgs.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(imgs)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            session_steps += 1

            run_loss += loss.item() * len(y)
            run_correct += ((logits.detach() >= 0).float() == y).sum().item()
            seen += len(y)
            bar.set_postfix(loss=f"{run_loss / seen:.4f}", lr=f"{scheduler.get_last_lr()[-1]:.2e}")

            if args.log_memory_every and global_step % args.log_memory_every == 0:
                sysm = log_memory(f"epoch {epoch} step {batch_idx}/{steps_per_epoch}", device)
                if commit_too_low(sysm, args.min_commit_gb):
                    bar.close()
                    stop_session(f"Available commit {sysm['commit_avail_gb']:.2f} GB < --min-commit-gb "
                                 f"{args.min_commit_gb}.", EXIT_LOW_COMMIT, epoch, batch_idx, time.time() - t0)
            if args.max_steps and session_steps >= args.max_steps:
                bar.close()
                stop_session(f"Reached --max-steps {args.max_steps} for this session.", EXIT_MAX_STEPS,
                             epoch, batch_idx, time.time() - t0)
            if args.checkpoint_every and global_step % args.checkpoint_every == 0 and batch_idx < steps_per_epoch:
                save_last(epoch, batch_idx, False, time.time() - t0)
        bar.close()
        train_seconds = time.time() - t0

        # Training part of the epoch is safe on disk before validation starts.
        save_last(epoch, steps_per_epoch, False, train_seconds)
        sysm = log_memory(f"epoch {epoch} before validation", device)
        if commit_too_low(sysm, args.min_commit_gb):
            stop_session(f"Available commit {sysm['commit_avail_gb']:.2f} GB < --min-commit-gb "
                         f"{args.min_commit_gb}.", EXIT_LOW_COMMIT, epoch, steps_per_epoch, train_seconds)

        val = validate(model, val_loader, criterion, use_amp, device)
        log_memory(f"epoch {epoch} after validation", device)
        rec = {"epoch": epoch, "train_loss": run_loss / max(1, seen), "train_acc": run_correct / max(1, seen),
               **{f"val_{k}": v for k, v in val.items()}, "lr_end": scheduler.get_last_lr()[-1],
               "epoch_seconds": round(time.time() - t0, 1), "global_step": global_step}
        history.append(rec)
        log(f"epoch {epoch:>2} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f} AUC {val['auc']:.4f} EER {val['eer'] * 100:.2f}% | "
            f"{rec['epoch_seconds']:.0f}s")

        improved = val["eer"] < best["eer"] - 1e-4 or (abs(val["eer"] - best["eer"]) <= 1e-4 and val["loss"] < best["loss"])
        if improved:
            best = {"eer": val["eer"], "loss": val["loss"], "epoch": epoch, "auc": val["auc"]}
            stale = 0
            atomic_torch_save({
                "backbone": args.backbone,
                "model_type": args.model_type,
                "positive_class": "FAKE",  # sigmoid(logit) = P(FAKE); evaluate.py checks this
                "input_size": INPUT_SIZE,
                "eval_transform": "center_crop_224_of_256_face_crop",
                "epoch": epoch,
                "val_metrics": val,
                "config": config,
                "state_dict": model.state_dict(),
            }, paths["best"])
            log(f"  --> new best (val EER {val['eer'] * 100:.2f}%), saved {paths['best']}")
        else:
            stale += 1
            if stale >= args.patience:
                early_stopped = True
                log(f"Early stopping: val EER has not improved for {args.patience} epochs.")
        fp = save_last(epoch, steps_per_epoch, True, time.time() - t0)
        write_history()
        log(f"[state] epoch {epoch} end: " + format_fingerprint(fp))

    if history:
        write_history()
    summary = {"epochs_completed": len(history), "best_epoch": best["epoch"], "best_val_eer": best["eer"],
               "best_val_loss": best["loss"], "best_val_auc": best.get("auc"), "global_step": global_step,
               "minutes_this_session": round((time.time() - t_start) / 60, 1), "config": config}
    atomic_write_text(paths["summary"], json.dumps(summary, indent=2))
    log(f"Training complete: {len(history)} epochs, best epoch {best['epoch']} "
        f"(val EER {best['eer'] * 100:.2f}%, loss {best['loss']:.4f}).")
    log("Next: python evaluate.py --split test")


if __name__ == "__main__":
    # Guarding the entry point is what lets DataLoader workers > 0 run on
    # Windows (spawn re-imports this module in each worker process).
    main()
