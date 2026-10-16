"""
VeriFace AI -- Phase 1 dataset audit, deduplication and leak-free re-split.

Stages (all run by a single `python audit_dataset.py`):
  1. Index every image under dataset_images/ (Kaggle 140k real-vs-fake and
     GRAVEX-200K), attributing each file to a source dataset, a generator
     and a set of split-group keys (FFHQ id, source video, identity, ...).
  2. Hash every image (MD5 + 64-bit DCT pHash) and collect resolution, aspect
     ratio, file size, format and JPEG quantisation statistics.
  3. Detect exact (MD5) and near (pHash Hamming <= --phash-threshold)
     duplicates, including across the two source datasets.
  4. Deduplicate, union every row into split groups (shared identity / video
     / duplicate cluster) and assign a StratifiedGroupKFold 80/10/10
     train/val/test split, stratified by generator.
  5. Run the metadata shortcut probe (Logistic Regression + Decision Tree on
     width, height, file size, aspect ratio) on the new split.
  6. Export cleaned_metadata.csv and a full report under audit_reports/.

VeriFace AI label convention: FAKE is the positive class (label_numeric = 1),
REAL = 0. The GRAVEX label CSVs use the opposite convention (1 = REAL); they
are only used here to cross-check the folder labels, and their split columns
are ignored.

cleaned_metadata.csv is written with local_image_path pointing at the raw
image. Run preprocess_data.py afterwards to produce the standardised face
crops and repoint local_image_path at them.
"""

import os

# Pin native thread pools BEFORE numpy/sklearn load. On Windows every Pool
# worker re-imports this module; with default OpenBLAS threading each worker
# reserves per-thread buffers and 14 workers exhausted the commit limit.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import io
import json
import re
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_images")
KAGGLE_DIR = os.path.join(DATA_DIR, "real-vs-fake")
GRAVEX_DIR = os.path.join(DATA_DIR, "my_real_vs_ai_dataset")
GRAVEX_IMG_DIR = os.path.join(GRAVEX_DIR, "my_real_vs_ai_dataset")
REPORT_DIR = os.path.join(BASE_DIR, "audit_reports")
INDEX_CACHE = os.path.join(REPORT_DIR, "image_index.parquet")
OUTPUT_CSV = os.path.join(BASE_DIR, "cleaned_metadata.csv")

SEED = 42
LABEL_TO_NUMERIC = {"REAL": 0, "FAKE": 1}  # FAKE is the positive class
# When duplicates collide, keep the copy from the canonical upstream source.
SOURCE_PRIORITY = {"kaggle_140k": 0, "gravex_200k": 1}

# Probe verdict thresholds: above these, metadata alone separates the classes.
PROBE_MAX_BALANCED_ACC = 0.60
PROBE_MAX_AUC = 0.65

FFPP_FAMILY = {
    "Deepfakes": "face_swap",
    "FaceSwap": "face_swap",
    "FaceShifter": "face_swap",
    "DeepFakeDetection": "face_swap",
    "Face2Face": "face_reenactment",
    "NeuralTextures": "face_reenactment",
}

PROBE_FEATURES = {
    "metadata (w, h, file size, aspect)": ["width", "height", "file_size", "aspect_ratio"],
    "metadata + JPEG quant table": ["width", "height", "file_size", "aspect_ratio", "jpeg_luma_q"],
}


def log(msg: str = ""):
    print(f"[VeriFace AI] {msg}", flush=True)


def rel(path: str) -> str:
    return os.path.relpath(path, BASE_DIR).replace("\\", "/")


# ---------------------------------------------------------------------------
# 1. Indexing & generator attribution
# ---------------------------------------------------------------------------
def attribute_gravex(stem: str, label: str):
    """
    Map a GRAVEX-200K filename to (generator, generator_family, group_keys, flag).

    GRAVEX is a remix of several public corpora; the filename is the only
    provenance signal, so attribution is pattern based. group_keys tie
    together every image that shares an identity / source video, so that
    frames of one video can never straddle train and test.
    """
    if stem.startswith("aug_ai_") or stem.startswith("Aumented real"):
        # Pre-baked augmentations of other images in the set (flip, blur,
        # masking, ...): derived duplicates, not independent samples.
        return "augmented_copy", "augmented", [], "augmented_copy"

    if label == "REAL":
        if re.fullmatch(r"\d{5}", stem):
            return "ffhq", "real", [f"ffhq:{stem}"], ""
        if m := re.fullmatch(r"real_(\d{3})_\d+", stem):
            return "ffpp_original", "real", [f"ffpp:{m[1]}"], ""
        if m := re.match(r"celebdf_[a-z]+_id(\d+)_\d+_frame", stem):
            return "celebdf_real", "real", [f"celebdf_id:{m[1]}"], ""
        if m := re.match(r"celebdf_[a-z]+_(\d+)_frame", stem):
            return "celebdf_youtube_real", "real", [f"celebdf_yt:{m[1]}"], ""
        if m := re.fullmatch(r"dfdc_[a-z]+_([a-z]+)_\d+_\d+", stem):
            return "dfdc_real", "real", [f"dfdc:{m[1]}"], ""
        if stem.startswith("ai_detect_real_"):
            return "web_photo_real", "real", [], ""
    else:
        if re.fullmatch(r"[A-Z0-9]{10}", stem):
            return "stylegan", "gan", [f"stylegan:{stem}"], ""
        if m := re.fullmatch(r"fake_(Deepfakes|Face2Face|FaceSwap|FaceShifter|NeuralTextures)_(\d{3})_(\d{3})_\d+", stem):
            # target video + source-face video: both identities stay together
            return f"ffpp_{m[1]}", FFPP_FAMILY[m[1]], [f"ffpp:{m[2]}", f"ffpp:{m[3]}"], ""
        if m := re.match(r"fake_DeepFakeDetection_(\d+)_(\d+)__", stem):
            return "ffpp_DeepFakeDetection", "face_swap", [f"dfd_actor:{m[1]}", f"dfd_actor:{m[2]}"], ""
        if m := re.match(r"celebdf_[a-z]+_id(\d+)_id(\d+)_\d+_frame", stem):
            # Grouped by the target identity (whose real videos are also in
            # the set). Linking the source identity too collapses Celeb-DF's
            # swap graph into one giant group; see the audit report caveats.
            return "celebdf_synthesis", "face_swap", [f"celebdf_id:{m[1]}"], ""
        if m := re.match(r"dfdc_[a-z]+_fake_([a-z]+)_", stem):
            return "dfdc_fake", "face_swap", [f"dfdc:{m[1]}"], ""
        if stem.startswith("ai_detect_fake_"):
            return "text_to_image_unknown", "diffusion", [], ""
        if re.search(r"_SD_\d+$", stem):
            return "stable_diffusion", "diffusion", [], ""
        if re.search(r"_men_\d+$", stem):
            return "unattributed_ai", "unknown_ai", [], ""

    return "unknown", "unknown", [], "unattributed"


def build_index() -> tuple[pd.DataFrame, dict]:
    rows = []

    for split in ("train", "valid", "test"):
        for folder, label in (("real", "REAL"), ("fake", "FAKE")):
            d = os.path.join(KAGGLE_DIR, split, folder)
            for f in sorted(os.listdir(d)):
                stem = os.path.splitext(f)[0]
                if label == "REAL":
                    gen, fam, keys = "ffhq", "real", [f"ffhq:{stem}"]
                else:
                    gen, fam, keys = "stylegan", "gan", [f"stylegan:{stem}"]
                rows.append(dict(
                    source_dataset="kaggle_140k", filename=f, raw_path=rel(os.path.join(d, f)),
                    label=label, orig_split=split, source_csv_label="",
                    generator=gen, generator_family=fam, group_keys="|".join(keys), flag="",
                ))

    # GRAVEX: folder gives the label, CSVs are only a cross-check.
    csv_rows = {}
    for split in ("train", "val", "test"):
        c = pd.read_csv(os.path.join(GRAVEX_DIR, f"{split}_labels.csv"))
        for fn, lab in zip(c["filename"], c["label"]):
            csv_rows[fn] = (split, int(lab))

    for folder, label in (("real", "REAL"), ("ai_images", "FAKE")):
        d = os.path.join(GRAVEX_IMG_DIR, folder)
        for f in sorted(os.listdir(d)):
            stem = os.path.splitext(f)[0]
            gen, fam, keys, flag = attribute_gravex(stem, label)
            split, csv_lab = csv_rows.get(f, ("", -1))
            rows.append(dict(
                source_dataset="gravex_200k", filename=f, raw_path=rel(os.path.join(d, f)),
                label=label, orig_split=split, source_csv_label=str(csv_lab),
                generator=gen, generator_family=fam, group_keys="|".join(keys), flag=flag,
            ))

    df = pd.DataFrame(rows)

    # Infer the GRAVEX CSV convention from the folder labels instead of assuming it.
    gx = df[df["source_dataset"] == "gravex_200k"]
    xtab = pd.crosstab(gx["label"], gx["source_csv_label"])
    real_code = xtab.loc["REAL"].idxmax()
    csv_text = gx["source_csv_label"].map(lambda v: "REAL" if v == real_code else ("FAKE" if v != "-1" else ""))
    conflict = (csv_text != "") & (csv_text != gx["label"])
    df.loc[gx.index[conflict], "flag"] = "csv_folder_label_conflict"
    df.loc[gx.index[csv_text == ""], "flag"] = df.loc[gx.index[csv_text == ""], "flag"].replace("", "missing_from_csv")

    csv_info = {
        "gravex_csv_real_code": real_code,
        "gravex_folder_vs_csv_crosstab": {str(k): {str(c): int(v) for c, v in r.items()} for k, r in xtab.iterrows()},
        "gravex_csv_folder_label_conflicts": int(conflict.sum()),
    }

    # image_id: unique, filesystem-safe, readable provenance
    prefix = df["source_dataset"].map({"kaggle_140k": "kg", "gravex_200k": "gx"})
    df["image_id"] = prefix + "_" + df["label"].str.lower() + "_" + df["filename"].str.replace(r"\.[^.]+$", "", regex=True)
    df["image_id"] = df["image_id"].str.replace(r"[^A-Za-z0-9_\-]+", "_", regex=True)
    dup_id = df["image_id"].duplicated(keep=False)
    if dup_id.any():
        df.loc[dup_id, "image_id"] = df.loc[dup_id, "image_id"] + "_" + df.loc[dup_id, "orig_split"]
    assert df["image_id"].is_unique, "image_id collision"
    return df, csv_info


# ---------------------------------------------------------------------------
# 2. Hashing
# ---------------------------------------------------------------------------
def phash64(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash (same construction as imagehash.phash)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _hash_one(path: str):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            fmt, mode = im.format, im.mode
            q = getattr(im, "quantization", None)
            jq = float(np.mean(q[0])) if q else np.nan
            gray = np.asarray(im.convert("L"))
        return md5, w, h, fmt, mode, len(data), jq, phash64(gray), ""
    except Exception as e:  # noqa: BLE001 -- record and keep going
        return "", 0, 0, "", "", 0, np.nan, 0, f"{type(e).__name__}: {e}"


def hash_images(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    paths = [os.path.join(BASE_DIR, p) for p in df["raw_path"]]
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(_hash_one, paths, chunksize=256), total=len(paths),
                            desc="[VeriFace AI] hashing", mininterval=5))
    cols = ["md5", "width", "height", "img_format", "img_mode", "file_size", "jpeg_luma_q", "phash", "read_error"]
    h = pd.DataFrame(results, columns=cols, index=df.index)
    h["phash"] = h["phash"].astype(np.uint64)
    out = pd.concat([df, h], axis=1)
    out["aspect_ratio"] = out["width"] / out["height"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# 3. Near-duplicate search
# ---------------------------------------------------------------------------
def near_duplicate_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 1024):
    """
    All index pairs (i < j) with Hamming(hashes[i], hashes[j]) <= max_dist.

    Multi-index hashing: split the 64 bits into max_dist + 1 bands; by the
    pigeonhole principle any pair within max_dist agrees exactly on at least
    one band, so only rows sharing a band value need comparing.
    """
    uniq, inverse = np.unique(hashes, return_inverse=True)
    nb = max_dist + 1
    widths = [64 // nb + (1 if b < 64 % nb else 0) for b in range(nb)]
    pair_set = set()
    shift = 64
    for w in widths:
        shift -= w
        keys = (uniq >> np.uint64(shift)) & np.uint64((1 << w) - 1)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        ends = np.r_[starts[1:], len(sk)]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            members = order[s:e]
            vals = uniq[members]
            for c0 in range(0, len(members), chunk):
                block = vals[c0:c0 + chunk]
                d = np.bitwise_count(block[:, None] ^ vals[None, :])
                ii, jj = np.nonzero(d <= max_dist)
                keep = (c0 + ii) < jj
                for a, b in zip(members[c0 + ii[keep]], members[jj[keep]]):
                    pair_set.add((a, b))

    # Expand unique-hash pairs back to row pairs (incl. rows with identical pHash).
    rows_by_hash = pd.Series(np.arange(len(hashes))).groupby(inverse).apply(list).to_dict()
    out_i, out_j, out_d = [], [], []
    for u, rows in rows_by_hash.items():
        for x in range(len(rows)):
            for y in range(x + 1, len(rows)):
                out_i.append(rows[x]); out_j.append(rows[y]); out_d.append(0)
    for a, b in pair_set:
        dist = int(np.bitwise_count(uniq[a] ^ uniq[b]))
        for ra in rows_by_hash[a]:
            for rb in rows_by_hash[b]:
                out_i.append(ra); out_j.append(rb); out_d.append(dist)
    return np.array(out_i, dtype=np.int64), np.array(out_j, dtype=np.int64), np.array(out_d, dtype=np.int64)


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


# ---------------------------------------------------------------------------
# 4. Deduplication & grouping
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame, pi, pj, pd_) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = df.copy()
    df["drop_reason"] = ""
    df["duplicate_of"] = ""
    stats = {}

    def drop(mask, reason):
        m = mask & (df["drop_reason"] == "")
        df.loc[m, "drop_reason"] = reason
        stats[reason] = stats.get(reason, 0) + int(m.sum())

    drop(df["read_error"] != "", "unreadable")
    drop(df["flag"] == "augmented_copy", "augmented_copy")
    drop(df["flag"] == "unattributed", "unattributed_generator")
    drop(df["flag"] == "csv_folder_label_conflict", "csv_folder_label_conflict")

    # Representative of a duplicate cluster: canonical source first, then path.
    order = df.assign(_p=df["source_dataset"].map(SOURCE_PRIORITY)).sort_values(["_p", "raw_path"]).index
    df.loc[order, "_order"] = np.arange(len(df))

    # 4a. exact duplicates (MD5)
    active = df[df["drop_reason"] == ""]
    md5_groups = active.groupby("md5")
    conflict_md5 = md5_groups["label"].nunique()
    conflict_md5 = set(conflict_md5[conflict_md5 > 1].index)
    drop(df["md5"].isin(conflict_md5) & (df["drop_reason"] == ""), "md5_label_conflict")
    active = df[df["drop_reason"] == ""]
    rep = active.sort_values("_order").groupby("md5")["image_id"].transform("first").reindex(active.index)
    is_dup = rep != active["image_id"]
    df.loc[is_dup[is_dup].index, "duplicate_of"] = rep[is_dup]
    drop(df.index.isin(is_dup[is_dup].index), "md5_duplicate")

    # 4b. near duplicates (pHash) among survivors -- same generator only.
    # A whole-image pHash cannot see a face-only manipulation: FF++ Deepfakes /
    # FaceSwap / Face2Face of one frame, and that frame's real original, all
    # hash alike. Those are distinct samples (paired data), so cross-generator
    # near-pairs are kept and only forced into one split by assign_groups().
    alive = (df["drop_reason"] == "").to_numpy()
    lab = df["label"].to_numpy()
    gen = df["generator"].to_numpy()
    both = alive[pi] & alive[pj]
    same = both & (gen[pi] == gen[pj])
    cross = both & (lab[pi] != lab[pj])
    stats["phash_same_generator_pairs"] = int(same.sum())
    stats["phash_cross_generator_same_label_pairs (kept, grouped)"] = int((both & ~same & (lab[pi] == lab[pj])).sum())
    stats["phash_cross_label_pairs (kept, grouped)"] = int(cross.sum())

    comp = components(len(df), pi[same], pj[same])
    df["_dupcomp"] = comp
    active = df[df["drop_reason"] == ""]
    rep = active.sort_values("_order").groupby("_dupcomp")["image_id"].transform("first").reindex(active.index)
    is_dup = rep != active["image_id"]
    df.loc[is_dup[is_dup].index, "duplicate_of"] = rep[is_dup]
    drop(df.index.isin(is_dup[is_dup].index), "phash_near_duplicate")

    cross_pairs = pd.DataFrame({
        "image_a": df["image_id"].to_numpy()[pi[cross]], "generator_a": df["generator"].to_numpy()[pi[cross]],
        "image_b": df["image_id"].to_numpy()[pj[cross]], "generator_b": df["generator"].to_numpy()[pj[cross]],
        "hamming": pd_[cross],
    })
    return df.drop(columns=["_order", "_dupcomp"]), stats, cross_pairs


def _gray64(path: str) -> np.ndarray:
    a = cv2.resize(cv2.imread(os.path.join(BASE_DIR, path), cv2.IMREAD_GRAYSCALE), (64, 64),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def assign_groups(df: pd.DataFrame, pi, pj, min_ncc: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Split group = connected component over: shared group keys (identity,
    source video, FFHQ id, StyleGAN seed name), exact-MD5 matches, and pHash
    near-duplicate edges (including cross-label ones).

    A pHash edge between two rows that are NOT already in the same key/MD5
    component is only trusted if the images are also pixel-level duplicates
    (NCC of 64x64 grayscale >= min_ncc). 64-bit pHash collides on aligned,
    low-texture face crops: in the raw data it chains every Celeb-DF identity
    into one ~9k-image group and links unrelated FFHQ / StyleGAN faces, while
    genuinely derived pairs (e.g. a DFDC fake and its real source frame)
    pass the NCC check.

    Returns (group ids, mask of pHash edges used, stats).
    """
    n = len(df)
    ei, ej = [], []

    key_rows, key_ids = [], []
    key_index = {}
    for r, keys in enumerate(df["group_keys"]):
        if not keys:
            continue
        for k in keys.split("|"):
            key_rows.append(r)
            key_ids.append(n + key_index.setdefault(k, len(key_index)))
    ei.append(np.array(key_rows, dtype=np.int64))
    ej.append(np.array(key_ids, dtype=np.int64))

    md5_codes = pd.factorize(df["md5"])[0]
    first = pd.Series(np.arange(n)).groupby(md5_codes).transform("first").to_numpy()
    ei.append(np.arange(n)); ej.append(first)

    total = n + len(key_index)
    base = pd.factorize(components(total, np.concatenate(ei), np.concatenate(ej))[:n])[0]

    crossing = np.flatnonzero(base[pi] != base[pj])
    paths = df["raw_path"].to_numpy()
    cache = {}

    def vec(r):
        if r not in cache:
            cache[r] = _gray64(paths[r])
        return cache[r]

    ncc = np.array([float(vec(a) @ vec(b)) / 4096 for a, b in
                    tqdm(zip(pi[crossing], pj[crossing]), total=len(crossing),
                         desc="[VeriFace AI] verifying cross-group pHash edges", mininterval=5)])
    use = base[pi] == base[pj]
    use[crossing[ncc >= min_ncc]] = True
    stats = {"phash_edges_crossing_identity_groups": int(len(crossing)),
             f"  verified duplicates (NCC >= {min_ncc})": int((ncc >= min_ncc).sum()),
             "  rejected as hash collisions": int((ncc < min_ncc).sum())}

    # rows 0..n-1 plus one node per key/MD5 component (n + base) joined by the trusted pHash edges
    comp = components(2 * n, np.r_[np.arange(n), pi[use]], np.r_[n + base, pj[use]])[:n]
    return pd.factorize(comp)[0], use, stats


def assign_splits(kept: pd.DataFrame, seed: int) -> np.ndarray:
    """
    80/10/10 via StratifiedGroupKFold(10), stratified by generator.

    SGKF folds are not size-balanced when some groups are large (Celeb-DF
    identities, FF++ source videos), so instead of taking folds 0/1 blindly
    the two folds whose per-generator share is closest to 10% (each generator
    weighted equally) become test and val; the other eight are train.
    """
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    y = kept["generator"].to_numpy()
    folds = [te for _, te in sgkf.split(np.zeros(len(kept)), y, kept["group_id"].to_numpy())]
    gen_total = kept["generator"].value_counts()

    def score(idx):
        share = kept["generator"].iloc[idx].value_counts().reindex(gen_total.index, fill_value=0) / gen_total
        return float((share - 0.10).abs().mean())

    best = sorted(range(len(folds)), key=lambda f: score(folds[f]))
    split = np.full(len(kept), "train", dtype=object)
    split[folds[best[0]]] = "test"
    split[folds[best[1]]] = "val"
    return split


# ---------------------------------------------------------------------------
# 5. Shortcut probe (also imported by preprocess_data.py)
# ---------------------------------------------------------------------------
def run_shortcut_probe(df: pd.DataFrame, feature_sets: dict = None, split_col: str = "dataset_split") -> list:
    feature_sets = feature_sets or PROBE_FEATURES
    tr, te = df[df[split_col] == "train"], df[df[split_col] == "test"]
    ytr, yte = tr["label_numeric"].to_numpy(), te["label_numeric"].to_numpy()
    majority = max(yte.mean(), 1 - yte.mean())
    results = []
    for fs_name, cols in feature_sets.items():
        Xtr = tr[cols].to_numpy(dtype=float)
        Xte = te[cols].to_numpy(dtype=float)
        fill = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), fill, Xtr)
        Xte = np.where(np.isnan(Xte), fill, Xte)
        models = {
            "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        }
        for m_name, model in models.items():
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            pred = (p >= 0.5).astype(int)
            bacc = balanced_accuracy_score(yte, pred)
            auc = roc_auc_score(yte, p)
            results.append(dict(
                feature_set=fs_name, model=m_name, n_train=len(tr), n_test=len(te),
                accuracy=round(accuracy_score(yte, pred), 4), balanced_accuracy=round(bacc, 4),
                roc_auc=round(auc, 4), majority_baseline=round(majority, 4),
                verdict="SHORTCUT RISK" if (bacc > PROBE_MAX_BALANCED_ACC or auc > PROBE_MAX_AUC) else "ok (near chance)",
            ))
    return results


def print_probe(results: list):
    log(f"{'feature set':<36} {'model':<20} {'acc':>7} {'bal_acc':>8} {'AUC':>7}  verdict")
    for r in results:
        log(f"{r['feature_set']:<36} {r['model']:<20} {r['accuracy']:>7.4f} {r['balanced_accuracy']:>8.4f} "
            f"{r['roc_auc']:>7.4f}  {r['verdict']}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def same_name_overlap(df: pd.DataFrame) -> dict:
    """Kaggle vs GRAVEX files with the same FFHQ id / StyleGAN name: are they the same image?"""
    out = {}
    h = df.set_index("image_id")["phash"]
    for gen in ("ffhq", "stylegan"):
        sub = df[df["generator"] == gen]
        kg = sub[sub["source_dataset"] == "kaggle_140k"].drop_duplicates("filename").set_index("filename")
        gx = sub[sub["source_dataset"] == "gravex_200k"].drop_duplicates("filename").set_index("filename")
        shared = kg.index.intersection(gx.index)
        if len(shared) == 0:
            continue
        d = np.bitwise_count(kg.loc[shared, "phash"].to_numpy(np.uint64) ^ gx.loc[shared, "phash"].to_numpy(np.uint64))
        same_md5 = (kg.loc[shared, "md5"].to_numpy() == gx.loc[shared, "md5"].to_numpy())
        out[gen] = {
            "shared_filenames": int(len(shared)),
            "identical_bytes": int(same_md5.sum()),
            "phash_hamming_median": float(np.median(d)),
            "phash_hamming_le_4": int((d <= 4).sum()),
            "phash_hamming_le_10": int((d <= 10).sum()),
        }
    rng = np.random.default_rng(SEED)
    a, b = rng.integers(0, len(h), 20000), rng.integers(0, len(h), 20000)
    rd = np.bitwise_count(h.to_numpy(np.uint64)[a] ^ h.to_numpy(np.uint64)[b])
    out["random_pairs_baseline"] = {
        "median": float(np.median(rd)), "p0_1": float(np.percentile(rd, 0.1)),
        "frac_le_4": float((rd <= 4).mean()), "frac_le_10": float((rd <= 10).mean()),
    }
    return out


def dataset_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["per_source_label"] = {f"{s}/{l}": int(n) for (s, l), n in df.groupby(["source_dataset", "label"]).size().items()}
    stats["per_generator"] = (
        df.groupby(["source_dataset", "generator", "label"]).size().rename("n").reset_index()
        .to_dict(orient="records")
    )
    res = (df["width"].astype(str) + "x" + df["height"].astype(str))
    stats["resolution_top"] = {f"{s}": {k: int(v) for k, v in res[df["source_dataset"] == s].value_counts().head(5).items()}
                               for s in df["source_dataset"].unique()}
    stats["aspect_ratio"] = {s: {k: round(float(v), 4) for k, v in g["aspect_ratio"].describe().items()}
                             for s, g in df.groupby("source_dataset")}
    stats["format_mode"] = {f"{f}/{m}": int(n) for (f, m), n in df.groupby(["img_format", "img_mode"]).size().items()}
    stats["file_size_and_jpeg_q_by_generator"] = (
        df.groupby("generator").agg(n=("image_id", "size"), median_kb=("file_size", lambda s: round(s.median() / 1024, 1)),
                                    median_jpeg_luma_q=("jpeg_luma_q", "median"))
        .reset_index().to_dict(orient="records")
    )
    return stats


def write_markdown(report: dict, path: str):
    L = []
    L.append("# VeriFace AI -- Phase 1 Dataset Audit Report\n")
    L.append(f"Generated: {report['generated']}  |  pHash threshold: Hamming <= {report['phash_threshold']}  |  seed: {SEED}\n")
    L.append("Label convention: **FAKE = 1 (positive class)**, REAL = 0.\n")

    L.append("## Raw inventory\n")
    L.append("| source | label | images |\n|---|---|---:|")
    for k, v in report["raw_stats"]["per_source_label"].items():
        s, l = k.split("/")
        L.append(f"| {s} | {l} | {v:,} |")
    L.append("\n### Per generator (raw)\n")
    L.append("| source | generator | label | images |\n|---|---|---|---:|")
    for r in report["raw_stats"]["per_generator"]:
        L.append(f"| {r['source_dataset']} | {r['generator']} | {r['label']} | {r['n']:,} |")
    L.append("\n### Resolution / format\n")
    for s, d in report["raw_stats"]["resolution_top"].items():
        L.append(f"- **{s}**: " + ", ".join(f"{k} ({v:,})" for k, v in d.items()))
    L.append("- format/mode: " + ", ".join(f"{k} ({v:,})" for k, v in report["raw_stats"]["format_mode"].items()))
    L.append("\n### File size & JPEG quantisation by generator (raw)\n")
    L.append("| generator | n | median KB | median luma quant |\n|---|---:|---:|---:|")
    for r in report["raw_stats"]["file_size_and_jpeg_q_by_generator"]:
        L.append(f"| {r['generator']} | {r['n']:,} | {r['median_kb']} | {r['median_jpeg_luma_q']:.2f} |")

    L.append("\n## GRAVEX label CSV cross-check\n")
    L.append(f"- CSV code meaning REAL: `{report['csv_info']['gravex_csv_real_code']}` (VeriFace AI uses FAKE = 1; CSV labels are not used)")
    L.append(f"- folder vs CSV label conflicts: {report['csv_info']['gravex_csv_folder_label_conflicts']}")

    L.append("\n## Cross-dataset overlap (same filename in Kaggle 140k and GRAVEX)\n")
    for k, v in report["same_name_overlap"].items():
        L.append(f"- **{k}**: {json.dumps(v)}")

    L.append("\n## Deduplication\n")
    L.append("| step | images removed / pairs |\n|---|---:|")
    for k, v in report["dedup_stats"].items():
        L.append(f"| {k} | {v:,} |")
    L.append(f"\nTotal removed: **{report['total_removed']:,}** of {report['total_raw']:,}; kept **{report['total_kept']:,}**.")

    L.append("\n## Leak-free split (StratifiedGroupKFold, 10 folds -> 8/1/1)\n")
    L.append(f"- split groups: {report['split']['n_groups']:,}; largest group: {report['split']['largest_group']:,} images")
    L.append(f"- leakage checks (group / MD5 / pHash shared across splits): {report['split']['leak_checks']}")
    L.append("\n| split | REAL | FAKE | total |\n|---|---:|---:|---:|")
    for s, d in report["split"]["counts"].items():
        L.append(f"| {s} | {d.get('REAL', 0):,} | {d.get('FAKE', 0):,} | {d.get('total', 0):,} |")
    L.append("\n| generator | train | val | test |\n|---|---:|---:|---:|")
    for g, d in report["split"]["per_generator"].items():
        L.append(f"| {g} | {d.get('train', 0):,} | {d.get('val', 0):,} | {d.get('test', 0):,} |")

    L.append("\n## Metadata shortcut probe (train -> test, raw files)\n")
    L.append("| feature set | model | accuracy | balanced acc | ROC-AUC | verdict |\n|---|---|---:|---:|---:|---|")
    for r in report["probe"]:
        L.append(f"| {r['feature_set']} | {r['model']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['roc_auc']:.4f} | {r['verdict']} |")

    L.append("\n## Caveats\n")
    for c in report["caveats"]:
        L.append(f"- {c}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VeriFace AI -- Phase 1 dataset audit, dedup and re-split")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--phash-threshold", type=int, default=4)
    ap.add_argument("--min-ncc", type=float, default=0.95,
                    help="pixel NCC needed to trust a pHash edge that links two different identity groups")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rehash", action="store_true", help="ignore the cached hash index")
    args = ap.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)
    t0 = time.time()

    log("=" * 72)
    log("VeriFace AI -- Phase 1 dataset audit")
    log("=" * 72)

    # 1 + 2. Index and hash (cached)
    idx, csv_info = build_index()
    log(f"Indexed {len(idx):,} images "
        f"(kaggle_140k={int((idx.source_dataset == 'kaggle_140k').sum()):,}, "
        f"gravex_200k={int((idx.source_dataset == 'gravex_200k').sum()):,})")
    df = None
    if os.path.exists(INDEX_CACHE) and not args.rehash:
        cached = pd.read_parquet(INDEX_CACHE)
        if len(cached) == len(idx) and set(cached["raw_path"]) == set(idx["raw_path"]):
            hash_cols = ["raw_path", "md5", "width", "height", "img_format", "img_mode",
                         "file_size", "jpeg_luma_q", "phash", "read_error", "aspect_ratio"]
            df = idx.merge(cached[hash_cols], on="raw_path", how="left")
            log(f"Loaded cached hashes from {rel(INDEX_CACHE)}")
    if df is None:
        log(f"Hashing with {args.workers} workers (MD5 + pHash + metadata)...")
        df = hash_images(idx, args.workers)
        df.to_parquet(INDEX_CACHE, index=False)
    df = df.reset_index(drop=True)
    log(f"Unreadable files: {int((df.read_error != '').sum())}")

    raw_stats = dataset_stats(df[df["read_error"] == ""])
    overlap = same_name_overlap(df[df["read_error"] == ""])
    for k, v in overlap.items():
        log(f"Same-name overlap {k}: {v}")

    # 3. Near-duplicate search over every readable image
    ok = df["read_error"] == ""
    ok_idx = np.flatnonzero(ok.to_numpy())
    log(f"Searching pHash near-duplicates (Hamming <= {args.phash_threshold})...")
    ti = time.time()
    pi, pj, pdist = near_duplicate_pairs(df.loc[ok, "phash"].to_numpy(np.uint64), args.phash_threshold)
    pi, pj = ok_idx[pi], ok_idx[pj]
    log(f"  {len(pi):,} near-duplicate pairs in {time.time() - ti:.0f}s "
        f"(distance histogram: {dict(zip(*np.unique(pdist, return_counts=True)))})")

    # 4. Dedup, groups, split
    df, dedup_stats, cross_pairs = deduplicate(df, pi, pj, pdist)
    for k, v in dedup_stats.items():
        log(f"  dedup | {k:<28} {v:>8,}")
    df["group_id"], used_edges, group_stats = assign_groups(df, pi, pj, args.min_ncc)
    for k, v in group_stats.items():
        log(f"  group | {k:<40} {v:>8,}")
    dedup_stats.update(group_stats)

    kept = df[df["drop_reason"] == ""].copy().reset_index(drop=True)
    kept["label_numeric"] = kept["label"].map(LABEL_TO_NUMERIC).astype(int)
    log(f"Kept {len(kept):,} / {len(df):,} images after dedup. Assigning StratifiedGroupKFold split...")
    kept["dataset_split"] = assign_splits(kept, args.seed)

    # Leakage checks on the final split
    leak = {}
    for col in ("group_id", "md5"):
        leak[col] = int((kept.groupby(col)["dataset_split"].nunique() > 1).sum())
    alive_pos = pd.Series(np.arange(len(kept)), index=kept["image_id"])
    ids = df["image_id"].to_numpy()
    a, b = alive_pos.reindex(ids[pi]).to_numpy(), alive_pos.reindex(ids[pj]).to_numpy()
    m = ~(np.isnan(a) | np.isnan(b))
    sp = kept["dataset_split"].to_numpy()
    across = np.zeros(len(pi), dtype=bool)
    across[m] = sp[a[m].astype(int)] != sp[b[m].astype(int)]
    leak["verified_near_dup_pairs_across_splits"] = int((across & used_edges).sum())
    collisions_across = int((across & ~used_edges).sum())
    log(f"pHash collisions (NCC-rejected) spanning splits, informational: {collisions_across}")
    log(f"Leakage checks (values shared across splits, should all be 0): {leak}")
    assert all(v == 0 for v in leak.values()), f"split leakage detected: {leak}"

    counts = {}
    for s in ("train", "val", "test"):
        sub = kept[kept["dataset_split"] == s]
        counts[s] = {**{k: int(v) for k, v in sub["label"].value_counts().items()}, "total": int(len(sub))}
        log(f"  {s:<5}: {counts[s]['total']:>7,} images | REAL {counts[s].get('REAL', 0):>7,} | FAKE {counts[s].get('FAKE', 0):>7,}")
    per_gen = pd.crosstab(kept["generator"], kept["dataset_split"])
    group_sizes = kept["group_id"].value_counts()

    # 5. Shortcut probe on the new split
    log("Running metadata shortcut probe (train -> test)...")
    probe = run_shortcut_probe(kept)
    print_probe(probe)

    # 6. Export
    kept["local_image_path"] = kept["raw_path"]
    kept["phash"] = kept["phash"].map(lambda v: f"{int(v):016x}")
    cols = ["image_id", "local_image_path", "raw_path", "label", "label_numeric", "dataset_split",
            "source_dataset", "generator", "generator_family", "group_id", "orig_split",
            "width", "height", "aspect_ratio", "file_size", "jpeg_luma_q", "md5", "phash"]
    kept[cols].to_csv(OUTPUT_CSV, index=False)
    log(f"Wrote {rel(OUTPUT_CSV)} ({len(kept):,} rows)")

    removed = df[df["drop_reason"] != ""][["image_id", "raw_path", "source_dataset", "generator", "label",
                                            "drop_reason", "duplicate_of"]]
    removed.to_csv(os.path.join(REPORT_DIR, "removed_images.csv"), index=False)
    cross_pairs.to_csv(os.path.join(REPORT_DIR, "cross_label_near_duplicates.csv"), index=False)

    caveats = [
        "pHash cannot see horizontal flips or crops; GRAVEX pre-augmented copies are therefore dropped by "
        "filename provenance (augmentation_log.csv lists their originals), not by hashing.",
        "Celeb-DF fakes are grouped by target identity only; the source-face identity may appear (as REAL "
        "video of that person) in another split. Linking both collapses Celeb-DF into a single group. "
        "Celeb-DF YouTube real clips carry no identity label, so a celebrity can appear in them and in another split.",
        "DFDC fake videos cannot be linked to their real source video by filename; only pairs that are "
        "pixel-level duplicates (pHash + NCC check) are grouped, so a DFDC actor can still appear in multiple splits.",
        "pHash edges linking two different identity groups are trusted only when NCC >= --min-ncc; weaker "
        "matches were verified to be collisions between different people (Celeb-DF, FFHQ vs StyleGAN).",
        "Cross-label near-duplicate pairs are kept (both labels) but forced into the same split; review "
        "audit_reports/cross_label_near_duplicates.csv for possible label noise.",
        "Diffusion coverage is small (stable_diffusion + text_to_image_unknown + unattributed_ai); "
        "per-generator metrics for these will have wide confidence intervals.",
    ]
    report = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"), phash_threshold=args.phash_threshold,
        csv_info=csv_info, raw_stats=raw_stats, same_name_overlap=overlap,
        dedup_stats=dedup_stats, total_raw=int(len(df)), total_removed=int((df["drop_reason"] != "").sum()),
        total_kept=int(len(kept)),
        split=dict(counts=counts, n_groups=int(kept["group_id"].nunique()), largest_group=int(group_sizes.iloc[0]),
                   leak_checks=leak, per_generator={g: {s: int(v) for s, v in r.items()} for g, r in per_gen.iterrows()}),
        probe=probe, caveats=caveats,
    )
    with open(os.path.join(REPORT_DIR, "audit_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    write_markdown(report, os.path.join(REPORT_DIR, "audit_report.md"))
    log(f"Reports written to {rel(REPORT_DIR)}/ (audit_report.md, audit_report.json, removed_images.csv, "
        f"cross_label_near_duplicates.csv)")
    log(f"Done in {time.time() - t0:.0f}s. Next: python preprocess_data.py")


if __name__ == "__main__":
    main()
