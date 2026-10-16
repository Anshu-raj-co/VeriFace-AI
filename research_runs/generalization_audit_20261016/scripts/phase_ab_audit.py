"""
VeriFace AI research -- Phase A (artifact / split / leakage audit) and Phase B
(descriptive error analysis of the EXISTING locked test predictions).

Read-only with respect to the project: reads cleaned_metadata.csv, the
existing predictions CSV and dataset images; writes only under OUT_DIR.
The test predictions are used for description only -- nothing computed here
feeds model selection, thresholds or augmentation choices.

FAKE = 1 (positive), REAL = 0.
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\generalization_audit_20261016")
PRED = os.path.join(BASE, r"evaluation_reports\predictions_test_20261016_091737.csv")
RNG = np.random.default_rng(42)
N_BOOT = 1000


def log(m=""):
    print(f"[audit] {m}", flush=True)


# ---------------------------------------------------------------------------
# Group keys parsed from raw filenames (finer than the split group_id)
# ---------------------------------------------------------------------------
FFPP_FAKE = re.compile(r"fake_(Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures)_(\d+)_(\d+)_\d+$")
FFPP_REAL = re.compile(r"real_(\d+)_\d+$")
DFD = re.compile(r"fake_DeepFakeDetection_(\d+)_(\d+)__")
CDF_FAKE = re.compile(r"celebdf_[a-z]+_id(\d+)_id(\d+)_(\d+)_frame")
CDF_REAL = re.compile(r"celebdf_[a-z]+_id(\d+)_(\d+)_frame")
CDF_YT = re.compile(r"celebdf_[a-z]+_(\d+)_frame")
DFDC = re.compile(r"dfdc_[a-z]+_(?:fake_)?([a-z]+)_\d+_\d+$")


def parse_keys(raw_path):
    """Returns (video_key, identity_keys list) -- identity keys include every
    person whose face/video contributed (target AND source for swaps)."""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    if m := FFPP_FAKE.match(stem):
        return f"ffpp_fake:{m[1]}:{m[2]}_{m[3]}", [f"ffpp_vid:{m[2]}", f"ffpp_vid:{m[3]}"]
    if m := FFPP_REAL.match(stem):
        return f"ffpp_real:{m[1]}", [f"ffpp_vid:{m[1]}"]
    if m := DFD.match(stem):
        return f"dfd:{stem.rsplit('_', 1)[0]}", [f"dfd_actor:{m[1]}", f"dfd_actor:{m[2]}"]
    if m := CDF_FAKE.match(stem):
        return f"cdf_fake:{m[1]}_{m[2]}_{m[3]}", [f"cdf_id:{m[1]}", f"cdf_id:{m[2]}"]
    if m := CDF_REAL.match(stem):
        return f"cdf_real:{m[1]}_{m[2]}", [f"cdf_id:{m[1]}"]
    if m := CDF_YT.match(stem):
        return f"cdf_yt:{m[1]}", [f"cdf_yt:{m[1]}"]
    if m := DFDC.match(stem):
        return f"dfdc:{m[1]}", [f"dfdc_vid:{m[1]}"]
    return f"img:{stem}", []


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def cluster_boot(df, clusters, stat, n=N_BOOT):
    """Percentile CI resampling whole clusters (videos / split groups)."""
    keys = df[clusters].to_numpy()
    uniq, inv = np.unique(keys, return_inverse=True)
    idx_by = [[] for _ in range(len(uniq))]
    for i, g in enumerate(inv):
        idx_by[g].append(i)
    idx_by = [np.asarray(v) for v in idx_by]
    vals = []
    for _ in range(n):
        pick = RNG.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by[p] for p in pick])
        try:
            v = stat(df.iloc[ii])
        except ValueError:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n * 0.5:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def binary_block(y, s, thr=0.5):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sd = lambda a, b: a / b if b else np.nan
    prec_f, rec_f = sd(tp, tp + fp), sd(tp, tp + fn)
    prec_r, rec_r = sd(tn, tn + fn), sd(tn, tn + fp)
    f1 = lambda a, b: 2 * a * b / (a + b) if a + b else np.nan
    return {"n": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": sd(tp + tn, len(y)), "balanced_accuracy": np.nanmean([rec_f, rec_r]),
            "precision_fake": prec_f, "recall_fake": rec_f, "f1_fake": f1(prec_f, rec_f),
            "precision_real": prec_r, "recall_real_specificity": rec_r, "f1_real": f1(prec_r, rec_r),
            "fpr_real_flagged": sd(fp, fp + tn), "fnr_fake_missed": sd(fn, fn + tp)}


def subgroup_table(d, col, real_ref_all, cluster="video_key"):
    """Per-subgroup correctness (single-class groups) with Wilson + cluster-bootstrap CI,
    probability distribution, and separability (AUC/EER) of each FAKE group against
    (a) all test REAL and (b) the REAL images of the SAME source family, if any."""
    rows = []
    for (val, lab), sub in d.groupby([col, "label"]):
        s = sub["prob_fake"].to_numpy()
        fake = lab == "FAKE"
        correct = (s >= 0.5) if fake else (s < 0.5)
        k, n = int(correct.sum()), len(sub)
        lo, hi = wilson(k, n)
        blo, bhi = cluster_boot(sub, cluster, lambda x: float(((x.prob_fake >= 0.5) if fake else (x.prob_fake < 0.5)).mean()))
        r = {col: val, "label": lab, "n": n, "n_videos_or_clusters": sub[cluster].nunique(),
             "n_split_groups": sub["group_id"].nunique(),
             "correct_rate@0.5": k / n, "wilson_lo": lo, "wilson_hi": hi, "cluster_boot_lo": blo, "cluster_boot_hi": bhi,
             "errors": n - k, "mean_p": s.mean(), "median_p": float(np.median(s)),
             "p10": float(np.percentile(s, 10)), "p90": float(np.percentile(s, 90))}
        if fake:
            y = np.r_[np.zeros(len(real_ref_all)), np.ones(n)]
            sc = np.r_[real_ref_all, s]
            r["auc_vs_all_real"] = roc_auc_score(y, sc)
            r["eer_vs_all_real"] = eer(y, sc)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"))
    keys = md["raw_path"].map(parse_keys)
    md["video_key"] = [k[0] for k in keys]
    md["identity_keys"] = [k[1] for k in keys]
    md["family_source"] = md["generator"].map(lambda g: "ffpp" if g.startswith("ffpp") else
                                              "celebdf" if g.startswith("celebdf") else
                                              "dfdc" if g.startswith("dfdc") else g)
    A = {}

    # ---------------- Phase A: counts ----------------
    A["split_counts"] = md.groupby(["dataset_split", "label"]).size().unstack().to_dict()
    A["generator_split_counts"] = md.pivot_table(index=["generator", "label"], columns="dataset_split",
                                                 values="image_id", aggfunc="size", fill_value=0)
    A["source_split_counts"] = md.pivot_table(index=["source_dataset", "label"], columns="dataset_split",
                                              values="image_id", aggfunc="size", fill_value=0)
    A["family_split_counts"] = md.pivot_table(index=["generator_family", "label"], columns="dataset_split",
                                              values="image_id", aggfunc="size", fill_value=0)

    # Training-set label prior inside each source family (a source-level shortcut risk)
    tr = md[md.dataset_split == "train"]
    prior = tr.groupby("family_source")["label_numeric"].agg(["size", "mean"]).rename(
        columns={"size": "n_train", "mean": "train_frac_fake"})
    A["train_label_prior_by_source_family"] = prior

    # Preprocessing trace by generator (upscaling = small source face, cubic-enlarged)
    trace = md.groupby(["generator", "label"]).agg(
        n=("image_id", "size"), upscaled_frac=("upscaled", "mean"), crop_side_median=("crop_side", "median"),
        face_detected_frac=("face_detected", "mean"), proc_kb_median=("proc_file_size", lambda s: s.median() / 1024),
        raw_kb_median=("file_size", lambda s: s.median() / 1024))
    A["preprocessing_trace"] = trace
    up_tr = tr.groupby(["label", "upscaled"]).size().unstack(fill_value=0)
    A["train_upscaled_by_label"] = up_tr
    A["train_fake_frac_given_upscaled"] = tr.groupby("upscaled")["label_numeric"].mean().to_dict()

    # ---------------- Phase A: leakage ----------------
    leak = {}
    # 1. split group ids crossing splits
    leak["group_id_in_multiple_splits"] = int((md.groupby("group_id")["dataset_split"].nunique() > 1).sum())
    # 2. exact md5 across splits
    leak["md5_in_multiple_splits"] = int((md.groupby("md5")["dataset_split"].nunique() > 1).sum())
    # 3. exact raw pHash across splits
    leak["phash_exact_in_multiple_splits"] = int((md.groupby("phash")["dataset_split"].nunique() > 1).sum())
    # 4. near-duplicate pHash (Hamming <= 4) across splits via pigeonhole banding
    ph = np.array([int(h, 16) for h in md["phash"]], dtype=np.uint64)
    split_arr = md["dataset_split"].to_numpy()
    bands = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 64)]
    cand = set()
    for lo, hi in bands:
        mask = np.uint64((1 << (hi - lo)) - 1)
        b = (ph >> np.uint64(lo)) & mask
        order = np.argsort(b, kind="stable")
        bs = b[order]
        starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
        ends = np.r_[starts[1:], len(bs)]
        for st, en in zip(starts, ends):
            if en - st < 2 or en - st > 400:   # huge buckets = low-texture collisions; counted below
                continue
            ids = order[st:en]
            sp = split_arr[ids]
            if len(set(sp)) < 2:
                continue
            for a in range(len(ids)):
                for c in range(a + 1, len(ids)):
                    if sp[a] != sp[c]:
                        i, j = ids[a], ids[c]
                        cand.add((min(i, j), max(i, j)))
    near = [(i, j) for i, j in cand if bin(int(ph[i] ^ ph[j])).count("1") <= 4]
    leak["phash_le4_pairs_across_splits_raw"] = len(near)
    near_df = pd.DataFrame([{"a": md.image_id[i], "b": md.image_id[j], "gen_a": md.generator[i], "gen_b": md.generator[j],
                             "split_a": md.dataset_split[i], "split_b": md.dataset_split[j],
                             "label_a": md.label[i], "label_b": md.label[j],
                             "hamming": bin(int(ph[i] ^ ph[j])).count("1")} for i, j in near])
    near_df.to_csv(os.path.join(OUT, "data", "phash_le4_pairs_across_splits.csv"), index=False)
    if len(near_df):
        leak["phash_le4_across_by_generator_pair"] = near_df.groupby(["gen_a", "gen_b"]).size().sort_values(ascending=False).head(20).to_dict()
    # 5. video / identity keys across splits (finer than group_id; includes Celeb-DF source identity)
    ex = md[["image_id", "dataset_split", "generator", "identity_keys"]].explode("identity_keys").dropna()
    idk = ex.groupby("identity_keys")["dataset_split"].nunique()
    crossing = idk[idk > 1]
    leak["identity_or_video_keys_total"] = int(len(idk))
    leak["identity_or_video_keys_in_multiple_splits"] = int(len(crossing))
    leak["crossing_keys_by_prefix"] = crossing.index.str.split(":").str[0].value_counts().to_dict()
    cx = ex[ex.identity_keys.isin(crossing.index)]
    leak["images_touching_crossing_keys_by_split_generator"] = cx.groupby(["dataset_split", "generator"]).size().to_dict()
    vk = md.groupby("video_key")["dataset_split"].nunique()
    leak["video_keys_in_multiple_splits"] = int((vk > 1).sum())
    A["leakage"] = leak

    # ---------------- Phase B: join predictions ----------------
    pr = pd.read_csv(PRED)
    j = pr.merge(md, on="image_id", how="left", suffixes=("", "_md"), validate="one_to_one")
    join = {"pred_rows": len(pr), "unmatched": int(j["dataset_split"].isna().sum()),
            "path_mismatch": int((j["local_image_path"] != j["local_image_path_md"]).sum()),
            "label_mismatch": int((j["label_numeric"] != j["label_numeric_md"]).sum()),
            "non_test_rows": int((j["dataset_split"] != "test").sum()),
            "test_rows_in_metadata": int((md.dataset_split == "test").sum())}
    A["prediction_join"] = join
    if join["unmatched"] or join["path_mismatch"] or join["label_mismatch"] or join["non_test_rows"]:
        log(f"JOIN PROBLEM: {join}")
        sys.exit(2)
    d = j
    y, s = d["label_numeric"].to_numpy(), d["prob_fake"].to_numpy()

    B = {}
    overall = binary_block(y, s)
    overall.update({"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "eer": eer(y, s)})
    for name, fn in {"accuracy": lambda x: float(((x.prob_fake >= .5) == x.label_numeric).mean()),
                     "roc_auc": lambda x: roc_auc_score(x.label_numeric, x.prob_fake),
                     "eer": lambda x: eer(x.label_numeric, x.prob_fake),
                     "fpr_real": lambda x: float((x.prob_fake[x.label_numeric == 0] >= .5).mean()),
                     "fnr_fake": lambda x: float((x.prob_fake[x.label_numeric == 1] < .5).mean())}.items():
        overall[f"{name}_ci95_group_boot"] = cluster_boot(d, "group_id", fn)
    B["overall_test@0.5"] = overall

    real_all = d.loc[d.label_numeric == 0, "prob_fake"].to_numpy()
    tabs = {}
    for col in ["generator", "generator_family", "source_dataset", "family_source"]:
        tabs[col] = subgroup_table(d, col, real_all)
        tabs[col].to_csv(os.path.join(OUT, "reports", f"test_subgroups_by_{col}.csv"), index=False)

    # Within-source separability: fakes of a source family vs REAL of the SAME family
    ws = []
    for fam in ["ffpp", "celebdf", "dfdc"]:
        sub = d[d.family_source == fam]
        real_s = sub.loc[sub.label_numeric == 0, "prob_fake"].to_numpy()
        for gen, gs in sub[sub.label_numeric == 1].groupby("generator"):
            yy = np.r_[np.zeros(len(real_s)), np.ones(len(gs))]
            ss = np.r_[real_s, gs.prob_fake.to_numpy()]
            both = pd.concat([sub[sub.label_numeric == 0], gs])
            ws.append({"family": fam, "fake_generator": gen, "n_real_same_source": len(real_s), "n_fake": len(gs),
                       "auc_vs_same_source_real": roc_auc_score(yy, ss), "eer_vs_same_source_real": eer(yy, ss),
                       "auc_ci95_video_boot": cluster_boot(both, "group_id",
                                                           lambda x: roc_auc_score(x.label_numeric, x.prob_fake)),
                       "balanced_acc@0.5": binary_block(yy, ss)["balanced_accuracy"]})
        allf = sub
        yy, ss = allf.label_numeric.to_numpy(), allf.prob_fake.to_numpy()
        blk = binary_block(yy, ss)
        ws.append({"family": fam, "fake_generator": "ALL", "n_real_same_source": int((yy == 0).sum()), "n_fake": int(yy.sum()),
                   "auc_vs_same_source_real": roc_auc_score(yy, ss), "eer_vs_same_source_real": eer(yy, ss),
                   "auc_ci95_video_boot": cluster_boot(allf, "group_id", lambda x: roc_auc_score(x.label_numeric, x.prob_fake)),
                   "balanced_acc@0.5": blk["balanced_accuracy"], "test_frac_fake": yy.mean()})
    ws = pd.DataFrame(ws)
    ws.to_csv(os.path.join(OUT, "reports", "test_within_source_separability.csv"), index=False)
    B["within_source_separability"] = ws

    # Error drivers (descriptive): REAL false-positive rate vs preprocessing trace
    real = d[d.label_numeric == 0].copy()
    real["fp"] = real.prob_fake >= 0.5
    real["crop_bin"] = pd.cut(real.crop_side, [0, 100, 140, 200, 255, 10_000], labels=["<100", "100-139", "140-199", "200-255", ">=256"])
    drv = {}
    drv["real_fpr_by_upscaled"] = real.groupby("upscaled")["fp"].agg(["size", "mean"])
    drv["real_fpr_by_generator_upscaled"] = real.groupby(["generator", "upscaled"])["fp"].agg(["size", "mean"])
    drv["real_fpr_by_crop_bin"] = real.groupby("crop_bin", observed=True)["fp"].agg(["size", "mean"])
    drv["real_fpr_by_face_detected"] = real.groupby("face_detected")["fp"].agg(["size", "mean"])
    real["jq_bin"] = pd.cut(real.jpeg_quality, [59, 70, 80, 90, 100])
    drv["real_fpr_by_jpeg_quality"] = real.groupby("jq_bin", observed=True)["fp"].agg(["size", "mean"])
    fake = d[d.label_numeric == 1].copy()
    fake["fn"] = fake.prob_fake < 0.5
    drv["fake_fnr_by_generator_upscaled"] = fake.groupby(["generator", "upscaled"])["fn"].agg(["size", "mean"])
    ffo = d[d.generator == "ffpp_original"]
    drv["ffpp_original_by_video"] = ffo.groupby("video_key")["prob_fake"].agg(["size", "mean", lambda s: (s >= .5).mean()]).rename(
        columns={"<lambda_0>": "fp_rate"}).sort_values("mean", ascending=False)
    drv["ffpp_original_videos_summary"] = {
        "n_videos": int(ffo.video_key.nunique()),
        "videos_all_frames_fp": int((drv["ffpp_original_by_video"].fp_rate == 1).sum()),
        "videos_no_frames_fp": int((drv["ffpp_original_by_video"].fp_rate == 0).sum()),
        "spearman_prob_vs_crop_side": float(ffo[["prob_fake", "crop_side"]].corr("spearman").iloc[0, 1])}
    # Is the per-video probability consistent? (intra-class correlation proxy: var between / total)
    vm = ffo.groupby("video_key")["prob_fake"].transform("mean")
    drv["ffpp_original_between_video_var_share"] = float(vm.var() / ffo.prob_fake.var())
    B["error_drivers"] = drv

    # Label-noise check: test errors that sit in cross-label near-duplicate groups from the audit
    cl = pd.read_csv(os.path.join(BASE, "audit_reports", "cross_label_near_duplicates.csv"), nrows=5)
    B["cross_label_csv_columns"] = list(cl.columns)

    # ---------------- Threshold-dependence (descriptive) ----------------
    thr_rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.6412, 0.7, 0.8, 0.9]:
        b = binary_block(y, s, t)
        ffo_fp = float((ffo.prob_fake >= t).mean())
        nt = d[d.generator == "ffpp_NeuralTextures"].prob_fake
        thr_rows.append({"threshold": t, **{k: b[k] for k in ["accuracy", "balanced_accuracy", "fpr_real_flagged", "fnr_fake_missed"]},
                         "ffpp_original_fpr": ffo_fp, "neuraltextures_tpr": float((nt >= t).mean())})
    B["test_threshold_sweep_descriptive_only"] = pd.DataFrame(thr_rows)

    # ---------------- Save ----------------
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return json.loads(o.reset_index().to_json(orient="records"))
        if isinstance(o, pd.Series):
            return json.loads(o.to_json())
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (tuple, list)):
            return [conv(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        return o
    with open(os.path.join(OUT, "reports", "phase_a_audit.json"), "w") as fh:
        json.dump(conv(A), fh, indent=2, default=str)
    with open(os.path.join(OUT, "reports", "phase_b_error_analysis.json"), "w") as fh:
        json.dump(conv(B), fh, indent=2, default=str)
    md[["image_id", "video_key"]].to_csv(os.path.join(OUT, "data", "video_keys.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    for k, v in A.items():
        log(f"--- A.{k}"); print(v if not isinstance(v, dict) else json.dumps(conv(v), indent=1)[:4000])
    for k, v in B.items():
        if k == "error_drivers":
            for kk, vv in v.items():
                log(f"--- B.error_drivers.{kk}"); print(vv.head(40) if isinstance(vv, pd.DataFrame) else vv)
        else:
            log(f"--- B.{k}"); print(v if not isinstance(v, dict) else json.dumps(conv(v), indent=1))
    for col, t in tabs.items():
        log(f"--- test subgroups by {col}"); print(t.round(4).to_string(index=False))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
