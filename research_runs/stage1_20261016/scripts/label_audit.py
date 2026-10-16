"""
VeriFace AI stage 1 -- dataset / label-integrity audit (read-only; writes under OUT only).

Checks
  1. Provenance consistency: label vs. folder / filename tokens of the raw path.
  2. Exact duplicates across labels (MD5) and identical processed files.
  3. Cross-label near-duplicate pairs from audit_reports/cross_label_near_duplicates.csv, verified at
     pixel level on the RAW images: a FAKE that is pixel-identical (or nearly) to a REAL image carries no
     manipulation -> label conflict. Also measures where the difference is (centre face box vs border).
  4. Model-disagreement candidates: images whose label strongly contradicts the baseline model
     (val predictions computed in the previous study; train video-source predictions; existing test
     predictions used descriptively). These are SUSPICIOUS, not confirmed errors.
FAKE = 1, REAL = 0.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\stage1_20261016")
PREV = os.path.join(BASE, r"research_runs\generalization_audit_20261016")


def log(m):
    print(f"[label_audit] {m}", flush=True)


def load_raw(p):
    with Image.open(os.path.join(BASE, p)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def pair_stats(a, b):
    if a.shape != b.shape:
        return dict(same_shape=False)
    d = np.abs(a - b)
    mse = float((d ** 2).mean())
    h, w = d.shape[:2]
    c = d[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    border = d.copy(); border[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = np.nan
    return dict(same_shape=True, mae=float(d.mean()), max_abs=float(d.max()),
                psnr=99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
                frac_px_diff_gt8=float((d.max(axis=2) > 8).mean()),
                mae_center=float(c.mean()), mae_border=float(np.nanmean(border)))


def main():
    t0 = time.time()
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"))
    R = {}

    # 1. provenance tokens
    p = md.raw_path.str.lower()
    tok_fake = p.str.contains(r"/fake/|/ai_images/|/fake_|_fake_|ai_detect_fake|_sd_|_men_")
    tok_real = p.str.contains(r"/real/|/real_|ai_detect_real")
    md["prov_token"] = np.select([tok_fake & ~tok_real, tok_real & ~tok_fake], ["FAKE", "REAL"], "AMBIGUOUS")
    R["provenance_vs_label"] = md.groupby(["label", "prov_token"]).size().unstack(fill_value=0)
    conflict = md[(md.prov_token != "AMBIGUOUS") & (md.prov_token != md.label)]
    conflict.to_csv(os.path.join(OUT, "reports", "label_provenance_conflicts.csv"), index=False)
    R["provenance_conflicts"] = len(conflict)
    amb = md[md.prov_token == "AMBIGUOUS"]
    R["provenance_ambiguous_by_generator"] = amb.groupby("generator").size().to_dict()

    # 2. exact duplicates
    R["md5_shared_across_labels"] = int((md.groupby("md5")["label"].nunique() > 1).sum())
    R["md5_duplicates_total"] = int(md.md5.duplicated().sum())

    # 3. cross-label near duplicates, pixel verification on raw images
    cl = pd.read_csv(os.path.join(BASE, "audit_reports", "cross_label_near_duplicates.csv"))
    keep = md.set_index("image_id")
    cl = cl[cl.image_a.isin(keep.index) & cl.image_b.isin(keep.index)].copy()
    la = keep.loc[cl.image_a, "label"].to_numpy(); lb = keep.loc[cl.image_b, "label"].to_numpy()
    cl = cl[la != lb].copy()
    # orient: real, fake
    swap = keep.loc[cl.image_a, "label"].to_numpy() == "FAKE"
    cl["real_id"] = np.where(swap, cl.image_b, cl.image_a); cl["fake_id"] = np.where(swap, cl.image_a, cl.image_b)
    cl = cl.drop_duplicates(["real_id", "fake_id"]).reset_index(drop=True)
    log(f"cross-label pairs with both images kept: {len(cl):,}")
    cache, rows = {}, []
    def img(i):
        if i not in cache:
            if len(cache) > 4000:
                cache.clear()
            cache[i] = load_raw(keep.at[i, "raw_path"])
        return cache[i]
    cl = cl.sort_values("real_id")
    for k, r in enumerate(cl.itertuples()):
        s = pair_stats(img(r.real_id), img(r.fake_id))
        rows.append({"real_id": r.real_id, "fake_id": r.fake_id, "hamming": r.hamming, **s})
        if k % 20000 == 0:
            log(f"  {k:,}/{len(cl):,} pairs ({time.time() - t0:.0f}s)")
    ps = pd.DataFrame(rows)
    for side in ("real", "fake"):
        ps[f"{side}_generator"] = keep.loc[ps[f"{side}_id"], "generator"].to_numpy()
        ps[f"{side}_split"] = keep.loc[ps[f"{side}_id"], "dataset_split"].to_numpy()
    ps["class"] = np.select(
        [ps.psnr >= 45, ps.psnr >= 35, ps.mae_center > 2 * ps.mae_border],
        ["pixel_identical(>=45dB)", "near_identical(35-45dB)", "localized_center_edit"], "global_difference")
    ps.to_csv(os.path.join(OUT, "reports", "cross_label_pairs_pixel_check.csv"), index=False)
    R["cross_label_pairs_by_class"] = ps.groupby(["real_generator", "fake_generator", "class"]).size().unstack(fill_value=0)
    R["cross_label_psnr_quantiles"] = ps.groupby(["real_generator", "fake_generator"])["psnr"].describe(percentiles=[.01, .5, .99])
    sus = ps[ps.psnr >= 35]
    R["fake_images_near_identical_to_a_real"] = int(sus.fake_id.nunique())
    R["near_identical_by_fake_generator"] = sus.groupby("fake_generator")["fake_id"].nunique().to_dict()

    # 4. model-disagreement candidates
    preds = []
    v = pd.read_csv(os.path.join(PREV, "data", "preds_baseline_val.csv"))[["image_id", "prob_fake"]]; v["pred_set"] = "val"
    t = pd.read_csv(os.path.join(PREV, "data", "preds_baseline_train_videosources.csv"))[["image_id", "prob_fake"]]; t["pred_set"] = "train(video sources, fitted)"
    te = pd.read_csv(os.path.join(BASE, "evaluation_reports", "predictions_test_20261016_091737.csv"))[["image_id", "prob_fake"]]; te["pred_set"] = "test(existing, descriptive)"
    allp = pd.concat([v, t, te]).merge(md[["image_id", "label", "label_numeric", "generator", "dataset_split", "local_image_path", "raw_path"]], on="image_id")
    allp["confident_disagreement"] = np.where(allp.label_numeric == 1, allp.prob_fake < 0.02, allp.prob_fake > 0.98)
    dis = allp[allp.confident_disagreement].sort_values("prob_fake")
    dis.to_csv(os.path.join(OUT, "reports", "label_model_disagreement_candidates.csv"), index=False)
    R["model_disagreement_by_set_generator"] = (allp.groupby(["pred_set", "generator", "label"])["confident_disagreement"]
                                                .agg(["size", "sum", "mean"]).query("sum > 0"))
    # which disagreement candidates are also in a near-identical cross-label pair?
    R["disagreement_also_near_identical_pair"] = int(dis.image_id.isin(set(sus.fake_id) | set(sus.real_id)).sum())

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    for k, val in R.items():
        log(f"--- {k}"); print(val, flush=True)
    import json
    with open(os.path.join(OUT, "reports", "label_audit_summary.json"), "w") as fh:
        json.dump({k: (json.loads(val.reset_index().to_json(orient="records")) if isinstance(val, pd.DataFrame) else val)
                   for k, val in R.items()}, fh, indent=2, default=str)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
