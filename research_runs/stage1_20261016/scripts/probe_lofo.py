"""
Linear-probe ALL + leave-family-out (LOFO) evaluation on frozen features (CLIP or FFT), as pre-registered in
experiment_plan.md. Fit on TRAIN rows only (minus the held-out family), evaluate on VAL. No hyper-parameter tuning.
Usage: probe_lofo.py --features clip|fft
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"D:\PROJECT\VeriFace AI"
S1 = os.path.join(BASE, r"research_runs\stage1_20261016")
RNG = np.random.default_rng(5)
FAMILIES = {
    "diffusion_t2i": ["stable_diffusion", "text_to_image_unknown", "unattributed_ai"],
    "gan": ["stylegan"],
    "faceswap_video": ["celebdf_synthesis", "dfdc_fake", "ffpp_Deepfakes", "ffpp_FaceSwap", "ffpp_FaceShifter", "ffpp_DeepFakeDetection"],
    "reenactment": ["ffpp_Face2Face", "ffpp_NeuralTextures"],
}
MATCHED_REAL = {"diffusion_t2i": ["ffhq"], "gan": ["ffhq"],
                "faceswap_video": ["celebdf_real", "celebdf_youtube_real", "dfdc_real", "ffpp_original"],
                "reenactment": ["ffpp_original"]}


def load(features, split):
    if features == "clip":
        X = np.load(os.path.join(S1, "data", "clip", f"clip_b16_{split}.npy")).astype(np.float32)
        ids = pd.read_csv(os.path.join(S1, "data", "clip", f"clip_b16_{split}_ids.csv")).image_id
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    else:
        X = np.load(os.path.join(S1, "data", "fft", f"fft_{split}.npy")).astype(np.float32)
        ids = pd.read_csv(os.path.join(S1, "data", "fft", f"fft_{split}_ids.csv")).image_id
    return X, ids


def boot_auc(neg, pos, n=500):
    v = [roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[RNG.choice(neg, len(neg)), RNG.choice(pos, len(pos))])
         for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fit_predict(Xtr, ytr, Xva):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xva))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True, choices=["clip", "fft"]); a = ap.parse_args()
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "generator", "label_numeric", "group_id", "dataset_split"])
    Xtr, idtr = load(a.features, "train"); Xva, idva = load(a.features, "val")
    tr = md.set_index("image_id").loc[idtr].reset_index(); va = md.set_index("image_id").loc[idva].reset_index()
    assert set(tr.dataset_split) == {"train"} and set(va.dataset_split) == {"val"}
    out_dir = os.path.join(S1, "experiments", f"{a.features}_probe"); os.makedirs(out_dir, exist_ok=True)
    rows = []
    for arm, held in [("ALL", [])] + list(FAMILIES.items()):
        m = ~tr.generator.isin(FAMILIES.get(arm, []))
        z = fit_predict(Xtr[m.to_numpy()], tr.label_numeric[m].to_numpy(), Xva)
        vp = va[["image_id", "generator", "label_numeric", "group_id"]].copy(); vp["logit"] = z; vp["prob_fake"] = 1 / (1 + np.exp(-z))
        vp.to_csv(os.path.join(out_dir, f"val_predictions_{arm}.csv"), index=False)
        real_all = vp.loc[vp.label_numeric == 0, "prob_fake"].to_numpy()
        base = {"features": a.features, "arm": arm, "n_train": int(m.sum()), "val_auc_overall": roc_auc_score(vp.label_numeric, vp.prob_fake)}
        for fam, gens in FAMILIES.items():
            s = vp.loc[vp.generator.isin(gens), "prob_fake"].to_numpy()
            mr = vp.loc[vp.generator.isin(MATCHED_REAL[fam]), "prob_fake"].to_numpy()
            auc_m = roc_auc_score(np.r_[np.zeros(len(mr)), np.ones(len(s))], np.r_[mr, s])
            lo, hi = boot_auc(mr, s)
            rows.append({**base, "eval_family": fam, "family_held_out": fam == arm, "n_fake": len(s), "tpr@0.5": float((s >= .5).mean()),
                         "auc_vs_all_real": roc_auc_score(np.r_[np.zeros(len(real_all)), np.ones(len(s))], np.r_[real_all, s]),
                         "auc_vs_matched_real": auc_m, "auc_matched_ci_lo": lo, "auc_matched_ci_hi": hi,
                         "matched_real_fpr@0.5": float((mr >= .5).mean()), "all_real_fpr@0.5": float((real_all >= .5).mean())})
        print(f"[probe] {a.features} {arm} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(S1, "reports", f"lofo_{a.features}_probe_val.csv"), index=False)
    pd.set_option("display.width", 250)
    print(r[["arm", "eval_family", "family_held_out", "n_fake", "tpr@0.5", "auc_vs_matched_real", "auc_matched_ci_lo", "auc_matched_ci_hi",
             "auc_vs_all_real", "all_real_fpr@0.5", "val_auc_overall"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
