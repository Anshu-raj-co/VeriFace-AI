"""Phase C leave-generator-out analysis on VALIDATION (see experiment_plan.md, Addendum 1).
Compares C_LOGO_nonGAN_holdout (SD / text-to-image / unattributed removed from train) with C_ALL_control
(same budget, all generators). Writes reports/logo_unseen_generator_val.csv."""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
OUT = r"D:\PROJECT\VeriFace AI\research_runs\generalization_audit_20261016"
RNG = np.random.default_rng(11)
HELD = ["stable_diffusion", "text_to_image_unknown", "unattributed_ai"]
SEEN = ["stylegan", "celebdf_synthesis", "dfdc_fake", "ffpp_Deepfakes", "ffpp_NeuralTextures"]


def boot_auc(neg, pos, n=1000):
    vals = [roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))],
                          np.r_[RNG.choice(neg, len(neg)), RNG.choice(pos, len(pos))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


rows = []
for arm in ["C_LOGO_nonGAN_holdout", "C_ALL_control"]:
    d = pd.read_csv(os.path.join(OUT, "experiments", arm, "val_predictions.csv"))
    real_all = d.loc[d.label_numeric == 0, "prob_fake"].to_numpy()
    ffhq = d.loc[d.generator == "ffhq", "prob_fake"].to_numpy()
    groups = [(g, [g]) for g in HELD + SEEN] + [("HELD_OUT_ALL", HELD)]
    for name, gens in groups:
        s = d.loc[d.generator.isin(gens), "prob_fake"].to_numpy()
        k = int((s >= .5).sum())
        lo, hi = wilson(k, len(s))
        a_all = roc_auc_score(np.r_[np.zeros(len(real_all)), np.ones(len(s))], np.r_[real_all, s])
        a_ffhq = roc_auc_score(np.r_[np.zeros(len(ffhq)), np.ones(len(s))], np.r_[ffhq, s])
        ci = boot_auc(real_all, s)
        rows.append({"arm": arm, "group": name, "status": "held_out_in_LOGO" if set(gens) <= set(HELD) else "seen",
                     "n": len(s), "tpr@0.5": k / len(s), "tpr_wilson_lo": lo, "tpr_wilson_hi": hi,
                     "auc_vs_all_val_real": a_all, "auc_ci_lo": ci[0], "auc_ci_hi": ci[1], "auc_vs_val_ffhq": a_ffhq,
                     "mean_p": s.mean()})
    rows.append({"arm": arm, "group": "REAL_all", "status": "-", "n": len(real_all),
                 "tpr@0.5": float((real_all >= .5).mean()), "mean_p": real_all.mean(),
                 "auc_vs_all_val_real": roc_auc_score(d.label_numeric, d.prob_fake)})
r = pd.DataFrame(rows)
r.to_csv(os.path.join(OUT, "reports", "logo_unseen_generator_val.csv"), index=False)
pd.set_option("display.width", 250)
print(r.round(4).to_string(index=False))
h = r[(r.group == "HELD_OUT_ALL")].set_index("arm")
ok = (h.loc["C_LOGO_nonGAN_holdout", "auc_vs_all_val_real"] >= 0.95 and
      h.loc["C_ALL_control", "tpr@0.5"] - h.loc["C_LOGO_nonGAN_holdout", "tpr@0.5"] <= 0.10)
print("PRE-REGISTERED RULE (AUC>=0.95 and TPR gap<=10pp):", "SUPPORTED" if ok else "NOT SUPPORTED")
