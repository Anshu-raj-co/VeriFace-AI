"""
Stage-1 analysis on VALIDATION only. Sections run only when their inputs exist.
  A. JPEG robustness (baseline, E0, J1): clean / q50 / q35 -> pre-registered J1-vs-E0 criteria.
  B. Full val panel + guardrails for every candidate model (baseline, E0, J1, CLIP-ALL probe, FFT-ALL probe).
  C. EfficientNet LOFO (corrected) vs LOFO_ALL control + pre-registered interpretation rule; side by side with
     the CLIP / FFT probes.
  D. Calibration / threshold / operating-point tables for baseline and J1.
Reuses panel()/group_boot()/calibration() from the previous study's compare_val.py (imported, not modified).
"""
import importlib.util
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = r"D:\PROJECT\VeriFace AI"
S1 = os.path.join(BASE, r"research_runs\stage1_20261016")
PREV = os.path.join(BASE, r"research_runs\generalization_audit_20261016")
spec = importlib.util.spec_from_file_location("cv", os.path.join(PREV, "scripts", "compare_val.py"))
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
pr = importlib.util.spec_from_file_location("pl", os.path.join(S1, "scripts", "probe_lofo.py"))
pl = importlib.util.module_from_spec(pr); pr.loader.exec_module(pl)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
R = os.path.join(S1, "reports")
J = os.path.join(S1, "data", "jpeg_eval")


def ex(p):
    return os.path.exists(p)


# ---------------- A. JPEG ----------------
if all(ex(os.path.join(J, f"{m}_{a}.csv")) for m in ["baseline", "E0", "J1"] for a in ["clean", "q50", "q35"]):
    rows = []
    for m in ["baseline", "E0", "J1"]:
        for a in ["clean", "q50", "q35"]:
            d = pd.read_csv(os.path.join(J, f"{m}_{a}.csv"))
            p = cv.panel(d)
            rows.append({"model": m, "arm": a, **{k: p[k] for k in ["val_roc_auc", "val_eer", "balanced_acc@0.5", "real_fpr@0.5",
                                                                      "fake_fnr@0.5", "fpr@0.5|ffhq", "tpr@0.5|stylegan",
                                                                      "tpr@0.5|stable_diffusion", "fpr@0.5|ffpp_original",
                                                                      "within_auc|ffpp", "tpr@0.5|celebdf_synthesis"]}})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(R, "jpeg_robustness_val.csv"), index=False)
    print("=== A. JPEG robustness (val) ===\n", t.round(4).to_string(index=False))
    g = t.set_index(["model", "arm"])
    drop = lambda m, k, sgn: sgn * (g.loc[(m, "q35"), k] - g.loc[(m, "clean"), k])
    dec = {"stylegan_tpr_drop_q35": {m: drop(m, "tpr@0.5|stylegan", -1) for m in ["E0", "J1", "baseline"]},
           "ffhq_fpr_rise_q35": {m: drop(m, "fpr@0.5|ffhq", +1) for m in ["E0", "J1", "baseline"]}}
    dec["J1_stylegan_drop_le_half_E0"] = bool(dec["stylegan_tpr_drop_q35"]["J1"] <= 0.5 * dec["stylegan_tpr_drop_q35"]["E0"])
    dec["J1_ffhq_rise_le_half_E0"] = bool(dec["ffhq_fpr_rise_q35"]["J1"] <= 0.5 * dec["ffhq_fpr_rise_q35"]["E0"])
    dec["J1_clean_eer_le_E0_plus_0.3pp"] = bool(g.loc[("J1", "clean"), "val_eer"] <= g.loc[("E0", "clean"), "val_eer"] + 0.003)
    dec["J1_within_ffpp_ge_E0_minus_0.02"] = bool(g.loc[("J1", "clean"), "within_auc|ffpp"] >= g.loc[("E0", "clean"), "within_auc|ffpp"] - 0.02)
    dec["J1_ALL_CRITERIA_MET"] = all(v for k, v in dec.items() if k.startswith("J1_"))
    a = pd.read_csv(os.path.join(J, "E0_q35.csv")); b = pd.read_csv(os.path.join(J, "J1_q35.csv"))
    dec["paired_ci_J1_minus_E0_q35|val_eer"] = cv.paired_boot_diff(a, b, "val_eer")
    dec["paired_ci_J1_minus_E0_q35|real_fpr@0.5"] = cv.paired_boot_diff(a, b, "real_fpr@0.5")
    json.dump(dec, open(os.path.join(R, "jpeg_decision.json"), "w"), indent=2, default=float)
    print(json.dumps(dec, indent=1, default=float))

# ---------------- B. Full val panel for candidates ----------------
cands = {"baseline_epoch7": os.path.join(PREV, "data", "preds_baseline_val.csv"),
         "E0_control_prev": os.path.join(PREV, "experiments", "E0_control", "val_predictions.csv"),
         "J1_jpeg_aug": os.path.join(S1, "experiments", "J1_jpeg_aug", "val_predictions.csv"),
         "CLIP_B16_probe_ALL": os.path.join(S1, "experiments", "clip_probe", "val_predictions_ALL.csv"),
         "FFT_probe_ALL": os.path.join(S1, "experiments", "fft_probe", "val_predictions_ALL.csv")}
keys = ["val_eer", "fpr@0.5|ffpp_original", "within_auc|ffpp", "real_fpr@0.5"]
rows, base = [], None
for n, p in cands.items():
    if not ex(p):
        continue
    d = pd.read_csv(p)
    r = {"model": n, **cv.panel(d)}
    ci = cv.group_boot(d, keys, n=300)
    for k, (lo, hi) in ci.items():
        r[f"{k}__ci_lo"], r[f"{k}__ci_hi"] = lo, hi
    rows.append(r)
P = pd.DataFrame(rows).set_index("model")
b = P.loc["baseline_epoch7"]
P["guard_val_eer_le_base+0.5pp"] = P.val_eer <= b.val_eer + 0.005
P["guard_ffhq_fpr_rise_le_0.5pp"] = P["fpr@0.5|ffhq"] - b["fpr@0.5|ffhq"] <= 0.005
for g_ in ["stylegan", "stable_diffusion", "celebdf_synthesis"]:
    P[f"guard_{g_}_tpr_drop_le_1.5pp"] = b[f"tpr@0.5|{g_}"] - P[f"tpr@0.5|{g_}"] <= 0.015
P.to_csv(os.path.join(R, "stage1_val_panel.csv"))
print("=== B. val panel ===\n", P[[c for c in P.columns if "__ci" not in c]].T.round(4).to_string())

# ---------------- C. LOFO (EfficientNet, corrected) ----------------
E = os.path.join(S1, "experiments")
arms = {"ALL": "LOFO_ALL_control", "diffusion_t2i": "LOFO_diffusion_t2i", "faceswap_video": "LOFO_faceswap_video",
        "reenactment": "LOFO_reenactment", "gan": "LOFO_gan"}
lrows = []
for arm, d_ in arms.items():
    f = os.path.join(E, d_, "val_predictions.csv"); s = os.path.join(E, d_, "summary.json")
    if not (ex(f) and ex(s)):
        continue
    sm = json.load(open(s))
    if sm["steps_done"] != sm["steps_planned"] or sm.get("smoke_test"):
        print(f"SKIP {d_}: incomplete run {sm}"); continue
    vp = pd.read_csv(f)
    real_all = vp.loc[vp.label_numeric == 0, "prob_fake"].to_numpy()
    for fam, gens in pl.FAMILIES.items():
        s_ = vp.loc[vp.generator.isin(gens), "prob_fake"].to_numpy()
        mr = vp.loc[vp.generator.isin(pl.MATCHED_REAL[fam]), "prob_fake"].to_numpy()
        lo, hi = pl.boot_auc(mr, s_)
        lrows.append({"features": "efficientnet_b0_2ep", "arm": arm, "eval_family": fam, "family_held_out": fam == arm,
                      "steps": sm["steps_done"], "n_fake": len(s_), "tpr@0.5": float((s_ >= .5).mean()),
                      "auc_vs_matched_real": roc_auc_score(np.r_[np.zeros(len(mr)), np.ones(len(s_))], np.r_[mr, s_]),
                      "auc_matched_ci_lo": lo, "auc_matched_ci_hi": hi,
                      "auc_vs_all_real": roc_auc_score(np.r_[np.zeros(len(real_all)), np.ones(len(s_))], np.r_[real_all, s_]),
                      "all_real_fpr@0.5": float((real_all >= .5).mean()), "val_auc_overall": roc_auc_score(vp.label_numeric, vp.prob_fake)})
if lrows:
    L = pd.DataFrame(lrows); L.to_csv(os.path.join(R, "lofo_efficientnet_val.csv"), index=False)
    comb = [L]
    for f in ["clip", "fft"]:
        p = os.path.join(R, f"lofo_{f}_probe_val.csv")
        if ex(p):
            comb.append(pd.read_csv(p).assign(features=f"{f}_probe"))
    C = pd.concat(comb, ignore_index=True)
    held = C[C.family_held_out].set_index(["features", "eval_family"])
    allc = C[(C.arm == "ALL")].set_index(["features", "eval_family"])
    V = held[["tpr@0.5", "auc_vs_matched_real", "auc_matched_ci_lo", "auc_matched_ci_hi"]].join(
        allc[["auc_vs_matched_real", "tpr@0.5"]].rename(columns=lambda c: f"ALL_{c}"))
    V["gap_auc"] = V.ALL_auc_vs_matched_real - V.auc_vs_matched_real
    V["verdict"] = np.where((V.auc_vs_matched_real >= .90) & (V.gap_auc <= .05), "supported",
                            np.where(V.auc_vs_matched_real >= .75, "partial", "not supported"))
    V.to_csv(os.path.join(R, "lofo_verdicts_all_models.csv"))
    print("=== C. LOFO verdicts ===\n", V.round(4).to_string())

# ---------------- D. Calibration / operating points ----------------
cal, thr, ops, band = [], [], [], []
for n in ["baseline_epoch7", "J1_jpeg_aug", "E0_control_prev"]:
    if ex(cands[n]):
        c, t, o, bd = cv.calibration(pd.read_csv(cands[n]), n)
        cal.append(c); thr.append(t); ops.append(o); band.append(bd)
pd.DataFrame(cal).to_csv(os.path.join(R, "calibration_val_stage1.csv"), index=False)
pd.concat(thr).to_csv(os.path.join(R, "threshold_table_val_stage1.csv"), index=False)
pd.concat(ops).to_csv(os.path.join(R, "operating_points_val_stage1.csv"), index=False)
pd.concat(band).to_csv(os.path.join(R, "abstention_band_val_stage1.csv"), index=False)
print("=== D. calibration ===\n", pd.DataFrame(cal).round(4).to_string(index=False))
print(pd.concat(ops).round(4).to_string(index=False))
