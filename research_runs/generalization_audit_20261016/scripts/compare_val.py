"""
VeriFace AI research -- Phase E (fair comparison on VALIDATION) and Phase F (calibration /
thresholds on VALIDATION). Inputs: baseline val predictions + each arm's val_predictions.csv.
Applies the criteria pre-registered in experiment_plan.md. Writes experiment_results.csv,
reports/val_comparison_*.csv|json, reports/calibration_*.csv.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
OUT = r"D:\PROJECT\VeriFace AI\research_runs\generalization_audit_20261016"
RNG = np.random.default_rng(7)
fam = lambda g: "ffpp" if g.startswith("ffpp") else "celebdf" if g.startswith("celebdf") else "dfdc" if g.startswith("dfdc") else g


def eer_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def panel(d):
    y, p = d.label_numeric.to_numpy(), d.prob_fake.to_numpy()
    pr = p >= 0.5
    tpr, tnr = pr[y == 1].mean(), (~pr[y == 0]).mean()
    e, _ = eer_thr(y, p)
    r = {"val_roc_auc": roc_auc_score(y, p), "val_pr_auc": average_precision_score(y, p), "val_eer": e,
         "accuracy@0.5": (pr == y).mean(), "balanced_acc@0.5": (tpr + tnr) / 2,
         "real_fpr@0.5": 1 - tnr, "fake_fnr@0.5": 1 - tpr}
    for g in ["ffhq", "ffpp_original", "dfdc_real", "celebdf_real", "celebdf_youtube_real", "web_photo_real"]:
        s = d[d.generator == g]
        r[f"fpr@0.5|{g}"] = float((s.prob_fake >= .5).mean())
    for g in ["stylegan", "stable_diffusion", "celebdf_synthesis", "dfdc_fake", "ffpp_Deepfakes", "ffpp_Face2Face",
              "ffpp_FaceShifter", "ffpp_FaceSwap", "ffpp_NeuralTextures", "unattributed_ai", "text_to_image_unknown"]:
        s = d[d.generator == g]
        r[f"tpr@0.5|{g}"] = float((s.prob_fake >= .5).mean())
    for f in ["ffpp", "dfdc", "celebdf"]:
        s = d[d.generator.map(fam) == f]
        r[f"within_auc|{f}"] = roc_auc_score(s.label_numeric, s.prob_fake)
    s = d[d.generator.isin(["ffpp_original", "ffpp_NeuralTextures"])]
    r["within_auc|ffpp_NeuralTextures"] = roc_auc_score(s.label_numeric, s.prob_fake)
    fr = d[d.generator.isin(["ffpp_original", "ffpp_Face2Face", "ffpp_NeuralTextures"])]
    r["within_auc|ffpp_reenactment"] = roc_auc_score(fr.label_numeric, fr.prob_fake)
    return r


def group_boot(d, keys, n=500):
    grp = d.group_id.unique()
    by = {g: ix for g, ix in d.groupby("group_id").indices.items()}
    acc = {k: [] for k in keys}
    for _ in range(n):
        pick = RNG.choice(grp, len(grp))
        bb = d.iloc[np.concatenate([by[g] for g in pick])]
        try:
            r = panel(bb)
        except ValueError:
            continue
        for k in keys:
            acc[k].append(r[k])
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in acc.items()}


def paired_boot_diff(a, b, key, n=500):
    """CI of panel(b)[key] - panel(a)[key] resampling the same split groups for both."""
    m = a[["image_id", "group_id"]].merge(b[["image_id"]], on="image_id")
    a = a.set_index("image_id").loc[m.image_id].reset_index()
    b = b.set_index("image_id").loc[m.image_id].reset_index()
    grp = a.group_id.unique()
    by = {g: ix for g, ix in a.groupby("group_id").indices.items()}
    out = []
    for _ in range(n):
        pick = RNG.choice(grp, len(grp))
        ii = np.concatenate([by[g] for g in pick])
        try:
            out.append(panel(b.iloc[ii])[key] - panel(a.iloc[ii])[key])
        except ValueError:
            pass
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def calibration(d, name):
    """Temperature scaling + threshold table, cross-fitted across val split groups (5 folds) so
    the reported post-calibration numbers are not fitted on the rows they are scored on."""
    y, z = d.label_numeric.to_numpy(), d.logit.to_numpy()
    def ece(p, y, bins=15):
        e, edges = 0.0, np.linspace(0, 1, bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
            if m.any():
                e += m.mean() * abs(p[m].mean() - y[m].mean())
        return e
    def nll(p, y):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    def fit_T(z, y):
        Ts = np.exp(np.linspace(np.log(0.2), np.log(10), 400))
        return Ts[int(np.argmin([nll(1 / (1 + np.exp(-z / T)), y) for T in Ts]))]
    pc = np.zeros(len(z))
    for tr, te in GroupKFold(5).split(z, y, d.group_id):
        pc[te] = 1 / (1 + np.exp(-z[te] / fit_T(z[tr], y[tr])))
    p0 = 1 / (1 + np.exp(-z))
    T_all = fit_T(z, y)
    cal = {"model": name, "temperature_full_val": T_all, "ece_raw": ece(p0, y), "ece_temp_crossfit": ece(pc, y),
           "nll_raw": nll(p0, y), "nll_temp_crossfit": nll(pc, y)}
    # threshold table (raw probabilities; temperature does not change ranking)
    rows = []
    gen = d.generator.to_numpy()
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
        pr = p0 >= t
        rows.append({"model": name, "threshold": t, "real_fpr": pr[y == 0].mean(), "fake_fnr": (~pr[y == 1]).mean(),
                     "balanced_acc": (pr[y == 1].mean() + (~pr[y == 0]).mean()) / 2,
                     "ffpp_original_fpr": pr[gen == "ffpp_original"].mean(), "ffhq_fpr": pr[gen == "ffhq"].mean(),
                     "ffpp_fake_tpr": pr[(y == 1) & np.char.startswith(gen.astype(str), "ffpp")].mean()})
    # operating points chosen on val by cross-fitting: threshold for target genuine-face FPR
    ops = []
    for target in [0.01, 0.02, 0.05]:
        sel_fpr, sel_fnr, thr_list = [], [], []
        for tr, te in GroupKFold(5).split(z, y, d.group_id):
            real_tr = np.sort(p0[tr][y[tr] == 0])
            thr = real_tr[int(np.ceil((1 - target) * len(real_tr))) - 1]
            thr_list.append(thr)
            sel_fpr.append((p0[te][y[te] == 0] > thr).mean()); sel_fnr.append((p0[te][y[te] == 1] <= thr).mean())
        ops.append({"model": name, "target_real_fpr": target, "threshold_mean": float(np.mean(thr_list)),
                    "crossfit_real_fpr": float(np.mean(sel_fpr)), "crossfit_fake_fnr": float(np.mean(sel_fnr))})
    # abstention band: share of images and error rate inside / outside [lo, hi]
    band = []
    for lo, hi in [(0.2, 0.8), (0.1, 0.9), (0.05, 0.95)]:
        inb = (p0 > lo) & (p0 < hi)
        out = ~inb
        band.append({"model": name, "band": f"({lo},{hi})", "share_abstained": inb.mean(),
                     "error_rate_outside_band@0.5": float(((p0[out] >= .5) != y[out]).mean()),
                     "error_rate_inside_band@0.5": float(((p0[inb] >= .5) != y[inb]).mean()) if inb.any() else np.nan,
                     "ffpp_original_share_abstained": inb[gen == "ffpp_original"].mean()})
    return cal, pd.DataFrame(rows), pd.DataFrame(ops), pd.DataFrame(band)


def main():
    md = pd.read_csv(os.path.join(r"D:\PROJECT\VeriFace AI", "cleaned_metadata.csv"), usecols=["image_id", "group_id"])
    base = pd.read_csv(os.path.join(OUT, "data", "preds_baseline_val.csv"))
    models = {"baseline_epoch7": base[["image_id", "generator", "label_numeric", "group_id", "logit", "prob_fake"]]}
    for arm in sys.argv[1:]:
        p = os.path.join(OUT, "experiments", arm, "val_predictions.csv")
        models[arm] = pd.read_csv(p)
    keys_ci = ["val_eer", "real_fpr@0.5", "fpr@0.5|ffpp_original", "fpr@0.5|dfdc_real", "within_auc|ffpp",
               "within_auc|ffpp_NeuralTextures", "within_auc|dfdc", "balanced_acc@0.5"]
    rows, cis = [], {}
    for name, d in models.items():
        assert len(d) == 16441 and set(d.image_id) == set(base.image_id), name
        r = {"model": name, **panel(d)}
        summ = os.path.join(OUT, "experiments", name, "summary.json")
        if os.path.exists(summ):
            s = json.load(open(summ)); r.update({"train_minutes": s["train_minutes"], "vram_peak_gb": s["vram_peak_reserved_gb"],
                                                 "min_commit_avail_gb": s["min_commit_avail_gb"]})
        cis[name] = group_boot(d, keys_ci)
        for k, (lo, hi) in cis[name].items():
            r[f"{k}__ci_lo"], r[f"{k}__ci_hi"] = lo, hi
        rows.append(r)
    res = pd.DataFrame(rows)
    # pre-registered decision vs E0 control and baseline
    dec = []
    b = res.set_index("model").loc["baseline_epoch7"]
    if "E0_control" in res.model.values:
        c = res.set_index("model").loc["E0_control"]
        for _, r in res.iterrows():
            if r.model in ("baseline_epoch7", "E0_control"):
                continue
            crit = {
                "ffpp_fpr_drop_ge_15pp_vs_E0": c["fpr@0.5|ffpp_original"] - r["fpr@0.5|ffpp_original"] >= 0.15,
                "within_ffpp_auc_ge_E0_minus_0.02": r["within_auc|ffpp"] >= c["within_auc|ffpp"] - 0.02,
                "val_eer_le_baseline_plus_0.5pp": r["val_eer"] <= b["val_eer"] + 0.005,
                "ffhq_fpr_increase_le_0.5pp": r["fpr@0.5|ffhq"] - b["fpr@0.5|ffhq"] <= 0.005,
                "stylegan_tpr_drop_le_1.5pp": b["tpr@0.5|stylegan"] - r["tpr@0.5|stylegan"] <= 0.015,
                "sd_tpr_drop_le_1.5pp": b["tpr@0.5|stable_diffusion"] - r["tpr@0.5|stable_diffusion"] <= 0.015,
                "celebdf_synth_tpr_drop_le_1.5pp": b["tpr@0.5|celebdf_synthesis"] - r["tpr@0.5|celebdf_synthesis"] <= 0.015,
                "balanced_acc_ge_baseline_minus_0.5pp": r["balanced_acc@0.5"] >= b["balanced_acc@0.5"] - 0.005,
            }
            diffs = {k: paired_boot_diff(models["E0_control"], models[r.model], k)
                     for k in ["fpr@0.5|ffpp_original", "within_auc|ffpp", "val_eer", "tpr@0.5|ffpp_NeuralTextures"]}
            dec.append({"model": r.model, **crit, "ALL_CRITERIA_MET": all(crit.values()),
                        **{f"paired_diff_vs_E0_ci|{k}": v for k, v in diffs.items()}})
    res.to_csv(os.path.join(OUT, "experiment_results.csv"), index=False)
    with open(os.path.join(OUT, "reports", "val_decision.json"), "w") as fh:
        json.dump(dec, fh, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    cal_rows, thr, ops, band = [], [], [], []
    for name, d in models.items():
        c, t, o, bd = calibration(d, name)
        cal_rows.append(c); thr.append(t); ops.append(o); band.append(bd)
    pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "reports", "calibration_val.csv"), index=False)
    pd.concat(thr).to_csv(os.path.join(OUT, "reports", "threshold_table_val.csv"), index=False)
    pd.concat(ops).to_csv(os.path.join(OUT, "reports", "operating_points_val_crossfit.csv"), index=False)
    pd.concat(band).to_csv(os.path.join(OUT, "reports", "abstention_band_val.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 200)
    show = [c for c in res.columns if "__ci" not in c]
    print(res[show].set_index("model").T.round(4).to_string())
    print(json.dumps(dec, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    for k in keys_ci:
        print(k, {m: tuple(round(x, 4) for x in cis[m][k]) for m in cis})
    print(pd.DataFrame(cal_rows).round(4).to_string(index=False))
    print(pd.concat(thr).round(4).to_string(index=False))
    print(pd.concat(ops).round(4).to_string(index=False))
    print(pd.concat(band).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
