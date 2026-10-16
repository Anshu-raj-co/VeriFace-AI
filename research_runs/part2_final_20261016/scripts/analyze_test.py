"""
Part 2 test-set analysis (pre-declared in PREDECLARATION.md; one shot; no thresholds, temperature or weights are
fitted here). Inputs: evaluation_reports/predictions_test_20261016_091737.csv (primary baseline, evaluate.py) and
data/test_arms/{baseline,J1}_{clean,q50,q35}.csv (secondary comparison).

Threshold labelling: "@0.5" = conventional threshold on the raw sigmoid score; "app_" = validation-derived app policy
(raw score < 0.30 LIKELY AUTHENTIC, >= 0.90 LIKELY MANIPULATED, else INCONCLUSIVE); threshold-free: AUC, PR-AUC, EER.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

BASE = r"D:\PROJECT\VeriFace AI"
P2 = os.path.join(BASE, r"research_runs\part2_final_20261016")
R = os.path.join(P2, "reports")
A = os.path.join(P2, "data", "test_arms")
REAL_BELOW, FAKE_AT, T = 0.30, 0.90, 1.5523
RNG = np.random.default_rng(2026)
NB = 500


def eer(y, s):
    """Same definition as evaluate.equal_error_rate (linear interpolation between the closest ROC points)."""
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    if i + 1 < len(fpr) and (fpr[i] - fnr[i]) * (fpr[i + 1] - fnr[i + 1]) < 0:
        d0, d1 = fpr[i] - fnr[i], fpr[i + 1] - fnr[i + 1]
        t = d0 / (d0 - d1)
        return float(fpr[i] + t * (fpr[i + 1] - fpr[i]))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    real, fake = y == 0, y == 1
    m = {"n": len(y), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
         "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "eer": eer(y, p),
         "accuracy@0.5": (tp + tn) / len(y), "precision_fake@0.5": tp / max(1, tp + fp), "recall_fake@0.5": tp / max(1, tp + fn),
         "f1_fake@0.5": 2 * tp / max(1, 2 * tp + fp + fn), "genuine_fpr@0.5": fp / max(1, fp + tn), "fake_miss@0.5": fn / max(1, fn + tp),
         "tn@0.5": tn, "fp@0.5": fp, "fn@0.5": fn, "tp@0.5": tp}
    man, auth = p >= FAKE_AT, p < REAL_BELOW
    inc = ~(man | auth)
    m.update({"app_genuine_called_manipulated": man[real].mean(), "app_genuine_inconclusive": inc[real].mean(),
              "app_genuine_called_authentic": auth[real].mean(), "app_fake_called_authentic": auth[fake].mean(),
              "app_fake_inconclusive": inc[fake].mean(), "app_fake_called_manipulated": man[fake].mean(),
              "app_inconclusive_share": inc.mean(),
              "app_precision_of_MANIPULATED": (man & fake).sum() / max(1, man.sum()),
              "app_precision_of_AUTHENTIC": (auth & real).sum() / max(1, auth.sum()),
              "app_accuracy_on_decided": ((man & fake) | (auth & real)).sum() / max(1, (man | auth).sum()),
              "app_counts_real[auth,inc,man]": [int(auth[real].sum()), int(inc[real].sum()), int(man[real].sum())],
              "app_counts_fake[auth,inc,man]": [int(auth[fake].sum()), int(inc[fake].sum()), int(man[fake].sum())]})
    return m


BOOT_KEYS = ["roc_auc", "eer", "accuracy@0.5", "genuine_fpr@0.5", "fake_miss@0.5", "app_genuine_called_manipulated",
             "app_fake_called_authentic", "app_inconclusive_share"]


def fast(y, p):
    real, fake = y == 0, y == 1
    man, auth = p >= FAKE_AT, p < REAL_BELOW
    return {"roc_auc": roc_auc_score(y, p), "eer": eer(y, p), "accuracy@0.5": ((p >= .5) == (y == 1)).mean(),
            "genuine_fpr@0.5": (p[real] >= .5).mean(), "fake_miss@0.5": (p[fake] < .5).mean(),
            "app_genuine_called_manipulated": man[real].mean(), "app_fake_called_authentic": auth[fake].mean(),
            "app_inconclusive_share": (~(man | auth)).mean()}


def boot_index(groups):
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(u))]
    for _ in range(NB):
        pick = RNG.integers(0, len(u), len(u))
        yield np.concatenate([members[k] for k in pick])


def main():
    out = {}
    base_eval = pd.read_csv(os.path.join(BASE, "evaluation_reports", "predictions_test_20261016_091737.csv"))
    arms = {}
    for m in ["baseline", "J1"]:
        for a in ["clean", "q50", "q35"]:
            f = os.path.join(A, f"{m}_{a}.csv")
            if not os.path.exists(f):
                sys.exit(f"missing {f}")
            arms[(m, a)] = pd.read_csv(f).sort_values("image_id").reset_index(drop=True)
    ref = arms[("baseline", "clean")]
    for k, d in arms.items():
        assert (d.image_id.to_numpy() == ref.image_id.to_numpy()).all() and (d.label_numeric == ref.label_numeric).all()
    # pipeline sanity: fresh baseline clean arm vs stored evaluate.py predictions
    chk = ref.merge(base_eval[["image_id", "prob_fake"]], on="image_id", suffixes=("", "_eval"))
    dp = (chk.prob_fake - chk.prob_fake_eval).abs()
    out["sanity_baseline_clean_vs_evaluate_py"] = {"n_matched": len(chk), "max_abs_dprob": float(dp.max()),
                                                   "decisions_differ@0.5": int(((chk.prob_fake >= .5) != (chk.prob_fake_eval >= .5)).sum()),
                                                   "decisions_differ_app": int((np.digitize(chk.prob_fake, [REAL_BELOW, FAKE_AT]) !=
                                                                                np.digitize(chk.prob_fake_eval, [REAL_BELOW, FAKE_AT])).sum())}

    # --- primary: baseline clean from evaluate.py predictions ---
    be = base_eval.sort_values("image_id").reset_index(drop=True)
    y, p = be.label_numeric.to_numpy(), be.prob_fake.to_numpy()
    prim = metrics(y, p)
    cis = {k: [] for k in BOOT_KEYS}
    for idx in boot_index(be.group_id.to_numpy() if "group_id" in be else ref.group_id.to_numpy()):
        f = fast(y[idx], p[idx])
        for k in BOOT_KEYS:
            cis[k].append(f[k])
    prim["ci95_group_bootstrap"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in cis.items()}
    prim["n_groups"] = int(ref.group_id.nunique())
    # calibration (descriptive only; T was fitted on val and is NOT refitted)
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

    def ece(prob):
        bins = np.linspace(0, 1, 16); e = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
            if s.any():
                e += s.mean() * abs(prob[s].mean() - y[s].mean())
        return float(e)
    prim["ece_raw_15bin"] = ece(p); prim["ece_temperature_val_T1.5523_15bin"] = ece(1 / (1 + np.exp(-z / T)))
    out["primary_baseline_clean_evaluate_py"] = prim

    # --- secondary comparison table ---
    rows = []
    for (m, a), d in arms.items():
        rows.append({"model": m, "arm": a, **{k: v for k, v in metrics(d.label_numeric.to_numpy(), d.prob_fake.to_numpy()).items()
                                              if not isinstance(v, list)}})
    C = pd.DataFrame(rows)
    # group-bootstrap CIs per model x arm + paired J1 - baseline
    g = ref.group_id.to_numpy(); yy = ref.label_numeric.to_numpy()
    acc = {(m, a): {k: [] for k in BOOT_KEYS} for (m, a) in arms}
    diff = {a: {k: [] for k in BOOT_KEYS} for a in ["clean", "q50", "q35"]}
    for idx in boot_index(g):
        f = {key: fast(yy[idx], d.prob_fake.to_numpy()[idx]) for key, d in arms.items()}
        for key in arms:
            for k in BOOT_KEYS:
                acc[key][k].append(f[key][k])
        for a in diff:
            for k in BOOT_KEYS:
                diff[a][k].append(f[("J1", a)][k] - f[("baseline", a)][k])
    for k in ["eer", "genuine_fpr@0.5", "fake_miss@0.5", "app_genuine_called_manipulated", "app_fake_called_authentic"]:
        C[f"{k}__ci_lo"] = [np.percentile(acc[(r.model, r.arm)][k], 2.5) for r in C.itertuples()]
        C[f"{k}__ci_hi"] = [np.percentile(acc[(r.model, r.arm)][k], 97.5) for r in C.itertuples()]
    C.to_csv(os.path.join(R, "test_baseline_vs_J1_by_arm.csv"), index=False)
    D = []
    for a in diff:
        pt = {k: C.set_index(["model", "arm"]).loc[("J1", a), k] - C.set_index(["model", "arm"]).loc[("baseline", a), k] for k in BOOT_KEYS}
        for k in BOOT_KEYS:
            D.append({"arm": a, "metric": k, "J1_minus_baseline": pt[k], "ci_lo": np.percentile(diff[a][k], 2.5),
                      "ci_hi": np.percentile(diff[a][k], 97.5),
                      "ci_excludes_0": bool(np.percentile(diff[a][k], 2.5) > 0 or np.percentile(diff[a][k], 97.5) < 0)})
    D = pd.DataFrame(D); D.to_csv(os.path.join(R, "test_J1_minus_baseline_paired.csv"), index=False)

    # --- subgroups (per generator; per source) ---
    S = []
    for (m, a), d in arms.items():
        for col in ["generator", "source_dataset"]:
            for gname, s in d.groupby(col):
                lab = int(s.label_numeric.iloc[0]) if s.label_numeric.nunique() == 1 else -1
                pp = s.prob_fake.to_numpy()
                S.append({"model": m, "arm": a, "by": col, "group": gname, "label": {0: "REAL", 1: "FAKE", -1: "mixed"}[lab],
                          "n": len(s), "n_split_groups": s.group_id.nunique(), "flagged_fake@0.5": (pp >= .5).mean(),
                          "app_manipulated": (pp >= FAKE_AT).mean(), "app_inconclusive": ((pp >= REAL_BELOW) & (pp < FAKE_AT)).mean(),
                          "app_authentic": (pp < REAL_BELOW).mean()})
        ff = d[d.generator.str.startswith("ffpp_")]
        S.append({"model": m, "arm": a, "by": "within_source", "group": "ffpp_fakes_vs_ffpp_original_AUC", "label": "mixed",
                  "n": len(ff), "n_split_groups": ff.group_id.nunique(), "flagged_fake@0.5": roc_auc_score(ff.label_numeric, ff.prob_fake)})
    S = pd.DataFrame(S); S.to_csv(os.path.join(R, "test_subgroups_baseline_vs_J1.csv"), index=False)

    # descriptive analogue of the J1 criteria on test (vs baseline; E0 not evaluated on test)
    piv = S[S.by == "generator"].set_index(["model", "arm", "group"])["flagged_fake@0.5"]
    out["descriptive_jpeg_shift_test"] = {m: {"stylegan_tpr_drop_q35": float(piv[(m, "clean", "stylegan")] - piv[(m, "q35", "stylegan")]),
                                              "ffhq_fpr_rise_q35": float(piv[(m, "q35", "ffhq")] - piv[(m, "clean", "ffhq")])}
                                          for m in ["baseline", "J1"]}
    json.dump(out, open(os.path.join(R, "test_final_summary.json"), "w"), indent=2, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(json.dumps(out, indent=1, default=float))
    print(C[[c for c in C.columns if "__ci" not in c]].set_index(["model", "arm"]).T.round(4).to_string())
    print(D.round(4).to_string(index=False))
    sub = S[(S.by == "generator") | (S.by == "within_source")].pivot_table(index="group", columns=["model", "arm"], values="flagged_fake@0.5")
    print(sub.round(3).to_string())


if __name__ == "__main__":
    main()
