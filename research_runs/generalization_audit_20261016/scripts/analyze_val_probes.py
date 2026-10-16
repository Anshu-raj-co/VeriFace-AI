"""Validation-only analysis: baseline within-source metrics, counterfactual degradation
probe (paired), and train-set fit on video sources. Writes reports/*.csv|json."""
import json, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
fam = lambda g: "ffpp" if g.startswith("ffpp") else "celebdf" if g.startswith("celebdf") else "dfdc" if g.startswith("dfdc") else g
def within(d, label):
    rows = []
    for f in ["ffpp", "celebdf", "dfdc"]:
        s = d[d.generator.map(fam) == f]
        r = s[s.label_numeric == 0]
        for g, gs in [("ALL", s[s.label_numeric == 1])] + list(s[s.label_numeric == 1].groupby("generator")):
            both = pd.concat([r, gs])
            if both.label_numeric.nunique() < 2: continue
            auc = roc_auc_score(both.label_numeric, both.prob_fake)
            grp = both.group_id.unique(); boots = []
            for _ in range(500):
                pick = rng.choice(grp, len(grp)); bb = both.set_index("group_id").loc[pick]
                if bb.label_numeric.nunique() == 2: boots.append(roc_auc_score(bb.label_numeric, bb.prob_fake))
            rows.append({"set": label, "family": f, "fake_generator": g, "n_real": len(r), "n_fake": len(gs),
                         "n_groups": len(grp), "auc_within_source": auc,
                         "auc_ci_lo": np.percentile(boots, 2.5) if boots else np.nan, "auc_ci_hi": np.percentile(boots, 97.5) if boots else np.nan,
                         "real_fpr@0.5": float((r.prob_fake >= .5).mean()), "fake_tpr@0.5": float((gs.prob_fake >= .5).mean())})
    return pd.DataFrame(rows)
v = pd.read_csv("data/preds_baseline_val.csv")
t = pd.read_csv("data/preds_baseline_train_videosources.csv")
W = pd.concat([within(v, "val"), within(t, "train")])
W.to_csv("reports/baseline_within_source_val_train.csv", index=False)
pd.set_option("display.width", 250); print(W.round(4).to_string(index=False))
# val per-generator
G = v.groupby(["generator", "label_numeric"]).agg(n=("prob_fake", "size"), mean_p=("prob_fake", "mean"),
        correct=("prob_fake", lambda s: 0)).reset_index()
G["correct"] = [float(((sub.prob_fake >= .5) == bool(l)).mean()) for (g, l), sub in v.groupby(["generator", "label_numeric"])]
G.to_csv("reports/baseline_val_by_generator.csv", index=False); print(G.round(4).to_string(index=False))
# counterfactual paired
c = pd.read_csv("data/preds_baseline_val_counterfactual.csv")
piv = c.pivot_table(index=["image_id", "generator", "label_numeric"], columns="arm", values="logit").reset_index()
rows = []
for g, s in piv.groupby("generator"):
    for arm in ["reencode", "down120", "down160", "jpeg35"]:
        dl = (s[arm] - s["reencode"]) if arm != "reencode" else (s["reencode"] - s["none"])
        ref = "reencode" if arm != "reencode" else "none"
        p_arm = 1 / (1 + np.exp(-s[arm])); p_ref = 1 / (1 + np.exp(-s[ref]))
        bo = [rng.choice(dl.to_numpy(), len(dl)).mean() for _ in range(1000)]
        rows.append({"generator": g, "label": "FAKE" if s.label_numeric.iloc[0] else "REAL", "n": len(s), "arm": arm, "vs": ref,
                     "mean_logit_shift": dl.mean(), "ci_lo": np.percentile(bo, 2.5), "ci_hi": np.percentile(bo, 97.5),
                     "mean_p_ref": p_ref.mean(), "mean_p_arm": p_arm.mean(),
                     "flagged_fake_ref": float((p_ref >= .5).mean()), "flagged_fake_arm": float((p_arm >= .5).mean())})
C = pd.DataFrame(rows); C.to_csv("reports/counterfactual_degradation_val.csv", index=False); print(C.round(4).to_string(index=False))
