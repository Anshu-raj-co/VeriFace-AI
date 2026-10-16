"""Derive the app's decision bands from baseline VALIDATION predictions only (cross-fitted over split groups)."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
v = pd.read_csv(r"D:\PROJECT\VeriFace AI\research_runs\generalization_audit_20261016\data\preds_baseline_val.csv")
y, p, g, gen = v.label_numeric.to_numpy(), v.prob_fake.to_numpy(), v.group_id.to_numpy(), v.generator.to_numpy()
rows = []
for lo, hi in [(0.5, 0.5), (0.2, 0.8), (0.1, 0.9), (0.3, 0.9), (0.4, 0.9), (0.2, 0.95), (0.1, 0.95)]:
    fake = p >= hi; real = p < lo; unc = ~fake & ~real
    rows.append({"real_below": lo, "fake_at_or_above": hi, "uncertain_share": unc.mean(),
                 "genuine_called_FAKE": fake[y == 0].mean(), "fake_called_REAL": real[y == 1].mean(),
                 "precision_of_FAKE_verdict": (y[fake] == 1).mean(), "precision_of_REAL_verdict": (y[real] == 0).mean(),
                 "ffpp_original_uncertain": unc[gen == "ffpp_original"].mean(), "ffpp_original_called_FAKE": fake[gen == "ffpp_original"].mean(),
                 "ffhq_called_FAKE": fake[gen == "ffhq"].mean()})
t = pd.DataFrame(rows); pd.set_option("display.width", 250); print(t.round(4).to_string(index=False))
# cross-fitted check of the chosen band
lo, hi = 0.30, 0.90
cf = []
for tr, te in GroupKFold(5).split(p, y, g):
    yt, pt = y[te], p[te]
    cf.append({"genuine_called_FAKE": (pt[yt == 0] >= hi).mean(), "fake_called_REAL": (pt[yt == 1] < lo).mean(), "uncertain": ((pt >= lo) & (pt < hi)).mean()})
cf = pd.DataFrame(cf); print("5-fold group check of (0.30, 0.90):\n", cf.describe().loc[["mean", "min", "max"]].round(4))
t.to_csv(r"reports/app_policy_val_bands.csv", index=False)
json.dump({"real_below": lo, "fake_at_or_above": hi, "temperature": 1.5523, "source": "baseline val predictions, research_runs/generalization_audit_20261016/data/preds_baseline_val.csv",
           "val_band_stats": t[(t.real_below == lo) & (t.fake_at_or_above == hi)].to_dict(orient="records"),
           "crossfit_group_folds": cf.to_dict(orient="list")}, open(r"reports/app_policy.json", "w"), indent=2)
