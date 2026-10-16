"""Plot for the Phase C leave-generator-out result (validation)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
r = pd.read_csv("reports/logo_unseen_generator_val.csv"); r = r[r.group != "REAL_all"]
gs = list(r.group.unique()); fig, ax = plt.subplots(figsize=(9, 4.2))
for k, (arm, c) in enumerate([("C_ALL_control", "#2b7bba"), ("C_LOGO_nonGAN_holdout", "#d95f02")]):
    s = r[r.arm == arm].set_index("group").loc[gs]
    err = np.clip([s["tpr@0.5"] - s.tpr_wilson_lo, s.tpr_wilson_hi - s["tpr@0.5"]], 0, None)
    ax.bar(np.arange(len(gs)) + (k - .5) * .38, s["tpr@0.5"], .38, color=c, label=arm, yerr=err, capsize=2)
st = r[r.arm == "C_ALL_control"].set_index("group").loc[gs].status
ax.set_xticks(range(len(gs)), [f"{g}\n({s})" for g, s in zip(gs, st)], rotation=25, ha="right", fontsize=7)
ax.set_ylabel("detection rate (TPR) @0.5, Wilson 95% CI"); ax.legend(fontsize=8)
ax.set_title("Leave-generator-out (VALIDATION): non-GAN text-to-image generators held out of training", fontsize=9)
fig.tight_layout(); fig.savefig("plots/val_logo_unseen_generator.png")
