"""Stage-1 plots: LOFO comparison across model types; JPEG robustness. Validation only."""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
S1 = r"D:\PROJECT\VeriFace AI\research_runs\stage1_20261016"; P = os.path.join(S1, "plots")
V = pd.read_csv(os.path.join(S1, "reports", "lofo_verdicts_all_models.csv"))
fams = ["diffusion_t2i", "gan", "faceswap_video", "reenactment"]
fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, feat in zip(axs, ["efficientnet_b0_2ep", "clip_probe", "fft_probe"]):
    s = V[V.features == feat].set_index("eval_family").reindex(fams)
    x = np.arange(len(fams))
    ax.bar(x - .2, s.ALL_auc_vs_matched_real, .4, label="family seen (ALL)", color="#2b7bba")
    ax.bar(x + .2, s.auc_vs_matched_real, .4, label="family held out", color="#d95f02",
           yerr=np.clip([s.auc_vs_matched_real - s.auc_matched_ci_lo, s.auc_matched_ci_hi - s.auc_vs_matched_real], 0, None), capsize=2)
    ax.axhline(.5, ls=":", c="grey"); ax.axhline(.9, ls="--", c="green", lw=.7)
    ax.set_xticks(x, fams, rotation=20, fontsize=8); ax.set_title(feat, fontsize=9); ax.set_ylim(.4, 1.02)
axs[0].set_ylabel("val AUC: family fakes vs matched REAL"); axs[0].legend(fontsize=7)
fig.suptitle("Leave-family-out on VALIDATION (dashed: pre-registered 0.90 bar)", fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(P, "lofo_comparison_val.png")); plt.close(fig)
J = pd.read_csv(os.path.join(S1, "reports", "jpeg_robustness_val.csv"))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, k in zip(axs, ["val_eer", "fpr@0.5|ffhq", "real_fpr@0.5"]):
    for m, c in [("baseline", "#777"), ("E0", "#2b7bba"), ("J1", "#d95f02")]:
        s = J[J.model == m].set_index("arm").loc[["clean", "q50", "q35"]]
        ax.plot(range(3), s[k], "o-", c=c, label=m)
    ax.set_xticks(range(3), ["clean", "JPEG q50", "JPEG q35"]); ax.set_title(k, fontsize=9)
axs[0].legend(fontsize=8); fig.suptitle("JPEG robustness on VALIDATION: baseline vs E0 control vs J1 (JPEG-aug)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(P, "jpeg_robustness_val.png")); plt.close(fig)
print("ok")
