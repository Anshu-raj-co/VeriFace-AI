"""VeriFace AI research -- plots and error-example contact sheets (descriptive).
Reads reports/ + data/ produced by the other scripts; writes plots/*.png only."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\generalization_audit_20261016")
P = os.path.join(OUT, "plots")
os.makedirs(P, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def subgroup_bars():
    t = pd.read_csv(os.path.join(OUT, "reports", "test_subgroups_by_generator.csv")).sort_values(["label", "correct_rate@0.5"])
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(t))
    col = np.where(t.label == "REAL", "#2b7bba", "#d95f02")
    ax.barh(y, t["correct_rate@0.5"], color=col, alpha=.85)
    ax.errorbar(t["correct_rate@0.5"], y, xerr=[t["correct_rate@0.5"] - t.cluster_boot_lo, t.cluster_boot_hi - t["correct_rate@0.5"]],
                fmt="none", ecolor="k", lw=1, capsize=2)
    ax.set_yticks(y, [f"{g} ({l}, n={n}, groups={k})" for g, l, n, k in zip(t.generator, t.label, t.n, t.n_split_groups)])
    ax.set_xlabel("share correct @0.5 (REAL: passed as REAL; FAKE: flagged FAKE), 95% cluster-bootstrap CI")
    ax.set_title("Baseline (epoch 7) on the locked TEST split — descriptive only")
    ax.set_xlim(0, 1.02); ax.axvline(0.5, color="grey", lw=.5, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(P, "test_baseline_subgroup_correct_rate.png")); plt.close(fig)


def prob_dists():
    pr = pd.read_csv(os.path.join(BASE, r"evaluation_reports\predictions_test_20261016_091737.csv"))
    v = pd.read_csv(os.path.join(OUT, "data", "preds_baseline_val.csv"))
    for name, d in [("test", pr), ("val", v)]:
        order = d.groupby("generator")["prob_fake"].median().sort_values().index
        fig, ax = plt.subplots(figsize=(9, 5.5))
        data = [d.loc[d.generator == g, "prob_fake"].to_numpy() for g in order]
        bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True)
        for patch, g in zip(bp["boxes"], order):
            lab = d.loc[d.generator == g, "label_numeric"].iloc[0]
            patch.set_facecolor("#2b7bba" if lab == 0 else "#d95f02"); patch.set_alpha(.6)
        ax.set_yticks(range(1, len(order) + 1), order)
        ax.axvline(0.5, color="k", lw=.7, ls="--")
        ax.set_xlabel("predicted P(FAKE)  (box: IQR, whiskers: 5–95th pct; blue = REAL, orange = FAKE)")
        ax.set_title(f"Baseline P(FAKE) by generator — {name} split")
        fig.tight_layout(); fig.savefig(os.path.join(P, f"{name}_baseline_prob_by_generator.png")); plt.close(fig)


def counterfactual():
    c = pd.read_csv(os.path.join(OUT, "reports", "counterfactual_degradation_val.csv"))
    c = c[c.arm != "reencode"]
    gens = c.generator.unique(); arms = ["down160", "down120", "jpeg35"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = .25
    for k, arm in enumerate(arms):
        s = c[c.arm == arm].set_index("generator").loc[gens]
        ax.bar(np.arange(len(gens)) + (k - 1) * w, s.mean_logit_shift, w, yerr=[s.mean_logit_shift - s.ci_lo, s.ci_hi - s.mean_logit_shift],
               label=arm, capsize=2)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(len(gens)), gens, rotation=30, ha="right")
    ax.set_ylabel("mean logit shift vs re-encoded control (+ = more FAKE)")
    ax.set_title("Counterfactual degradation probe on VALIDATION images (baseline model)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(P, "val_counterfactual_logit_shift.png")); plt.close(fig)


def within_source():
    w = pd.read_csv(os.path.join(OUT, "reports", "baseline_within_source_val_train.csv"))
    w = w[w.fake_generator != "ALL"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    gens = w.fake_generator.unique()
    for k, st in enumerate(["train", "val"]):
        s = w[w.set == st].set_index("fake_generator").reindex(gens)
        ax.bar(np.arange(len(gens)) + (k - .5) * .38, s.auc_within_source, .38,
               yerr=[s.auc_within_source - s.auc_ci_lo, s.auc_ci_hi - s.auc_within_source], capsize=2, label=st)
    ax.set_xticks(range(len(gens)), gens, rotation=30, ha="right"); ax.set_ylim(.4, 1.01)
    ax.axhline(.5, color="grey", ls=":", lw=.7)
    ax.set_ylabel("AUC: fakes vs REAL of the SAME source"); ax.legend()
    ax.set_title("Within-source separability (baseline) — train vs validation")
    fig.tight_layout(); fig.savefig(os.path.join(P, "within_source_auc_train_val.png")); plt.close(fig)


def contact_sheets():
    pr = pd.read_csv(os.path.join(BASE, r"evaluation_reports\predictions_test_20261016_091737.csv"))
    sel = {
        "test_fp_ffpp_original": pr[(pr.generator == "ffpp_original")].sort_values("prob_fake", ascending=False),
        "test_tn_ffpp_original": pr[(pr.generator == "ffpp_original")].sort_values("prob_fake"),
        "test_fn_ffpp_NeuralTextures": pr[(pr.generator == "ffpp_NeuralTextures")].sort_values("prob_fake"),
        "test_fp_dfdc_real": pr[(pr.generator == "dfdc_real")].sort_values("prob_fake", ascending=False),
        "test_fn_dfdc_fake": pr[(pr.generator == "dfdc_fake")].sort_values("prob_fake"),
    }
    for name, d in sel.items():
        d = d.drop_duplicates("local_image_path").head(12)
        fig, axs = plt.subplots(2, 6, figsize=(12, 4.6))
        for ax, (_, r) in zip(axs.ravel(), d.iterrows()):
            with Image.open(os.path.join(BASE, r.local_image_path)) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{r.image_id}\nP(FAKE)={r.prob_fake:.2f}", fontsize=6.5); ax.axis("off")
        for ax in axs.ravel()[len(d):]:
            ax.axis("off")
        fig.suptitle(f"{name} (locked test, descriptive; existing processed 256px crops)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(P, f"examples_{name}.png")); plt.close(fig)


def comparison():
    f = os.path.join(OUT, "experiment_results.csv")
    if not os.path.exists(f):
        return
    r = pd.read_csv(f).set_index("model")
    keys = ["fpr@0.5|ffpp_original", "fpr@0.5|dfdc_real", "within_auc|ffpp", "within_auc|ffpp_NeuralTextures",
            "within_auc|dfdc", "tpr@0.5|ffpp_NeuralTextures", "val_eer", "real_fpr@0.5"]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.5))
    for ax, k in zip(axs.ravel(), keys):
        lo = r.get(f"{k}__ci_lo"); hi = r.get(f"{k}__ci_hi")
        ax.bar(range(len(r)), r[k], color=["#777", "#2b7bba", "#d95f02", "#1b9e77"][:len(r)])
        if lo is not None:
            ax.errorbar(range(len(r)), r[k], yerr=[r[k] - lo, hi - r[k]], fmt="none", ecolor="k", capsize=2)
        ax.set_xticks(range(len(r)), [m.replace("_", "\n", 1) for m in r.index], fontsize=6.5); ax.set_title(k, fontsize=8)
    fig.suptitle("Validation comparison (group-bootstrap 95% CI)"); fig.tight_layout()
    fig.savefig(os.path.join(P, "val_experiment_comparison.png")); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["subgroup_bars", "prob_dists", "counterfactual", "within_source", "contact_sheets", "comparison"]
    for w in which:
        globals()[w]()
        print("plot", w, "ok", flush=True)
