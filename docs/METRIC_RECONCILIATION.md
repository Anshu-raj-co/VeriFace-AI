# Diffusion / text-to-image hold-out AUC: reconciliation

Date: 2026-10-16. Status: **resolved**. No historical report was changed; this note explains how the different values
relate to each other.

## The discrepancy

Project documents give different numbers for "how well does the model detect diffusion/text-to-image faces when that
family is held out of training?":

| value | where it appears |
|---|---|
| **0.812** | `research_runs/stage1_20261016/STAGE1_REPORT.md` §3, `RESEARCH.md` §6, the A-to-Z guide |
| 0.74 (TPR 23.5 %) | `research_runs/generalization_audit_20261016/research_summary.md`, `generalization_audit.md` |
| 0.50 | an earlier request for the A-to-Z guide; the guide itself already flags it as wrong (its §10 correction note) |

## What each number actually is

The values come from two **different experiments**, measured against two **different genuine-image reference sets**:

* **`LOFO_diffusion_t2i`** (Stage 1) is the current experiment.
  * ImageNet init, **2 full epochs**, with a fixed sampler; step counts were asserted (`steps_done == steps_planned`).
* **`C_LOGO_nonGAN_holdout`** (generalisation audit) is **superseded**.
  * Because of a step-budget bug in `build_order`, it trained for **1 epoch instead of 2**.
  * `RESEARCH.md` §5 already marks it "superseded".

In both runs, `stable_diffusion`, `text_to_image_unknown` and `unattributed_ai` are held out of training. All
measurements below are on the **validation** split (948 held-out fakes).

The table below was recomputed independently for this note from each run's raw `val_predictions.csv`:

| run | vs FFHQ genuine (n = 7,075): the "matched REAL" used by Stage 1 | vs all validation genuine (n = 8,465) | TPR @0.5 |
|---|---:|---:|---:|
| Stage 1 `LOFO_diffusion_t2i` (current) | **0.8123** | 0.7621 | 15.3 % |
| Audit `C_LOGO_nonGAN_holdout` (superseded) | 0.7781 | 0.7405 | 23.5 % |

* **0.812** is the Stage 1 run, measured against FFHQ genuine images. It matches `lofo_efficientnet_val.csv`,
  `lofo_verdicts_all_models.csv` and the Part 2 re-verification `verify_lofo.csv` (0.8123498978694219).
* **0.74** is the superseded 1-epoch run, measured against *all* validation genuine images. It matches
  `logo_unseen_generator_val.csv` (`HELD_OUT_ALL`, 0.7405).
* **0.50** belongs to a different family. It is the **GAN (StyleGAN)** hold-out (0.504), not diffusion.

The matched reference set is defined in `research_runs/stage1_20261016/scripts/probe_lofo.py` (`MATCHED_REAL`):
diffusion and GAN fakes are compared with FFHQ genuine faces, because those are the same kind of photographic portrait.

## Which number to cite

* **Headline:** 0.812 [95 % CI 0.796, 0.826], with 15.3 % of held-out fakes detected at 0.5. This is the corrected,
  pre-registered experiment.
* When comparing with an all-genuine reference, use **0.762** for the same run.
* Either way, the conclusion does not change: detection of unseen diffusion generators is weak. The pre-registered rule
  required AUC ≥ 0.90 within 0.05 of the control, so the result is only "partial", and very few held-out fakes are
  caught at the default threshold.

## Reproduce

```python
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

held = ["stable_diffusion", "text_to_image_unknown", "unattributed_ai"]
v = pd.read_csv("research_runs/stage1_20261016/experiments/LOFO_diffusion_t2i/val_predictions.csv")
f = v.loc[v.generator.isin(held), "prob_fake"]
r = v.loc[v.generator == "ffhq", "prob_fake"]
print(roc_auc_score(np.r_[np.zeros(len(r)), np.ones(len(f))], np.r_[r, f]))  # 0.8123
```

The per-image prediction files are not published (they list dataset files); the summary CSVs named above are.
