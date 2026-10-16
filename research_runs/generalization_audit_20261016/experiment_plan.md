# VeriFace AI — Experiment plan (pre-registered before any training run)

Written 2026-10-16 after Phase A/B and the validation-only probes, **before** any
experiment was launched. Selection uses validation data only; the test split stays locked.

## Evidence that motivates the experiments (train/val only)

| Observation | Source |
|---|---|
| Within-FF++ separability is weak: val AUC (FF++ fakes vs FF++ originals) 0.72 [0.68, 0.76]; NeuralTextures 0.63 | `reports/baseline_within_source_val_train.csv` |
| The model does not fit FF++ even on **train**: within-FF++ train AUC 0.90; 45.7 % of train FF++ originals are flagged FAKE | same |
| Train label prior inside FF++ is 72 % FAKE (DFDC 74 %, Celeb-DF 33 %) | `reports/phase_a_audit.json` |
| FF++ original mean P(FAKE) ≈ 0.62–0.63 on train/val/test ≈ the 0.72 prior → consistent with prior fallback where the model cannot discriminate | probes |
| Counterfactual: FF++-style down/up-scaling does **not** turn genuine faces FAKE (Celeb-DF/DFDC/web shift toward REAL; FFHQ flagged 0.8 → 2.0 %) | `reports/counterfactual_degradation_val.csv` |
| Heavy JPEG (q35) raises genuine-face FAKE scores modestly (FFHQ 0.8 → 3.7 % flagged) and lowers StyleGAN TPR 99.8 → 97.5 % | same |
| FF++ is 5.5 % of training images; FFHQ + StyleGAN are 77 % | Phase A counts |

## Hypotheses

* **H1 — source-prior fallback.** Within FF++ (and DFDC) the label prior is FAKE-heavy, so when the model cannot
  discriminate it learns "looks like FF++ → FAKE". Removing the within-source prior (equal REAL/FAKE weight inside each
  source family) should cut FF++/DFDC genuine-face false positives. Prediction: FF++ REAL FPR@0.5 drops sharply;
  within-FF++ AUC (a ranking metric) changes little.
* **H2 — under-emphasis of hard video sources.** Easy FFHQ/StyleGAN dominate the loss; up-weighting the video sources
  (FF++, Celeb-DF, DFDC) may improve *ranking* inside FF++ (within-source AUC). Only tested if H1 leaves within-FF++ AUC
  unchanged and budget allows.

## Design (one controlled change per arm)

All arms: fine-tune **from the frozen baseline `best_face_model.pth` (epoch 7)**, 1 epoch over the full train split
(8,230 steps, batch 16), AdamW, head LR 1e-4 → 0 cosine, backbone LR ×0.3, stem frozen, AMP, 0 workers,
`train.get_train_transforms()` augmentation, identical seeded data order (seed 42). Validation: full val split,
`train.get_eval_transforms()`. Outputs under `experiments/<arm>/`.

| Arm | Change vs E0 | Tests |
|---|---|---|
| **E0 control** | none (plain fine-tune) | separates the effect of "one more epoch at low LR" from the intervention |
| **E1 source-class-balanced loss** | per-sample BCE weight so REAL and FAKE carry equal total weight **inside each multi-class source family** (FF++, DFDC, Celeb-DF); family totals unchanged; single-class sources weight 1; weights normalised to mean 1 | H1 |
| E2 source-balanced sampling (conditional) | sampler oversamples video-source families ×4, uniform loss | H2 |

Compute: ≈ 20 min GPU per arm (≈ 1.0 GB VRAM, ≈ 5 GB host commit, measured on the baseline run); ≤ 3 arms ≈ 1 h.

## Pre-declared comparison criteria (validation split only)

Primary weak-case metrics: FF++ original REAL FPR@0.5; within-FF++ AUC (FF++ fakes vs FF++ originals, group bootstrap
CI); DFDC real FPR@0.5 (reported, underpowered: 9 split groups).

A candidate is **better than baseline and E0** only if **all** hold:
1. FF++ original FPR@0.5 reduced by ≥ 15 percentage points vs E0;
2. within-FF++ AUC ≥ E0 − 0.02;
3. guardrails: overall val EER ≤ baseline EER + 0.5 pp; FFHQ FPR increase ≤ 0.5 pp; StyleGAN, Stable Diffusion and
   Celeb-DF-synthesis TPR@0.5 each drop ≤ 1.5 pp; overall balanced accuracy@0.5 not lower than baseline − 0.5 pp.

If the FPR drop is only matched by an equal TPR drop on FF++ fakes with no AUC change, the result is reported as a
**calibration/prior shift, not improved discrimination**.

Thresholds and calibration (Phase F) are fitted on validation only. The locked test set is evaluated once, at the end,
for the frozen selected model and the baseline, under identical preprocessing — and only if no test information was
used for any decision (it was not: test predictions were used only for descriptive Phase B analysis).

---

## Addendum 1 (written 2026-10-16 ~10:35, after E0/E1, before E2/LOGO were launched)

**E1 outcome (validation):** FF++ original FPR −14.3 pp vs E0 (paired CI −18.6 to −10.0), within-FF++ AUC unchanged
(paired diff CI −0.004 to +0.003), NeuralTextures TPR −13 pp, FFHQ FPR +0.6 pp vs baseline. Criterion 1 (≥ 15 pp) and the
FFHQ guardrail fail → **not selected; classified as a prior/calibration shift, not improved discrimination.** Note: E0
(no intervention) already moved FF++ original FPR 72 → 57 % and FFHQ FPR 0.5 → 1.2 % — subgroup operating points at
0.5 are unstable under small training perturbations.

The plan's condition for **E2** is met (within-FF++ AUC unchanged by H1). E2 runs exactly as specified
(`--sampler video_x4`, uniform loss, everything else as E0). Same acceptance criteria; additionally E2 must raise
within-FF++ AUC by ≥ 0.03 over E0 with a paired CI excluding 0 to count as support for H2.

### Phase C — leave-generator-out (unseen-generator) test

* **Question:** does the detector flag face images from synthesis generators it has never seen?
* **Held-out generators:** all non-GAN text-to-image sources — `stable_diffusion`, `text_to_image_unknown`,
  `unattributed_ai` — removed from TRAIN only. Training keeps StyleGAN + video face-swap/reenactment fakes.
* **C-LOGO:** ImageNet-initialised EfficientNet-B0, 15,509 steps (2 epochs of the reduced train set), LR 3e-4, 10 % warmup
  then cosine, stem frozen, backbone ×0.3, AMP, batch 16, same augmentation, seed 42.
* **C-ALL control:** identical, trained on the full train split with the same number of steps (`--epochs 1.8844`).
* **Metric (validation only):** detection rate @0.5 and AUC of the held-out generators vs all val REAL and vs val FFHQ;
  same metrics for the seen generators as a sanity check. Group-bootstrap CIs.
* **Interpretation rule:** unseen-generator generalisation is supported for this family only if C-LOGO's held-out AUC ≥
  0.95 *and* its TPR@0.5 is within 10 pp of C-ALL. Otherwise report a generalisation gap of the measured size.
* No checkpoint from these runs is a deployment candidate (budget-limited, not comparable to the 10-epoch baseline).

Compute: E2 ≈ 20–25 min; each LOGO arm ≈ 45 min; total ≈ 2 h GPU, sequential, ≈ 1.2 GB VRAM.

### Deviation (recorded 2026-10-16 11:20, before the C-ALL result was seen)

`build_order(..., sampler="uniform")` returns `permutation(len(df))[:steps*bs]`, which is capped at **one epoch**. Therefore
C-LOGO ran 7,754 steps (1 epoch of the reduced train set) and C-ALL runs 8,230 steps (1 epoch of the full train set),
instead of 15,509 each; the cosine schedule was sized for 15,509 steps, so both stopped at ≈ 50 % of the schedule
(LR ≈ 1.5e-4, not annealed). Both arms are truncated identically (6 % step difference), so the LOGO-vs-ALL comparison
remains controlled, but both models are under-trained relative to the plan and to the 10-epoch baseline.
E0/E1 (exactly one epoch requested) and E2 (sampling with replacement) are unaffected. Not re-run (≈ 90 min GPU);
the fix (tile permutations across epochs) is noted for future runs.
