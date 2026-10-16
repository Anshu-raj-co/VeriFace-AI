# VeriFace AI — Stage 1 experiment plan (pre-registered 2026-10-16 ~12:30, before any Stage-1 result)

Selection and analysis use **train for fitting, validation for evaluation/selection**. The test split is not read
by any Stage-1 script (scripts assert this). Baseline = `best_face_model.pth` (epoch 7), immutable.

## Generator families used for leave-family-out (LOFO) tests

| family | generators held out of TRAIN | val images held out |
|---|---|---:|
| diffusion_t2i | stable_diffusion, text_to_image_unknown, unattributed_ai | 948 |
| gan | stylegan | 5,790 |
| faceswap_video | celebdf_synthesis, dfdc_fake, ffpp_Deepfakes, ffpp_FaceSwap, ffpp_FaceShifter, ffpp_DeepFakeDetection | 993 |
| reenactment | ffpp_Face2Face, ffpp_NeuralTextures | 245 |

Metrics on val for the held-out family: TPR@0.5; AUC vs all val REAL; AUC vs the *matched* REAL set (FFHQ for
diffusion/GAN; Celeb-DF + DFDC + FF++ originals for faceswap_video; FF++ originals for reenactment); Wilson / bootstrap
CIs. Each LOFO arm is compared with its same-recipe ALL control.

**Interpretation rule (fixed now):** cross-family generalisation is "supported" for a family only if the LOFO arm's AUC
vs matched REAL ≥ 0.90 **and** is within 0.05 of the ALL control; "partial" if AUC ≥ 0.75; otherwise "not supported".

## S1-A. CLIP ViT-B/16 linear probe (frozen features)

* Features: OpenAI CLIP ViT-B/16 image tower (timm `vit_base_patch16_clip_224.openai`, 571 MiB, downloaded once),
  CenterCrop 224 of the 256 face crop, CLIP normalisation; train + val only.
* Probe: L2-normalised features → logistic regression, C = 1.0, lbfgs, max_iter 3000, unweighted — fixed in advance,
  no tuning on validation.
* Arms: CLIP-ALL + four CLIP-LOFO arms. Compared with the EfficientNet baseline on the full val panel and with
  EfficientNet LOFO (S1-C).
* Hypothesis: frozen CLIP features generalise across generator families better than a fine-tuned CNN, at the cost of
  lower in-distribution accuracy (literature: UnivFD).

## S1-B. JPEG-compression robustness (fine-tune, matched control)

* Measured gap (previous study, val): heavy JPEG q35 lowers StyleGAN TPR 99.8 → 97.5 % and raises FFHQ FPR 0.8 → 3.7 %.
* **J1:** fine-tune from the baseline, 1 epoch (8,230 steps), recipe identical to the previous study's **E0 control**
  (same seed / order / LR 1e-4 cosine, AMP, bs 16) plus `--aug-jpeg 30,60,0.5` (with p = 0.5 re-encode at
  q ~ U[30, 60] before the standard augmentation). **Matched control = E0** (already trained, identical recipe without
  the augmentation; its sampler requested exactly one epoch, so the old sampler bug did not affect it).
* Evaluation (val): clean, and re-encoded at q50 and q35 (the same counterfactual arms as before), for baseline, E0, J1.
* Success: J1 vs E0 on q35 — StyleGAN TPR drop and FFHQ FPR rise each at most half of E0's; clean val EER ≤ E0 + 0.3 pp;
  within-FF++ AUC ≥ E0 − 0.02.

## S1-C. Corrected EfficientNet leave-family-out (fixes the sampler bug)

* `scripts/train_exp.py` (fixed `build_order`, unit-tested in `logs/build_order_unit_test.log`; the run asserts that
  steps_done == steps_planned).
* Recipe: ImageNet init, **2 full epochs**, LR 3e-4, 10 % warmup + cosine, stem frozen, backbone LR ×0.3, AMP, bs 16,
  0 workers, standard augmentation, seed 42. Final epoch = selected (no early stopping on val).
* Arms (sequential): C-ALL, then LOFO diffusion_t2i, faceswap_video, reenactment, gan (in that priority order).
  ≈ 45–50 min each. If time runs short, lower-priority arms are recorded as skipped, not partially reported.
* These are 2-epoch models (the baseline had 10): results are **relative** comparisons against the same-budget C-ALL,
  not deployable candidates.

## S1-D. FFT / frequency probe (CPU)

* Hypothesis: if generator artefacts are spectral, a simple azimuthally-averaged log power spectrum probe should detect
  seen generators, and its LOFO behaviour tells whether spectral cues transfer across families.
* Features: grayscale of the processed 256 px crop (CenterCrop 224), 2-D FFT, log power, radial average (112 bins) +
  high-frequency energy ratios; logistic regression (standardised, C = 1.0). Same ALL/LOFO arms as S1-A.
* A dual-domain (RGB + FFT) CNN is trained **only if** the probe transfers better than CLIP and EfficientNet on some
  family — otherwise skipped with this evidence.

## S1-E. Higher-resolution FF++ crops

Feasibility check before any work: are FF++ source frames available above the 256 px raw size? (If not → skipped.)

## Selection for Part 2 (final, locked-test evaluation)

A candidate is recommended for final evaluation only if, on validation, it beats the baseline on its target weakness by
the pre-declared margin without guardrail violations (val EER ≤ baseline + 0.5 pp; FFHQ FPR + ≤ 0.5 pp; StyleGAN /
Stable Diffusion / Celeb-DF-synthesis TPR − ≤ 1.5 pp). LOFO models are diagnostic only.
