# VeriFace AI — Stage 1 report

Directory: `research_runs/stage1_20261016/` · Date: 2026-10-16 · Baseline: `best_face_model.pth` (EfficientNet-B0,
epoch 7, SHA-256 `c9a5eb5e…0eb11`, unchanged). Pre-registration: `experiment_plan.md` (written before any Stage-1 result).
All model comparisons use the **validation** split; **no Stage-1 script read the test split** (scripts assert it).

## 0. Status overview

| item | status | key evidence |
|---|---|---|
| State inspection, hashes, resources | **completed** | `data/protected_hashes_stage1_start.sha256`, `data/prior_research_hashes.sha256` |
| Dataset / label-integrity audit | **completed** (no confirmed errors; suspicious DFDC subset) | `label_integrity_report.md` |
| App fix + tests | **completed** | `inference.py` (new), `app.py` (modified), `app_tests/` |
| Corrected sampler + LOFO (EfficientNet, 2 full epochs) | **completed** — 5/5 runs, step counts verified | `reports/lofo_efficientnet_val.csv`, `reports/lofo_verdicts_all_models.csv` |
| JPEG robustness (J1 vs matched E0) | **completed** — improvement, pre-registered margin **not met** | `reports/jpeg_robustness_val.csv`, `reports/jpeg_decision.json` |
| CLIP ViT-B/16 linear probe (ALL + LOFO) | **completed** — negative | `reports/lofo_clip_probe_val.csv` |
| FFT spectral probe (ALL + LOFO) | **completed** — negative | `reports/lofo_fft_probe_val.csv` |
| Dual-domain (RGB+FFT) CNN | **skipped** by pre-registered rule (probe did not transfer) | §4 |
| CLIP ViT-L/14 | **skipped** (1.6 GiB, ≈ 1 h GPU; B/16 clearly negative) — proposed for Part 2 only if wanted | §4 |
| Higher-resolution FF++ crops | **skipped — infeasible**: every FF++ source frame is 256×256, faces ≈ 120 px, no videos locally | `logs/ffpp_resolution_check.log` |
| Calibration / thresholds / abstention (val) | **completed** | `reports/calibration_val_stage1.csv`, `reports/operating_points_val_stage1.csv`, `reports/abstention_band_val_stage1.csv`, `reports/app_policy_val_bands.csv` |
| Grad-CAM + shortcut checks (val) | **completed** | `reports/gradcam_center_share_summary_val.csv`, `plots/gradcam_val_examples.png` |
| Final locked-test evaluation | **not run (by instruction)** — Part 2 | §7 |

## 1. Dataset and label integrity (details: `label_integrity_report.md`)

* Label vs. provenance (folder / filename tokens): **0 conflicts** across all 164,760 images; 0 MD5 duplicates.
* 7,903 REAL↔FAKE near-duplicate pairs pixel-verified on raw files: **0 pixel copies** — every FF++ "fake" differs from
  its original by a real, localized edit (median only ~5–7 % of the frame; NeuralTextures as low as 0.5 %).
* 106 confident model–label disagreements reviewed (42 val, 15 train, 49 test-descriptive). **No confirmed label error.**
  **Suspicious and unverifiable:** a few DFDC videos — 5/276 "fake" videos with every frame scored real (consistent with
  DFDC audio-only or unswapped-second-person fakes), 3/64 "real" videos scored fake (one heavily blurred).
* Structural risks, not errors: 49 Celeb-DF identities across splits; DFDC actor overlap possible; DFDC has only
  4–9 independent groups per evaluation split.
* We **cannot** certify all labels correct; ~70 images were visually reviewed. Recommended: obtain DFDC per-video
  metadata to verify the DFDC candidates.

## 2. Application fixes and tests

**Problems found in `app.py`** (original saved as `data/app.py.original_backup`):
1. Whole image `Resize((224,224))` — no face crop; did not match training/evaluation (`crop_face_square` 256 → `CenterCrop(224)`).
2. No `positive_class` check (a REAL=1 checkpoint would invert verdicts silently).
3. Binary verdict at 0.5, "confidence %" presented as certainty, a "Raw Logit Bias" metric that was actually `p − 0.5`.
4. Grad-CAM drawn on the full upload rather than on the model input.
5. Face gate (`detect_face`, min face 60 px on the full image) stricter than the training cropper (min 40 px on a
   ≤512 px copy) → rejected FF++-style frames the training pipeline accepted.

**Changes:** new `inference.py` (load with FAKE-positive check; exact eval preprocessing; validation-derived decision
bands; temperature-scaled score) used by `app.py`; three-way verdict LIKELY AUTHENTIC / INCONCLUSIVE / LIKELY MANIPULATED;
wording that the score is a calibrated model score, not a guaranteed probability; Grad-CAM on the 224 px model input;
face gate = the cropper's own detection. Bands (val, cross-fitted over split groups): authentic < 0.30 ≤ inconclusive <
0.90 ≤ manipulated → 1.0 % of genuine faces called manipulated, 0.7 % of fakes called authentic, 4.8 % inconclusive
(73 % of FF++ originals fall in the inconclusive band). Temperature T = 1.55 (val).

**Tests (all exit 0):**

| test | result |
|---|---|
| Parity on 255 raw val images (15 / generator) vs the evaluation pipeline | Spearman **0.993** (old app 0.938); median \|Δlogit\| **0.19** (old 0.93); decision agreement @0.5 **98.4 %** (old 95.3 %); crop size identical to the stored crop for 100 % |
| Edge cases (grayscale, RGBA PNG, 64 px, 3000 px, 1920×1080, blank) | all run, tensor 1×3×224×224 |
| REAL-positive checkpoint | refused with a clear error |
| Streamlit `AppTest` of the real `app.py` (FFHQ, StyleGAN, FF++ original, Stable Diffusion val images) | no exceptions; verdicts INCONCLUSIVE (0.57; eval pipeline 0.49) / MANIPULATED / INCONCLUSIVE / MANIPULATED; FF++ frame was blocked by the old gate, accepted after the gate fix |
| Headless `streamlit run app.py` | `/_stcore/health` = ok, page HTTP 200, server stopped cleanly |

Files: `app_tests/inference_tests.json`, `app_tests/parity_val_sample.csv`, `app_tests/streamlit_apptest.json`
(run 2, after gate fix), `app_tests/streamlit_apptest_run1_before_gate_fix.json`, `app_tests/streamlit_server_health.txt`.

## 3. Corrected generalisation experiments (leave-family-out, validation)

Sampler bug fixed in `scripts/train_exp.py` (one permutation per epoch; unit test `logs/build_order_unit_test.log`;
runs assert steps_done == steps_planned). All five runs: ImageNet init, **2 full epochs**, identical recipe.

| run | steps (planned = done) | minutes | val AUC (overall) |
|---|---:|---:|---:|
| LOFO_ALL_control | 16,460 | 37.5 | 0.996 |
| LOFO_diffusion_t2i | 15,508 | 30.5 | 0.967 |
| LOFO_faceswap_video | 15,236 | 31.3 | 0.943 |
| LOFO_reenactment | 16,212 | 33.4 | 0.996 |
| LOFO_gan | 10,668 | 23.6 | 0.584 |

Held-out family vs matched REAL (pre-registered rule: supported if AUC ≥ 0.90 and within 0.05 of ALL):

| held-out family | TPR@0.5 (seen → held out) | AUC vs matched REAL (seen → held out, 95 % CI) | verdict |
|---|---|---|---|
| diffusion / T2I | 99.6 % → **15.3 %** | 0.9997 → **0.812** [0.796, 0.826] | partial |
| GAN (StyleGAN) | 99.3 % → **0.0 %** | 0.9996 → **0.504** [0.493, 0.513] | not supported |
| face-swap (video) | 97.3 % → **19.9 %** | 0.930 → **0.591** [0.566, 0.614] | not supported |
| reenactment (F2F, NT) | 96.3 % → 67.8 %* | 0.606 → **0.587** [0.538, 0.631] | not supported |

\*Reenactment "TPR" reflects the FF++ source prior: FF++ originals are flagged equally often (within-FF++ AUC ≈ 0.6
even when reenactment is seen, at this 2-epoch budget; 0.68 for the 10-epoch baseline).

**Conclusion:** with the available data, the detector learns **family-specific** artefacts. Held-out GAN faces are
missed completely; held-out video face-swaps and diffusion faces are mostly missed. Unseen-generator generalisation is
**not supported**. These are 2-epoch models — the conclusion is a relative one (held-out vs the same-budget ALL control).

## 4. Robustness and alternative models (validation)

**J1 — JPEG augmentation (q30–60, p 0.5), 1-epoch fine-tune from the baseline vs matched control E0:**

| model | clean EER | q50 EER | q35 EER | q35 genuine-face FPR | q35 FFHQ FPR | q35 StyleGAN TPR |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.62 % | 4.59 % | 4.93 % | 7.5 % | 4.4 % | 97.9 % |
| E0 control | 2.82 % | 4.92 % | 5.50 % | 10.6 % | 8.0 % | 99.5 % |
| **J1** | **2.58 %** | **3.72 %** | **4.30 %** | **7.4 %** | 4.6 % | 99.5 % |

(Table: one fresh fp16-autocast inference pass per arm, `data/jpeg_eval/`; the fp32 baseline predictions used
elsewhere give clean EER 2.66 % — the 0.04 pp difference is numerical precision.)
J1 − E0 at q35 (paired group bootstrap): EER −0.94 to −1.44 pp; genuine-face FPR −2.8 to −3.6 pp. **Pre-registered
criteria: not met** — the FFHQ-FPR rise under q35 is 58 % of E0's (needed ≤ 50 %); the StyleGAN criterion is
uninformative because plain fine-tuning (E0) already removed the StyleGAN drop. J1 passes every clean-data guardrail vs
the baseline (EER 2.58 vs 2.66 %; FFHQ FPR 0.51 vs 0.52 %; StyleGAN / SD / Celeb-DF TPR not lower; at 1 % genuine-face
FPR it misses 5.9 % of fakes vs 6.5 %). FF++ originals remain the main weakness (57 % flagged at 0.5).

**CLIP ViT-B/16 frozen linear probe:** val AUC 0.958 (baseline 0.997); flags 11.7 % of genuine faces at 0.5; fails every
guardrail; LOFO held-out AUC 0.51–0.65 (not supported for any family). Hypothesis (CLIP features generalise better)
**rejected for B/16 in this setting**. Weights: 571 MiB, downloaded once to the HF cache after a size/disk check.

**FFT radial-spectrum probe:** val AUC 0.668; LOFO 0.49–0.61. Spectral cues are weak here (all images were JPEG
re-encoded by preprocessing, which suppresses spectral fingerprints). Dual-domain CNN therefore **skipped** per plan.

## 5. Calibration and explainability (validation)

* Calibration: baseline ECE 1.05 % (0.67 % after cross-fitted temperature, T = 1.55); J1 1.11 % (0.74 %, T = 1.63).
* Operating points (cross-fitted): at 1 % genuine-face FPR — baseline misses 6.5 % of fakes (threshold ≈ 0.90), J1 5.9 %;
  at 2 % — 3.8 % / 3.6 %; at 5 % — 0.6 % / 0.7 %.
* Abstention: app band (0.30, 0.90) → 4.8 % inconclusive, errors outside the band ≈ 1.0 % genuine-as-fake, 0.7 %
  fake-as-genuine.
* **Grad-CAM** (positive evidence toward FAKE; share inside the central 25 % face box): StyleGAN 0.83, SD 0.62,
  NeuralTextures detected 0.42, DFDC real false positives 0.43, **FF++ original false positives 0.37** vs FF++ originals
  passed 0.08, FFHQ 0.02. When the model calls an image fake, its evidence sits mostly on the face — **no sign of a
  broadcast-overlay / background shortcut**; this is descriptive, not causal.
* Crop-failure effect: FF++ originals whose face was not detected (whole frame used, 17 % of val) are flagged 87.5 % vs
  69 % with a face crop — whole-frame inputs look "more fake". The fixed app gate refuses such inputs.

## 6. Resource usage, failures, unresolved issues

* GPU jobs ran strictly one at a time (exit codes in `data/exit_codes.txt`, all 0). Peak VRAM 1.21–1.22 GB; minimum free
  commit during training 13.7 GB (gate: 6 GiB); C: 62 GB free, D: 497 GB free; stage directory 710 MB.
* GPU time ≈ 3.3 h (CLIP extraction 21 min, J1 22 min, JPEG eval ≈ 10 min, Grad-CAM ≈ 3 min, LOFO 2.6 h).
* Failures and fixes: (1) a launch with several quoted arguments was mangled by `cmd /c` quoting → nothing ran;
  switched to launcher files (verified no process had started). (2) `groupby.apply` dropped the grouping column in the
  app test → fixed and re-run. (3) Grad-CAM centre box first defined on the 7×7 map (51 % area) → corrected to the
  upsampled 224 px map before running. (4) App face-gate mismatch found by the AppTest → fixed and re-tested.
* Unresolved: DFDC label suspicions (needs DFDC metadata); FF++ within-source weakness (no higher-resolution source);
  JPEG robustness margin; unseen-generator generalisation (needs new data, not a new model); `JpegAug` RNG is not
  restored on `--resume` (no run was resumed).
* Side effects outside the stage directory: `app.py` (modified, intended), `inference.py` (new); Python bytecode cache
  `__pycache__/app.cpython-313.pyc` and `inference.cpython-313.pyc` (from `py_compile`); one CLIP weight download
  (571 MiB) in the user HF cache. Nothing else changed: 30/31 protected hashes identical (the 31st is `app.py`), 0 of 61
  prior-research files changed, 0 dataset files modified.

## 7. Candidate checkpoints

| checkpoint | path | SHA-256 (prefix) | role |
|---|---|---|---|
| **baseline (reference)** | `best_face_model.pth` | `c9a5eb5e` | deployed model |
| J1 JPEG-aug fine-tune | `research_runs/stage1_20261016/experiments/J1_jpeg_aug/candidate.pth` | `6ada9c78` | robustness candidate (criteria not fully met) |
| LOFO_ALL_control | `…/experiments/LOFO_ALL_control/candidate.pth` | `b196f334` | diagnostic only (2 epochs) |
| LOFO_diffusion_t2i | `…/experiments/LOFO_diffusion_t2i/candidate.pth` | `59a78381` | diagnostic only |
| LOFO_faceswap_video | `…/experiments/LOFO_faceswap_video/candidate.pth` | `6c287e30` | diagnostic only |
| LOFO_reenactment | `…/experiments/LOFO_reenactment/candidate.pth` | `634c56fc` | diagnostic only |
| LOFO_gan | `…/experiments/LOFO_gan/candidate.pth` | `dedcbde7` | diagnostic only |

Full hashes: `data/candidate_checkpoint_hashes.sha256`. Previous-study candidates (E0/E1/E2) remain under
`research_runs/generalization_audit_20261016/experiments/`.

## 8. Recommendation for Part 2 (final evaluation)

1. **Baseline** `best_face_model.pth` — mandatory reference, with the new app policy (bands 0.30 / 0.90, T = 1.55).
2. **J1** — the only candidate that passes all clean-data guardrails and measurably improves compression robustness; it
   did **not** meet its pre-registered margin. Evaluate it in Part 2 only as a **pre-declared secondary comparison**
   (baseline vs J1, clean + JPEG q50/q35, one shot), and report it as such. If Part 2 must test a single model, test
   the baseline.
3. Not recommended: CLIP / FFT probes (fail guardrails), LOFO models (diagnostic, 2 epochs), E0/E1/E2 (rejected earlier).
4. Before or alongside Part 2: acquire an **independent unseen-generator test set** — the LOFO results show that the
   locked test set cannot measure what matters most (new generators).
