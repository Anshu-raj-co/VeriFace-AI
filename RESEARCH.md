# VeriFace AI — Research record

Status as of 2026-10-16. This file consolidates the training run, the generalisation audit, Stage 1 and Part 2
(final selection and evaluation). Each number below was re-read from the artifact named next to it. Paths are relative
to the project root `D:\PROJECT\VeriFace AI`.

**Bottom line**

* **Selected model: `best_face_model.pth`** — the original baseline (EfficientNet-B0, epoch 7). SHA-256
  `c9a5eb5ea0d082ad99b2039fb63290f6c94c28372fc2ee88c75c2837a1d0eb11`. No other candidate met its pre-registered
  criteria.
* **Test-split performance (clean images):**
  * Threshold-free: ROC-AUC 0.9974, EER 2.68 %.
  * At the conventional 0.5 threshold: accuracy 97.35 %. 3.3 % of genuine images are flagged as fake, and 1.9 % of
    fakes are missed.
  * Under the app's three-way policy: 1.0 % of genuine images are called "manipulated", 1.4 % of fakes are called
    "authentic", and 4.3 % of images are left "inconclusive".
* **Not a clean benchmark:** this test split is not untouched — see §10.
* **Unseen fake families are not detected.** The model is reliable only on the fake *families* it was trained on.
  When a whole family is held out of training, held-out GAN faces score at chance (AUC 0.50), and face-swap and
  reenactment fakes score about 0.59 (§6).
* **Known weak spot:** FF++ broadcast frames remain the model's main weakness (genuine vs fake AUC ≈ 0.71).
* **No claim of robust real-world or unseen-generator detection is supported** by the data available here.

---

## 1. Objective and scope

VeriFace AI classifies a face image as genuine (REAL = 0) or AI-generated / manipulated (FAKE = 1, the positive
class). Sigmoid(logit) is the model's FAKE score. Scope:

* still images of single faces;
* sources limited to the two local collections in §2.

Video-level, audio, and multi-face forensics are out of scope.

## 2. Dataset, label audit, provenance, splits

**Composition** (`cleaned_metadata.csv`, 164,760 images after cleaning):

| split | REAL | FAKE | total | split groups |
|---|---:|---:|---:|---:|
| train | 65,993 | 65,691 | 131,684 | 110,982 |
| val | 8,465 | 7,976 | 16,441 | 13,980 |
| test | 8,522 | 8,113 | 16,635 | 13,991 |

Sources and generators (test counts in brackets):

* `kaggle_140k`:
  * FFHQ real (7,091)
  * StyleGAN (5,785)
* `gravex_200k`:
  * Stable Diffusion (749)
  * text_to_image_unknown (50)
  * unattributed_ai (150)
  * web_photo_real (73)
  * Celeb-DF real (598), Celeb-DF YouTube real (252), Celeb-DF synthesis (253)
  * DFDC real (230), DFDC fake (442)
  * FF++ originals (278), Deepfakes (133), Face2Face (131), FaceShifter (132), FaceSwap (135), NeuralTextures (130),
    DeepFakeDetection (23)

Every generator is single-label. FF++, DFDC and Celeb-DF are video frames, so many frames share a video.

**Split protocol** (`audit_reports/audit_report.md`):

* **Method:** StratifiedGroupKFold, 10 folds, assigned 8/1/1 to train/val/test.
* **Groups:** `group_id` is a connected component over identity and video keys plus verified near-duplicates.
* **Leakage checks:** 0 group, 0 MD5 and 0 verified near-duplicate overlaps across splits.
* **Known residual risks:**
  * 49 Celeb-DF identities appear in more than one split, so Celeb-DF numbers are optimistic.
  * DFDC actor overlap cannot be ruled out.
  * DFDC has only 4 (test) and 9 (val) independent split groups.

**Label audit** (Stage 1, `research_runs/stage1_20261016/label_integrity_report.md`):

* **Provenance:** labels vs provenance tokens in file paths gave 0 conflicts across all 164,760 images, and there were
  0 duplicate files (MD5).
* **Cross-label look-alike pairs:** 7,903 REAL↔FAKE near-duplicate pairs were checked at the pixel level. None is a
  copy; FF++ edits touch a median of only about 5–7 % of the frame.
* **Model–label disagreements:** 106 confident disagreements were reviewed (about 70 images visually). **No label
  error was confirmed.**
* **Unresolved — DFDC:**
  * 5 of 276 DFDC "fake" videos have every frame scored as genuine.
  * 3 of 64 DFDC "real" videos average a score above 0.90.
  * This is consistent with DFDC's audio-only or second-person fakes, but it is **unverifiable** without DFDC's
    per-video `metadata.json`.
  * Review lists: `research_runs/stage1_20261016/reports/dfdc_suspicious_videos.csv` and
    `research_runs/stage1_20261016/reports/label_model_disagreement_candidates.csv`.
* **Untouched:** no labels, metadata or split assignments were changed at any stage.

## 3. Baseline model and training

* **Architecture:** `model.py` — timm EfficientNet-B0 (ImageNet init), with a single-logit head.
* **Preprocessing** (`preprocess_data.py`, `face_utils.crop_face_square`): a 256 px square face crop. Training uses
  RandomCrop 224, horizontal flip, mild colour jitter (p 0.5) and Gaussian blur (p 0.2); evaluation uses CenterCrop 224 with ImageNet normalisation.
* **Training** (`train.py`, `training_reports/training_summary.json`):
  * Optimisation: BCE loss, AdamW (LR 3e-4, weight decay 0.01), 1-epoch warm-up then cosine schedule. The backbone
    runs at ×0.3 LR with the stem frozen.
  * Batch and runtime: batch 16, AMP, seed 42, 0 workers.
  * Stopping and selection: patience 3, max 15 epochs. The run stopped after epoch 10, and **epoch 7 was selected on
    val EER**.
  * Cost: 82,300 steps, 188 min.
* **Validation at epoch 7** (`training_reports/history.csv`): AUC 0.9974 and EER 2.62 %. Training-time validation
  runs in fp16 autocast; the fp32 re-inference used for research comparisons gives EER 2.66 %. The difference is
  precision only: median |Δlogit| 0.02, and 1 of 16,441 decisions at 0.5 differs.

## 4. Application: fixes, inference flow, verdict bands

**Defects found** in the original `app.py`, now fixed (backup at `research_runs/stage1_20261016/data/app.py.original_backup`):

1. It resized the whole upload to 224 px with no face crop, so the app did not see what the model was evaluated on.
2. It had no FAKE-positive checkpoint check.
3. It gave a binary verdict at 0.5 and showed a "confidence %".
4. It drew Grad-CAM on the wrong image.
5. Its face gate was stricter than the training cropper.

**Current flow** (`inference.py`, used by `app.py`):

1. `load_checkpoint` refuses a missing file, an unsupported format, an unknown backbone, or `positive_class != "FAKE"`.
2. Preprocessing: `crop_face_square` (256 px) → CenterCrop 224 → ImageNet normalisation. This is identical to
   `evaluate.py`.
3. The model outputs a logit; the raw score is `sigmoid(logit)`.
4. **The verdict is based on the raw score** (thresholds fitted on validation, `research_runs/stage1_20261016/reports/app_policy.json`):

   | raw score | verdict |
   |---|---|
   | below 0.30 | LIKELY AUTHENTIC |
   | 0.90 or above | LIKELY MANIPULATED |
   | in between | INCONCLUSIVE |

5. **The displayed score is temperature-scaled** with T = 1.5523, fitted on validation. The UI labels it "Manipulation
   score (calibrated on validation data) — not a guaranteed probability for this photo", and also shows the
   uncalibrated score and the raw logit.
6. If the cropper finds no face, the app refuses to analyse the image ("CANNOT VERIFY — NO FACE").
7. Grad-CAM is drawn on the 224 px model input.

**Verification:**

* **Stage 1:** parity with the evaluation pipeline on 255 raw validation images gives Spearman 0.993 (old app 0.938).
* **Part 2:** all tests pass (`research_runs/part2_final_20261016/app_tests/app_smoke_tests.json`).

| case (validation images / synthetic) | expected | result |
|---|---|---|
| FFHQ genuine (raw 0.006) | LIKELY AUTHENTIC | pass (AppTest banner + calibrated 3.5 %) |
| Celeb-DF genuine | LIKELY AUTHENTIC | pass |
| StyleGAN, Stable Diffusion | LIKELY MANIPULATED | pass |
| FF++ original (raw 0.45) | INCONCLUSIVE + warning | pass |
| blank image, random noise | refused, no verdict | pass ("No human face detected", Run button disabled) |
| REAL-positive / bare state_dict / wrong backbone / missing checkpoint | refused | pass (app shows "Model Unavailable", no verdict) |
| static: app loads `best_face_model.pth` with the selected SHA, no whole-image resize, constants = `app_policy.json` | — | pass |
| headless `streamlit run app.py` | health ok | `/_stcore/health` ok, page HTTP 200, server stopped |

**Test-run history:**

* The first two Part 2 runs failed on test-design errors, not app defects. Both are kept as
  `app_smoke_tests_run1_naive_pick.json` and `app_smoke_tests_run2_strict_banner_check.json`.
  * Run 1: the first FFHQ validation image chosen as "genuine" is genuinely borderline. It scored 0.57 in the app and
    0.49 in the evaluation pipeline, so INCONCLUSIVE is the correct verdict.
  * Run 2: the assertion wrongly rejected the app's "MODEL UNAVAILABLE" placeholder banner.
* Some "val" images live under a source folder named `train` (for example
  `dataset_images/real-vs-fake/train/real/00022.jpg`). The project split, not the source folder, decides membership.

**App limitations:**

* The bands and the temperature were fitted for the baseline only.
* The score reflects data like the training sources. It is not a probability for arbitrary photos.
* Genuine FF++ broadcast frames often land in the inconclusive band: 57 % on test and 73 % on validation (§8).
* Images where no face is found are refused.
* The app does not detect unseen fake families (§6).

## 5. Experiments (all selection on validation)

| study | experiment | configuration | result | status |
|---|---|---|---|---|
| audit | E0 control | +1 epoch from baseline, LR 1e-4 | val EER 2.82 % | control |
| audit | E1 source-class-balanced loss | 1 epoch | FF++ prior shift only, no discrimination gain | rejected |
| audit | E2 video sources ×4 | 1 epoch | no FF++ gain | rejected |
| audit | C-LOGO / C-ALL | text-to-image held out | — | superseded: sampler bug gave 1 epoch, not 2 |
| Stage 1 | **LOFO ×4 + ALL control** | ImageNet init, 2 full epochs, sampler fixed, step counts asserted | §6 | completed |
| Stage 1 | **J1 JPEG augmentation** | 1-epoch fine-tune, JPEG q30–60 at p = 0.5; matched control E0 | §7 | margin **not met** |
| Stage 1 | CLIP ViT-B/16 linear probe | frozen features, logistic regression C = 1 | val AUC 0.958; LOFO 0.51–0.65; fails all guardrails | negative |
| Stage 1 | FFT radial-spectrum probe | logistic regression on 112-bin log spectrum | val AUC 0.668; LOFO 0.49–0.61 | negative |
| Stage 1 | dual-domain RGB+FFT CNN | — | — | skipped by pre-registered rule (FFT probe did not transfer) |
| Stage 1 | CLIP ViT-L/14 | — | — | skipped (1.6 GB download, ≈ 1 h GPU, B/16 clearly negative) |
| Stage 1 | higher-resolution FF++ | — | — | **infeasible:** all FF++ frames are 256 px, faces ≈ 120 px, no source videos |

**Sources:**

* Audit: `research_runs/generalization_audit_20261016/research_summary.md`, `experiment_results.csv`.
* Stage 1: `research_runs/stage1_20261016/STAGE1_REPORT.md`, `experiment_plan.md` (pre-registered),
  `reports/stage1_val_panel.csv`.

## 6. Leave-family-out generalisation (corrected)

Part 2 re-verified these runs from the raw artifacts (`research_runs/part2_final_20261016/reports/verify_lofo.csv`):

* All 5 runs completed exactly 2 epochs (`steps_done == steps_planned == 2 × steps_per_epoch`).
* None is a smoke test.
* No held-out generator appears in its run's training list.
* Held-out AUCs recompute exactly from each run's `val_predictions.csv`.

| held-out family | TPR@0.5 seen → held out | AUC vs matched genuine, seen → held out [95 % CI] | verdict (pre-registered) |
|---|---|---|---|
| diffusion / text-to-image | 99.6 % → 15.3 % | 0.9997 → **0.812** [0.796, 0.826] | partial |
| GAN (StyleGAN) | 99.3 % → 0.0 % | 0.9996 → **0.504** [0.493, 0.513] | not supported |
| face-swap (video) | 97.3 % → 19.9 % | 0.930 → **0.591** [0.566, 0.614] | not supported |
| reenactment (F2F, NT) | 96.3 % → 67.8 % | 0.606 → **0.587** [0.538, 0.631] | not supported |

The model learns family-specific artefacts: **unseen-generator detection is not supported**. These are relative
comparisons between same-budget 2-epoch models, not deployable models. The reenactment "TPR" reflects the FF++ source
prior, not detection.

## 7. JPEG robustness and the J1 decision

**Pre-registered J1 criteria** (`research_runs/stage1_20261016/experiment_plan.md`, S1-B), each measured on validation
at q35 against the matched control E0. Part 2 re-verified them in
`research_runs/part2_final_20261016/reports/verify_handoff.json`.

| criterion | J1 / E0 | met? |
|---|---|---|
| C1 StyleGAN-TPR drop ≤ ½ of E0's | 0.41 / 0.45 pp (ratio 0.92) | **no** |
| C2 FFHQ-FPR rise ≤ ½ of E0's | 4.04 / 6.87 pp (ratio 0.59) | **no** |
| C3 clean EER ≤ E0 + 0.3 pp | 2.58 % vs 2.82 % | yes |
| C4 within-FF++ AUC ≥ E0 − 0.02 | 0.726 vs 0.734 | yes |

**J1 does not meet its margin, so under the selection rule the baseline is retained.** This was declared in writing
before any Part 2 test inference (`research_runs/part2_final_20261016/PREDECLARATION.md`).

**Handoff correction:** the Stage 1 handoff said J1 missed only the FFHQ criterion (C2, "58 %"). In fact C1 is also
formally false, and the C2 ratio is 0.588, which rounds to 59 %. The decision is unchanged.

**Validation summary** (`research_runs/stage1_20261016/reports/jpeg_robustness_val.csv`):

| model | EER clean | EER q50 | EER q35 |
|---|---:|---:|---:|
| baseline | 2.62 % | 4.59 % | 4.93 % |
| E0 | 2.82 % | 4.92 % | 5.50 % |
| J1 | 2.58 % | 3.72 % | 4.30 % |

J1 also has subgroup shifts on clean validation images that were not guardrails:

* Celeb-DF genuine FPR 1.9 → 3.4 %.
* web-photo genuine FPR 0 → 5.4 %.
* FF++ fake detection lower (Face2Face 94 → 85 %).
* FF++ original FPR lower (72 → 57 %).

## 8. Final test evaluation (Part 2)

**Protocol** (pre-declared in `research_runs/part2_final_20261016/PREDECLARATION.md`):

* **Primary:** the baseline on clean images, from the existing `evaluate.py` run
  (`evaluation_reports/evaluation_test_20261016_091737.json` and the predictions CSV). Recomputed: identical.
* **Secondary:** a one-shot, descriptive comparison of baseline vs J1 on clean, JPEG q50 and JPEG q35 images
  (`scripts/eval_test_arms.py`, fp32).
  * The fresh baseline clean arm reproduces the `evaluate.py` predictions: max |Δp| 1.1e-7, 0 decision changes.
  * No threshold, temperature or weight was fitted on test.
  * CIs come from a group bootstrap over `group_id` (500 resamples, 13,991 groups).

**Labels used in the tables below:**

* **@0.5** = conventional threshold on the raw score.
* **app** = the validation-derived three-way policy (§4).
* AUC, PR-AUC and EER are threshold-free.

### 8.1 Selected model (baseline), clean test split — `reports/test_final_summary.json`

| metric | value [95 % CI] |
|---|---|
| ROC-AUC | 0.9974 [0.9956, 0.9986] |
| PR-AUC (FAKE positive) | 0.9973 |
| EER | 2.68 % [2.02, 3.46] (threshold 0.641) |
| accuracy @0.5 | 97.35 % [96.54, 98.06] |
| precision / recall / F1 (FAKE) @0.5 | 96.58 % / 98.05 % / 97.31 % |
| genuine images flagged FAKE @0.5 | 3.31 % [2.35, 4.37] (282 / 8,522) |
| fakes missed @0.5 | 1.95 % [0.93, 3.31] (158 / 8,113) |
| confusion @0.5 [TN, FP, FN, TP] | [8,240, 282, 158, 7,955] |
| app: genuine → AUTHENTIC / INCONCLUSIVE / MANIPULATED | 95.81 % / 3.16 % / **1.03 %** [0.63, 1.49] (8,165 / 269 / 88) |
| app: fake → MANIPULATED / INCONCLUSIVE / AUTHENTIC | 93.20 % / 5.45 % / **1.36 %** [0.51, 2.60] (7,561 / 442 / 110) |
| app: inconclusive share | 4.27 % [3.40, 5.14] (validation estimate: 4.8 %) |
| app: precision of MANIPULATED / AUTHENTIC verdicts | 98.85 % / 98.67 %; accuracy on decided images 98.76 % |
| calibration (15-bin ECE, descriptive) | raw 1.03 %; with the validation temperature 0.39 % |

**Subgroups, baseline clean @0.5** (`reports/test_subgroups_baseline_vs_J1.csv`):

| generator (label) | n (split groups) | flagged FAKE @0.5 | app verdicts (auth / inconcl / manip) |
|---|---|---:|---|
| FFHQ (REAL) | 7,091 | 0.59 % | 99.1 / 0.8 / 0.2 % |
| Celeb-DF real (REAL) | 598 (6) | 2.3 % | 96.0 / 3.7 / 0.3 % |
| Celeb-DF YouTube (REAL) | 252 (29) | 3.2 % | 96.4 / 1.6 / 2.0 % |
| web photo (REAL) | 73 | 0 % | 100 / 0 / 0 % |
| DFDC real (REAL) | 230 (**4**) | 14.3 % | 83.0 / 12.6 / 4.3 % |
| **FF++ originals (REAL)** | 278 (50) | **66.5 %** | 21.6 / **57.2** / 21.2 % |
| StyleGAN | 5,785 | 99.69 % | 0.2 / 0.6 / 99.2 % |
| Stable Diffusion | 749 | 99.6 % | 0.1 / 0.8 / 99.1 % |
| text_to_image_unknown / unattributed_ai | 50 / 150 | 100 % / 100 % | — |
| Celeb-DF synthesis | 253 (5) | 92.1 % | 5.9 / 11.1 / 83.0 % |
| DFDC fake | 442 (6) | 87.1 % | 11.5 / 8.1 / 80.3 % |
| FF++ Deepfakes / FaceSwap / FaceShifter | 133 / 135 / 132 (50) | 96.2 / 93.3 / 91.7 % | 42–56 % inconclusive |
| FF++ Face2Face / NeuralTextures | 131 / 130 (50) | 90.8 / 82.3 % | 56 / 55 % inconclusive |
| **within-FF++ AUC** (FF++ fakes vs FF++ originals) | 962 (51) | **0.707** | — |

The DFDC and Celeb-DF rows rest on 4–6 independent groups each and are not reliable estimates.

### 8.2 Secondary comparison: baseline vs J1 (identical conditions) — `reports/test_baseline_vs_J1_by_arm.csv`, `reports/test_J1_minus_baseline_paired.csv`

| metric | baseline clean | J1 clean | baseline q50 | J1 q50 | baseline q35 | J1 q35 |
|---|---:|---:|---:|---:|---:|---:|
| ROC-AUC | 0.9974 | 0.9977 | 0.9931 | 0.9956 | 0.9894 | 0.9930 |
| EER | 2.68 % | 2.58 % | 4.40 % | 3.67 % | 5.21 % | 4.34 % |
| accuracy @0.5 | 97.35 % | 97.42 % | 95.40 % | 96.10 % | 94.72 % | 95.29 % |
| genuine flagged @0.5 (FP) | 3.31 % (282) | 3.11 % (265) | 6.64 % (566) | 5.62 % (479) | 6.85 % (584) | 6.86 % (585) |
| fakes missed @0.5 (FN) | 1.95 % (158) | 2.03 % (165) | 2.47 % (200) | 2.10 % (170) | 3.64 % (295) | 2.45 % (199) |
| app genuine → MANIPULATED | 1.03 % | 1.10 % | 2.29 % | 1.70 % | 2.16 % | 2.24 % |
| app fake → AUTHENTIC | 1.36 % | 1.33 % | 1.77 % | 1.37 % | 2.47 % | 1.43 % |
| app inconclusive | 4.27 % | 3.79 % | 6.45 % | 5.31 % | 7.98 % | 6.69 % |

The app columns apply the baseline's bands to J1 unchanged, for comparability only.

**Paired J1 − baseline differences, 95 % CI:**

* **Clean:** EER −0.10 pp [−0.38, +0.15], AUC +0.0003 [−0.0003, +0.0011], genuine FP −0.20 pp [−0.67, +0.25]. No
  difference is distinguishable from zero except a lower inconclusive share (−0.49 pp [−0.90, −0.10]).
* **q50:** EER −0.73 pp [−1.13, −0.38]; genuine FP −1.02 pp [−1.59, −0.40]; app genuine → manipulated −0.59 pp
  [−0.87, −0.28].
* **q35:** EER −0.86 pp [−1.28, −0.43]; fakes missed −1.18 pp [−1.87, −0.55]; app fake → authentic −1.04 pp
  [−1.63, −0.45]. Genuine FP is unchanged (+0.01 pp [−0.71, +0.58]).
* **Same pattern as C2:** under q35 the rise in FFHQ false positives is 3.53 pp for the baseline and 3.75 pp for J1.
  The validation criterion C2 failed for the same reason.

**J1 subgroup regressions on clean test images** (FP or detection counts @0.5; the Celeb-DF figures rest on few
groups):

* **Celeb-DF genuine:** FP 14 → 32 of 598.
* **Celeb-DF YouTube genuine:** FP 8 → 12 of 252.
* **web photo genuine:** FP 0 → 2 of 73.
* **FF++ fakes detected:**
  * Face2Face 119 → 100 of 131
  * FaceSwap 126 → 115 of 135
  * NeuralTextures 107 → 94 of 130

**J1 subgroup gains on clean test images:**

* **FF++ originals:** FP 185 → 152 of 278.
* **DFDC genuine:** FP 33 → 23 of 230.
* **Celeb-DF synthesis detected:** 233 → 246 of 253.
* **DFDC fakes detected:** 385 → 406 of 442.
* **Not a gain — within-FF++ AUC:** 0.707 → 0.714, essentially unchanged. J1 shifts FF++ scores; it does not
  separate FF++ fakes from originals better.

**Reading:**

* J1 is measurably more robust to heavy JPEG compression in aggregate, consistent with validation.
* It does not reduce genuine-face false positives under q35.
* It trades FF++ fake detection for fewer FF++ false positives.
* It raises false positives on some genuine video sources.

This is descriptive and cannot overturn the pre-declared selection; it also gives no reason to revisit it.

### 8.3 Selection

**Retained: `best_face_model.pth`.** The SHA-256 was verified before and after Part 2
(`c9a5eb5e…a1d0eb11`). `app.py` loads exactly this file (`app_tests/app_smoke_tests.json`, `static.sha_matches_selected = true`).

J1 (`research_runs/stage1_20261016/experiments/J1_jpeg_aug/candidate.pth`, SHA-256 `6ada9c78…5de18f`) remains a
documented robustness candidate. Adopting it would require a new pre-registered criterion and independent data —
not this test split.

## 9. Calibration, explainability, uncertainty

**Calibration** (validation, `research_runs/stage1_20261016/reports/calibration_val_stage1.csv`):

* Baseline ECE: 1.05 % raw, 0.67 % after cross-fitted temperature scaling (T = 1.55).
* On test (descriptive): 1.03 % raw → 0.39 % with the validation T.
* No global calibration fixes the FF++ subgroup, whose genuine faces score high on average.

**Operating points** (validation, cross-fitted), stated as fakes missed at a fixed genuine-image false-positive rate:

| genuine FPR target | threshold | fakes missed |
|---|---:|---:|
| 1 % | ≈ 0.90 | 6.5 % |
| 2 % | ≈ 0.79 | 3.8 % |
| 5 % | ≈ 0.28 | 0.6 % |

**Grad-CAM** (validation, `research_runs/stage1_20261016/reports/gradcam_center_share_summary_val.csv`). The table
gives the share of FAKE evidence inside the central 25 % face box:

| group | share in face box |
|---|---:|
| StyleGAN | 0.83 |
| Stable Diffusion | 0.62 |
| NeuralTextures detected | 0.42 |
| DFDC real false positives | 0.43 |
| FF++ original false positives | 0.37 |
| FF++ originals passed | 0.08 |
| FFHQ | 0.02 |

* There is no sign of a broadcast-overlay or background shortcut. This reading is descriptive, not causal.
* FF++ originals where the cropper found no face are flagged 87.5 % of the time, vs 69 % with a face crop. The app
  therefore refuses images where no face is found.

**Uncertainty handling:** the three-way bands. On test, 4.3 % of images are inconclusive, and the verdicts that are
given are correct 98.8 % of the time.

## 10. Test-set exposure (read before citing test numbers)

* The test split was evaluated once after training (`evaluation_reports/*_20261016_091737.*`).
* Its predictions were then examined in the generalisation audit (error analysis, subgroup tables, example images) and
  descriptively in the Stage 1 label audit (49 test disagreement candidates).
* Weak subgroups (FF++, NeuralTextures, DFDC) were first noticed on test. Every hypothesis was then re-tested on
  train/val, and no model, threshold or temperature was ever fitted on test.
* Part 2 added one pre-declared pass: baseline and J1, three arms each.
* **This split is therefore a used benchmark, not an untouched, independent one.**
* It also cannot measure what matters most: every source and generator in it also appears in training.
* **An independent test set with generators and sources absent from training** is required before any claim about
  real-world or unseen-generator detection.

## 11. Unresolved limitations

1. No generalisation to unseen fake families (§6). Untested cross-dataset; all sources appear in training.
2. FF++ within-source discrimination is weak (test AUC 0.707). Higher-resolution FF++ is infeasible with local data.
3. Compression lowers performance: test EER 2.68 → 5.21 % at q35. J1 partly addresses this but failed its criteria.
4. DFDC label suspicions are unverifiable without DFDC `metadata.json`. DFDC and Celeb-DF estimates rest on 4–29
   groups; Celeb-DF has identity overlap across splits.
5. Single-run results. No seed variance was measured for the baseline.
6. Score bands and temperature were fitted on validation data from the same sources. The app's score is not a
   probability for arbitrary photos.
7. `train.py` resume quirk: a mid-epoch resume shifts the augmentation RNG by one draw. `research_runs/stage1_20261016/scripts/train_exp.py` does not
   restore the JpegAug RNG on `--resume`. No reported run was resumed.

## 12. Reproduction

Use Python 3.13 with `pip install -r requirements.txt` (Windows; RTX 3060 6 GB used). Launch long GPU jobs one at a
time, detached, with ≥ 6 GiB Windows commit headroom.

```bash
# data (already done; do not re-run over the existing dataset)
python audit_dataset.py            # -> audit_reports/, cleaned_metadata.csv with group-aware splits
python preprocess_data.py          # -> dataset_processed/<split>/<label>/*.jpg (256 px face crops)

# baseline training (defaults: 15 epochs, bs 16, AMP, 0 workers, patience 3) and evaluation
python train.py                    # resume after interruption: python train.py --resume
python evaluate.py                 # test split, best_face_model.pth -> evaluation_reports/

# Stage 1 (run from research_runs/stage1_20261016; launcher: scripts\run_job.cmd NAME script args)
python scripts/train_exp.py --init imagenet --epochs 2 --lr 3e-4 --warmup-frac 0.1 --out experiments/LOFO_gan --exclude-generators stylegan
python scripts/train_exp.py --init "D:\PROJECT\VeriFace AI\best_face_model.pth" --out experiments/J1_jpeg_aug --epochs 1 --lr 1e-4 --aug-jpeg "30,60,0.5"
python scripts/eval_jpeg_val.py baseline=<ckpt> E0=<ckpt> J1=<ckpt>
python scripts/extract_clip.py; python scripts/extract_fft.py; python scripts/probe_lofo.py --features clip   # (and fft)
python scripts/gradcam_analysis.py; python scripts/analyze_stage1.py

# Part 2 (run from research_runs/part2_final_20261016)
python scripts/verify_handoff.py
python scripts/eval_test_arms.py "baseline=D:\PROJECT\VeriFace AI\best_face_model.pth" "J1=...\J1_jpeg_aug\candidate.pth"
python scripts/analyze_test.py
python scripts/app_smoke_tests.py
```

**App:**

```bash
streamlit run app.py
```

Run it from the project root; it loads `best_face_model.pth` and needs CUDA or CPU.

## 13. Artifact index

| what | path |
|---|---|
| selected checkpoint | `best_face_model.pth` (SHA-256 `c9a5eb5ea0d082ad99b2039fb63290f6c94c28372fc2ee88c75c2837a1d0eb11`) |
| app / inference | `app.py`, `inference.py` |
| training record | `training_reports/` (`history.csv`, `training_summary.json`, `train_full_20261016.log`) |
| original test evaluation | `evaluation_reports/evaluation_test_20261016_091737.json`, `predictions_test_20261016_091737.csv` |
| data audit / splits | `audit_reports/audit_report.md` |
| generalisation audit | `research_runs/generalization_audit_20261016/research_summary.md` |
| Stage 1 | `research_runs/stage1_20261016/STAGE1_REPORT.md`, `label_integrity_report.md`, `experiment_plan.md`, `reports/`, `experiments/` |
| Part 2 pre-declaration | `research_runs/part2_final_20261016/PREDECLARATION.md` |
| Part 2 handoff verification | `research_runs/part2_final_20261016/reports/verify_handoff.json`, `verify_lofo.csv` |
| Part 2 test results | `research_runs/part2_final_20261016/reports/test_final_summary.json`, `test_baseline_vs_J1_by_arm.csv`, `test_J1_minus_baseline_paired.csv`, `test_subgroups_baseline_vs_J1.csv`; predictions in `data/test_arms/` |
| Part 2 app tests | `research_runs/part2_final_20261016/app_tests/app_smoke_tests.json`, `streamlit_server_health.txt` |
| Part 2 logs | `research_runs/part2_final_20261016/logs/` |
| candidate hashes | `research_runs/stage1_20261016/data/candidate_checkpoint_hashes.sha256` |

**Part 2 integrity check:**

* **What Part 2 created:** this `RESEARCH.md` and `research_runs/part2_final_20261016/`. Nothing else was created or
  modified.
* **Protected files:** 30 of 31 hashes from the Stage 1 start list are identical. The exception is `app.py`, changed
  in Stage 1 as intended and unchanged since.
* **Earlier research and checkpoints:** 61 of 61 prior-research files and all 6 Stage 1 candidate checkpoints are
  identical.
* **Dataset:** `dataset_processed/` still holds 164,760 files, with 0 modified.
* **Disclosure:** an ad-hoc check wrote a bytecode cache at
  `research_runs/generalization_audit_20261016/scripts/__pycache__/`. It was removed, and that directory is restored.
