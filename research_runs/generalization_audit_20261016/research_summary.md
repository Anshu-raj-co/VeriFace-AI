# VeriFace AI — Research summary: error analysis, generalisation audit, targeted experiments

Run directory: `research_runs/generalization_audit_20261016/` · Date: 2026-10-16 · Baseline: `best_face_model.pth`
(EfficientNet-B0, epoch 7) — **unchanged and still the recommended model.**

## 1. What was done

| phase | done | key outputs |
|---|---|---|
| A. artifact / split / leakage audit | yes | `baseline_audit.md`, `reports/phase_a_audit.json`, `data/phash_le4_pairs_across_splits.csv` |
| B. error analysis (locked test, descriptive) + cause tests (train/val) | yes | `error_analysis.md`, `reports/test_subgroups_by_*.csv`, `reports/*within_source*.csv`, `reports/counterfactual_degradation_val.csv`, `plots/examples_*.png` |
| C. generalisation (within-source; leave-generator-out) | yes (cross-dataset untested) | `generalization_audit.md`, `reports/logo_unseen_generator_val.csv` |
| D. targeted experiments (pre-registered) | E0, E1, E2 | `experiment_plan.md`, `experiments/*/`, `experiment_results.csv` |
| E. fair comparison on validation | yes | `experiment_results.csv`, `reports/val_decision.json`, `plots/val_experiment_comparison.png` |
| F. calibration / thresholds (validation) | yes | `reports/calibration_val.csv`, `threshold_table_val.csv`, `operating_points_val_crossfit.csv`, `abstention_band_val.csv` |
| G. final test confirmation | **not run** — no candidate met the criteria, so there was nothing new to confirm | — |

## 2. Verified weak cases (test, recomputed from the stored predictions)

| subgroup | n (videos / split groups) | correct @0.5 | 95 % CI (video/group bootstrap) | mean P(FAKE) |
|---|---|---:|---|---:|
| FF++ originals (REAL) | 278 (100 / 50) | 33.5 % | 22.5–44.3 % | 0.615 |
| FF++ NeuralTextures | 130 (100 / 50) | 82.3 % | 73.7–90.2 % | 0.728 |
| face reenactment (NT + F2F) | 261 (200 / 50) | 86.6 % | 81.1–91.5 % | 0.768 |
| DFDC fake | 442 (48 / 6) | 87.1 % | 77.0–94.2 % | 0.869 |
| DFDC real | 230 (12 / **4**) | 85.7 % | 69.0–100 % | 0.143 |
| StyleGAN | 5,785 | 99.7 % | 99.5–99.8 % | 0.996 |
| Stable Diffusion | 749 | 99.6 % | 99.1–100 % | 0.996 |

All baseline figures you reported reproduce exactly. **The FF++ result is worse than the per-generator table suggests:**
against FF++ originals (same source, same pipeline) the model separates FF++ fakes with **AUC 0.707 test / 0.720 val**
(NeuralTextures 0.61 / 0.63). The 0.98+ "AUC vs all real" in `evaluate.py` is inflated by the easy FFHQ reals.

## 3. Leakage and shortcuts

* **Leakage:** no shared split groups, MD5s, exact pHashes, or FF++/DFDC video ids across splits. **Celeb-DF has
  identity leakage** (49 identities in more than one split) → Celeb-DF numbers are optimistic. DFDC actor overlap
  can't be ruled out. 97 cross-split near-duplicate pHash pairs (mostly FFHQ↔StyleGAN cross-label collisions).
* **Source-prior shortcut — supported.** FF++ training data are 72 % FAKE; the model cannot discriminate FF++ well even on
  *training* data (train within-FF++ AUC 0.90, 46 % of training originals flagged), and FF++ originals sit at
  P(FAKE) ≈ 0.62 on every split. Re-weighting the FF++ prior (E1) moved FF++ scores exactly as this predicts.
* **Small-face upscaling shortcut — rejected as main cause** by a counterfactual on validation images (FF++-style
  down/up-scaling doesn't turn genuine faces FAKE).
* **Compression:** heavy JPEG (q35, below the q60–100 training range) raises genuine-face FAKE scores (FFHQ flagged
  0.8 → 3.7 %) and lowers StyleGAN detection (99.8 → 97.5 %) — a robustness gap.
* **Label noise:** no evidence (top-confidence FF++ "errors" are genuine frames; errors are video-consistent).

## 4. Generalisation claims

| claim | status |
|---|---|
| Aggregate in-distribution performance (AUC 0.997, EER 2.7 %) | **supported** |
| Within-source: GAN / diffusion vs FFHQ; Celeb-DF | supported (Celeb-DF optimistic due to identity overlap) |
| Within-source: FF++ | **weak** (AUC ≈ 0.71) |
| Within-source: DFDC | **unresolved** (4–9 independent groups) |
| Cross-dataset | **untested** (every source is in training) |
| **Unseen generators** | **not supported.** Holding text-to-image generators out of training drops their detection from 99.7 % to **23.5 %** (AUC 0.999 → 0.74) at equal budget |

## 5. Experiments actually run (validation; pre-registered criteria)

All arms fine-tuned from the frozen baseline for 1 epoch (8,230 steps, ≈ 19–25 min, peak 1.2 GB VRAM, ≥ 17 GB commit free).

| model | val EER | bal. acc @0.5 | FF++ orig FPR | within-FF++ AUC | NT TPR | FFHQ FPR | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline (epoch 7) | **2.66 %** | 97.40 % | 72.4 % | 0.720 | 85.2 % | **0.52 %** | reference |
| E0 control (+1 epoch, low LR) | 2.82 % | 97.25 % | 57.0 % | 0.734 | 76.2 % | 1.16 % | control |
| E1 source-class-balanced loss | 3.02 % | 97.00 % | **42.7 %** | 0.734 | 63.1 % | 1.14 % | rejected: prior shift only (Δ AUC vs E0 CI ±0.004) |
| E2 video sources oversampled ×4 | **2.61 %** | **97.45 %** | 59.1 % | 0.722 | 77.9 % | 0.83 % | rejected: no FF++ gain (H2 unsupported) |

Plus the Phase C pair C-LOGO / C-ALL (§4). Full tables: `experiment_results.csv`, `reports/val_decision.json`.

**Findings:** (1) E0 shows subgroup error rates at 0.5 shift a lot under *any* small training change, so single-run
subgroup gains must be judged against a control. (2) Removing the FF++ prior trades FF++ false positives for FF++ misses
one-for-one; it does not improve discrimination. (3) More FF++ exposure doesn't help at this face resolution. (4) At
matched genuine-face FPR the baseline is as good as or better than every candidate (miss rate 6.5 % at 1 % FPR, 3.8 % at
2 % FPR, cross-fitted on val). **No new candidate is selected; the baseline remains the model.** Candidate checkpoints
exist only under `experiments/*/candidate.pth` for reproducibility.

## 6. Calibration and thresholds (validation only, baseline)

* Aggregate calibration is good: ECE 1.05 % raw, 0.67 % after temperature scaling (T = 1.55, cross-fitted over val
  split groups). Temperature scaling doesn't change decisions at 0.5 — it only makes reported probabilities more
  honest. FF++ is miscalibrated *as a subgroup* (mean P(FAKE) 0.63 for genuine faces) and no global calibration fixes that.
* Threshold trade-off (val):

| threshold | REAL FPR | FAKE FNR | FF++ orig FPR | FFHQ FPR | use |
|---:|---:|---:|---:|---:|---|
| 0.28 | ≈ 5 % | 0.6 % | ≈ 87 % | 0.9 % | catch nearly all fakes (screening) |
| 0.50 | 3.9 % | 1.25 % | 72 % | 0.52 % | current default |
| 0.79 | ≈ 2 % | 3.8 % | ≈ 35 % | 0.2 % | 2 % genuine-face FPR target |
| 0.90 | ≈ 1 % | 6.5 % | 12 % | 0.17 % | 1 % genuine-face FPR target (accusations are costly) |

* **Abstention band** (0.1, 0.9): abstains on 5.7 % of val images, error rate on the rest 0.72 %, and it captures 80 % of
  FF++ originals. For a user-facing app where wrongly calling a genuine face "fake" is the expensive error, a
  threshold near 0.8–0.9 **plus** an "uncertain" band is better supported than 0.5. Choose the threshold from this table
  for your cost ratio; these are val estimates and should be confirmed once on independent data.

## 7. Recommended next actions (with compute)

1. **Fix `app.py` preprocessing (high priority, minutes):** it resizes the whole upload to 224 px instead of
   `crop_face_square` → 256 px → `CenterCrop(224)`, so the deployed app doesn't see what the model was evaluated
   on. Also add the `positive_class` check and an uncertainty band.
2. **Get unseen-generator test data** (recent diffusion models, other face-swap tools; with video/identity ids) — the LOGO
   result shows this is the biggest unknown. No GPU needed to evaluate (≈ 2 min per 16 k images).
3. **FF++ at higher resolution:** re-extract FF++ faces from source frames with tighter crops (no up-scaling) and retrain
   (≈ 3 h GPU for a full 10-epoch run). Tests whether NeuralTextures/Face2Face failures are resolution-limited.
4. **Compression augmentation** (JPEG q 30–60 at train time) for the measured robustness gap: one 10-epoch run ≈ 3 h,
   compared with a matched control.
5. **Multi-generator-held-out protocol** (fix the step-budget bug first): LOGO per family (GAN, diffusion, face-swap,
   reenactment), ≈ 1–1.5 h GPU per held-out family at 2 epochs.
6. CLIP-based baseline: **not possible offline** — no CLIP weights are cached locally; it would need a download (~350–600 MB).
   Proposed, not run.

## 8. Integrity, deviations, disclosures

* **Protected files unchanged:** all 28 hashes in `data/protected_hashes_before.sha256` re-verified OK after all runs
  (`best_face_model.pth`, `last_checkpoint.pth`, `cleaned_metadata.csv`, `train.py`, `model.py`, `evaluate.py`,
  `preprocess_data.py`, `audit_dataset.py`, all `training_reports/`, `evaluation_reports/`, `audit_reports/` files,
  `evaluation_test_20261016.log`). `dataset_processed/`: 164,760 files, 0 modified.
* Python refreshed `__pycache__/train.cpython-313.pyc` and `model.cpython-313.pyc` at 09:38 when the research scripts
  imported `train.py`/`model.py` (automatic bytecode cache built from the unchanged sources; harmless).
* **Test-set use:** the weak cases came from the existing test evaluation (your report, re-verified descriptively). Every
  hypothesis was then re-established on train/val, and all experiment design, selection, calibration and thresholds used
  validation only. No new test inference was run. Because no candidate was selected, there was no Phase G test run; any
  future candidate should be confirmed on the locked test **and** on independent data, since the weak cases were first
  noticed on test.
* **Deviation:** the LOGO arms ran 1 epoch instead of 2 (step-budget bug in `build_order`); both arms were affected
  identically, and this was recorded in `experiment_plan.md` before C-ALL's result was seen.
* `app.py` was not modified.

## 9. File index

`baseline_audit.md` · `error_analysis.md` · `generalization_audit.md` · `experiment_plan.md` (pre-registration +
addenda) · `experiment_results.csv` · `reports/` (CSV/JSON metrics) · `plots/` (12 PNGs) · `scripts/`
(`phase_ab_audit.py`, `infer.py`, `analyze_val_probes.py`, `finetune.py`, `train_logo.py`, `compare_val.py`,
`analyze_logo.py`, `make_plots.py`, `plot_logo.py`) · `experiments/<arm>/` (config, history, val predictions, summary,
checkpoints) · `logs/` (every run's log + exit code) · `data/` (hashes, prediction files, row lists).
