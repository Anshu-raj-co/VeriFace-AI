# VeriFace AI — Error analysis (Phase B)

Baseline: `best_face_model.pth` (epoch 7). Test numbers are **descriptive only** (recomputed from the existing locked
predictions; nothing here was used for tuning). Hypothesis tests use **train/validation** data only.
Machine-readable: `reports/phase_b_error_analysis.json`, `reports/test_subgroups_by_*.csv`,
`reports/test_within_source_separability.csv`, `reports/baseline_within_source_val_train.csv`,
`reports/counterfactual_degradation_val.csv`. Plots: `plots/`.

CIs: Wilson (frames treated as independent — optimistic) **and** cluster bootstrap over videos / split groups (1,000
resamples; the honest one when frames share a video). `n_split_groups` is the number of independent units.

## 1. Overall (test, threshold 0.5)

| metric | value | 95 % CI (split-group bootstrap) |
|---|---:|---|
| accuracy | 0.9735 | 0.966 – 0.981 |
| balanced accuracy | 0.9737 | |
| ROC-AUC / PR-AUC | 0.9974 / 0.9973 | AUC 0.9957 – 0.9986 |
| EER | 2.68 % | 1.99 – 3.43 % |
| REAL false-positive rate | 3.31 % (282 / 8,522) | 2.39 – 4.39 % |
| FAKE false-negative rate | 1.95 % (158 / 8,113) | 0.94 – 3.27 % |
| FAKE precision / recall / F1 | 0.9658 / 0.9805 / 0.9731 | |
| REAL precision / recall (specificity) / F1 | 0.9812 / 0.9669 / 0.9740 | |

Confusion @0.5: TN 8,240 · FP 282 · FN 158 · TP 7,955.

## 2. Verified weak-case statistics (test)

| subgroup | n frames | videos | split groups | correct @0.5 | Wilson CI | cluster CI | mean / median P(FAKE) |
|---|---:|---:|---:|---:|---|---|---|
| **FF++ original (REAL)** | 278 | 100 | 50 | **33.5 %** | 28.2–39.2 | 22.5–44.3 | 0.615 / 0.709 |
| FF++ NeuralTextures | 130 | 100 | 50 | 82.3 % | 74.9–87.9 | 73.7–90.2 | 0.728 / 0.825 |
| FF++ Face2Face | 131 | 100 | 50 | 90.8 % | 84.7–94.7 | 84.6–96.2 | 0.808 |
| face_reenactment (F2F + NT) | 261 | 200 | 50 | 86.6 % | 81.9–90.2 | 81.1–91.5 | 0.768 |
| DFDC fake | 442 | 48 | 6 | 87.1 % | 83.7–89.9 | 77.0–94.2 | 0.869 / 0.9995 |
| **DFDC real** | 230 | 12 | **4** | 85.7 % | 80.5–89.6 | **69.0–100** | 0.143 |
| Celeb-DF synthesis | 253 | 170 | 5 | 92.1 % | 88.1–94.8 | 88.8–95.3 | 0.905 |
| StyleGAN | 5,785 | — | 5,785 | 99.7 % | 99.5–99.8 | 99.5–99.8 | 0.996 |
| Stable Diffusion | 749 | — | 749 | 99.6 % | 98.8–99.9 | 99.1–100 | 0.996 |
| FFHQ (REAL) | 7,091 | — | 7,091 | 99.4 % | 99.2–99.6 | 99.2–99.6 | 0.009 |

All reported baseline weak-case figures were reproduced exactly (counts and rates match `evaluate.py`).

## 3. The key finding: aggregate AUC hides weak within-source discrimination

`evaluate.py` reports each fake generator's AUC **against all test REAL images** (≈ 83 % FFHQ). That mixes in the easy
FFHQ contrast. Measured against REAL images **from the same source** (same capture, same crop pipeline):

| comparison (test) | AUC vs all REAL | **AUC vs same-source REAL** (group-bootstrap CI) | balanced acc @0.5 |
|---|---:|---:|---:|
| FF++ (all 6 methods) vs FF++ originals | ≈ 0.985 | **0.707** (0.650–0.761) | 0.623 |
| FF++ NeuralTextures vs FF++ originals | 0.980 | **0.609** (0.558–0.660) | 0.579 |
| FF++ Face2Face vs FF++ originals | 0.985 | 0.691 (0.622–0.765) | 0.621 |
| FF++ Deepfakes vs FF++ originals | 0.989 | 0.769 (0.708–0.825) | 0.648 |
| DFDC fake vs DFDC real | 0.985 | 0.954 (0.939–1.0) | 0.864 |
| Celeb-DF synthesis vs Celeb-DF real | 0.991 | 0.989 (0.979–0.996) | 0.948 |

The same pattern holds on **validation** (FF++ within-source AUC 0.720 [0.677, 0.762]; NeuralTextures 0.635) and is
not a test-set accident.

## 4. Cause analysis (each explanation backed by a measurement)

| candidate explanation | evidence | verdict |
|---|---|---|
| **Few examples** | FF++ test originals: 50 independent groups; CI 22–44 % correct, far below every other REAL group. DFDC real: 4 groups, CI 69–100 % | FF++ failure is **real, not noise**. DFDC-real "86 %" is **statistically unresolved**. |
| **Label errors** | Highest-confidence FF++ "false positives" are visibly genuine broadcast frames (`plots/examples_test_fp_ffpp_original.png`); FF++ labels come from folder provenance (0 folder/CSV conflicts in the audit); errors are consistent across frames of the same video (83 % of P(FAKE) variance is between videos; 64/100 test videos have every frame flagged) | No evidence of label noise. Video-level consistency points to a **domain/source effect**, not random mislabels. |
| **Preprocessing artifact: small-face upscaling** | Upscaled REAL images have 39.5 % FPR vs 1.5 % for non-upscaled — but this is confounded by FF++. Within FF++ originals, non-upscaled frames are flagged *more* (87 % vs 64 %). Upscaled FFHQ: 0.6 % FPR. **Causal probe on val** (`reports/counterfactual_degradation_val.csv`): down-to-120 px + cubic up-scaling of genuine faces moves Celeb-DF/DFDC/web reals toward REAL; FFHQ flagged 0.8 → 2.0 %; FF++ originals 72.4 → 76.2 % | Upscaling is a **minor** contributor at most; it does **not** reproduce the FF++ false positives. Rejected as main cause. |
| **Compression** | Val probe: JPEG q35 raises FAKE logits of genuine faces (FFHQ flagged 0.8 → 3.7 %, Celeb-DF 2.0 → 4.9 %) and lowers StyleGAN TPR 99.8 → 97.5 %. Test REAL FPR is flat over pipeline JPEG quality 60–100 | A real **robustness weakness** for heavy compression outside the training range; not the cause of the FF++ errors (FF++ shift under q35: +0.05 logit, n.s.). |
| **Source-prior shortcut** | Train label prior inside FF++ = 72 % FAKE. The model does not separate FF++ even **on training data**: within-FF++ train AUC 0.90, 45.7 % of *training* FF++ originals flagged FAKE, while Celeb-DF and DFDC train AUC ≥ 0.999. FF++ original mean P(FAKE) is 0.62–0.63 on train/val/test, i.e. close to the FF++ prior | **Supported.** Where the model cannot discriminate manipulated FF++ frames, it falls back to "FF++-looking → probably FAKE". |
| **Genuinely difficult manipulation** | NeuralTextures alters only the mouth/expression; FF++ faces are ≈ 120 px in the source and up-scaled (`plots/examples_test_fn_ffpp_NeuralTextures.png`). Within-source AUC on train is only 0.82 for NeuralTextures (vs 0.94 Deepfakes) | **Supported**: reenactment methods (NT, Face2Face) are intrinsically harder at this resolution. |
| **Threshold-dependent** | Test sweep (descriptive): raising the threshold to 0.9 cuts FF++ original FPR 66.5 → 21.2 % but NT TPR falls 82.3 → 33.8 % and overall FNR rises 1.9 → 6.8 %. Validation tells the same story | Errors are **not fixable by a global threshold**: the FF++ score distributions overlap (within-source AUC ≈ 0.7). |
| **Miscalibration** | Aggregate val ECE 1.05 % (0.67 % after cross-fitted temperature scaling, T ≈ 1.55). FF++ originals average P(FAKE) 0.63 while being 0 % FAKE | Calibrated **on average**, badly **miscalibrated inside FF++** (a subgroup calibration failure caused by the source prior). |
| **DFDC overfitting** | DFDC train within-source AUC 0.999, val 0.871 (group-bootstrap CI ≈ 0.36–1.0 across two independent bootstrap runs, 9 groups), test 0.954 (6 groups) | Consistent with memorising few DFDC videos/actors; underpowered to say more. |

## 5. Strong baseline groups

StyleGAN (99.7 %), Stable Diffusion (99.6 %), FFHQ (99.4 % correct) are near ceiling with tight CIs. These are
**in-distribution** results: the same generators (and for FFHQ/StyleGAN the same Kaggle source) are in train. They do
not show generalisation to unseen generators.

## 6. Summary of weak-case classification

| weak case | classification |
|---|---|
| FF++ originals flagged FAKE | source-prior shortcut + intrinsically hard FF++ discrimination; subgroup miscalibration; not label noise; not threshold-fixable |
| NeuralTextures / face reenactment | genuinely difficult manipulation at ≈120 px face resolution (also hard on train) |
| DFDC real / fake | small-sample (4–9 split groups); possible actor overlap; cannot be resolved with current data |
| Celeb-DF | good, but optimistic (49 identities cross splits) |
| Heavy compression | measurable robustness gap outside training range (q < 60) |
