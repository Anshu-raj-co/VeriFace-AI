# VeriFace AI — Generalisation audit (Phase C)

All model-to-model comparisons in this file use the **validation** split. Test numbers appear only as the locked,
descriptive baseline. Data: `reports/test_within_source_separability.csv`, `reports/baseline_within_source_val_train.csv`,
`reports/logo_unseen_generator_val.csv`. Plots: `plots/within_source_auc_train_val.png`,
`plots/val_logo_unseen_generator.png`.

## Four kinds of claim, kept apart

| claim | what would test it | what the data shows | status |
|---|---|---|---|
| **Aggregate performance** | the existing test split | test AUC 0.9974 (0.9957–0.9986), EER 2.68 % (1.99–3.43 %), accuracy 97.35 % | **supported** — but dominated by FFHQ vs StyleGAN (77 % of images) |
| **Within-dataset generalisation** (new images/videos of seen generators and sources) | same-source REAL vs FAKE on held-out groups | Celeb-DF 0.989 test / 0.994 val (identity overlap → optimistic); DFDC 0.954 test / 0.871 val (4–9 groups, CI 0.36–0.999); **FF++ 0.707 test / 0.720 val**; NeuralTextures 0.61 / 0.63 | **mixed**: strong for GAN/diffusion vs FFHQ and Celeb-DF; **weak for FF++**; unresolved for DFDC |
| **Cross-dataset generalisation** (train on source A, test on source B) | source-held-out training | not run as such: every source is in train. The within-source analysis shows performance depends strongly on source (FF++ vs Celeb-DF vs DFDC) | **untested**; existing numbers cannot support it |
| **Unseen-generator generalisation** | generator-held-out training | **C-LOGO experiment: held-out text-to-image fakes detected at 23.5 % (AUC 0.74) vs 99.7 % (AUC 0.999) when seen** | **not supported — large, significant gap** |

## A. Within-source evaluation

| source | fakes | REAL | test AUC (same-source REAL) | val AUC | train AUC |
|---|---|---|---:|---:|---:|
| FF++ (6 methods) | 684 | 278 | 0.707 [0.650, 0.761] | 0.720 [0.669, 0.764] | 0.900 |
| FF++ NeuralTextures | 130 | 278 | 0.609 [0.558, 0.660] | 0.635 [0.585, 0.691] | 0.818 |
| Celeb-DF | 253 | 850 | 0.989 [0.979, 0.996] | 0.994 [0.989, 0.997] | 0.9998 |
| DFDC | 442 | 230 | 0.954 [0.939, 1.0] | 0.871 [0.364, 0.999] | 0.999 |

CIs: split-group bootstrap. Train AUC being far above val/test for DFDC (and near 1 for Celeb-DF) indicates
memorisation of a few videos/actors; for FF++ the model does not even fit the training data well.

## B. Cross-generator / unseen-generator test (leave-generator-out)

Design and interpretation rule were written in `experiment_plan.md` (Addendum 1) before launch.

* **C-LOGO:** EfficientNet-B0 from ImageNet weights, trained without `stable_diffusion`, `text_to_image_unknown`,
  `unattributed_ai` (948 val images held out).
* **C-ALL:** identical recipe and budget, all generators.
* **Deviation:** both arms ran 1 epoch (7,754 / 8,230 steps) instead of 2 because of a step-budget bug; schedules were
  truncated identically. Both are under-trained relative to the 10-epoch baseline; the comparison between them stays
  controlled. See the deviation note in `experiment_plan.md`.

| validation group | C-ALL TPR@0.5 | C-LOGO TPR@0.5 | C-ALL AUC vs val REAL | C-LOGO AUC vs val REAL |
|---|---:|---:|---:|---:|
| **held-out: all three** (n = 948) | 99.7 % [99.1, 99.9] | **23.5 % [20.9, 26.3]** | 0.999 | **0.741 [0.725, 0.755]** |
| stable_diffusion (750) | 99.7 % | 23.5 % | 0.999 | 0.733 |
| text_to_image_unknown (49) | 98.0 % | 16.3 % | 0.995 | 0.713 |
| unattributed_ai (149) | 100 % | 26.2 % | 1.000 | 0.788 |
| seen: stylegan (5,790) | 99.5 % | 99.5 % | 0.998 | 0.998 |
| seen: celebdf_synthesis (385) | 93.8 % | 77.9 % | 0.984 | 0.951 |
| seen: dfdc_fake (240) | 93.3 % | 87.1 % | 0.974 | 0.964 |
| seen: ffpp_NeuralTextures (122) | 100 % | 99.2 % | 0.963 | 0.957 |

Pre-registered rule (held-out AUC ≥ 0.95 and TPR gap ≤ 10 pp): **not met** (AUC 0.74, gap 76 pp).

**Interpretation.** With StyleGAN and video face-swap/reenactment fakes in training but no text-to-image generators, the
detector misses three of four diffusion-style faces. The 99.6 % baseline detection of Stable Diffusion faces is an
**in-distribution** result: it does not show that the model would catch a new generator. (Under-training mainly
weakens the video sources; StyleGAN is at ceiling in both arms, so the held-out collapse is not a training-budget
artefact.)

## C. Leakage checks (summary; details in `baseline_audit.md` §5)

* No shared split groups, MD5s, exact raw pHashes or FF++ / DFDC video ids across splits.
* 97 cross-split raw-pHash pairs at Hamming ≤ 4 (77 FFHQ↔StyleGAN cross-label pairs, which the audit's pixel-NCC check
  classed as hash collisions; 11 DFDC real↔fake; 9 Celeb-DF).
* **Celeb-DF: 49 identities appear in more than one split** (synthesis source identity). Celeb-DF metrics are optimistic
  for unseen people.
* DFDC fakes cannot be linked to their source real video by name; actor overlap across splits cannot be ruled out.

No alternative split was created or applied to the locked test results.

## D. Statistical reliability

* Frames from one video are correlated: for FF++ originals 83 % of the variance of P(FAKE) is between videos. All
  subgroup CIs here resample **split groups / videos**, not frames. Frame-level Wilson intervals are reported alongside
  in the CSVs and are visibly too narrow for video sources.
* DFDC (4 test / 9 val split groups) and the small diffusion subsets (`text_to_image_unknown`: 49–50 images) have wide
  intervals; no conclusion about them should rest on a point estimate.

## What would be needed for stronger claims

1. **Unseen generators:** a test set from generators absent from training (e.g. newer diffusion models, other face-swap
   tools), collected with identity/video metadata. The LOGO result says this matters.
2. **Cross-dataset:** a source-held-out protocol (e.g. train without FF++, test on FF++) — needs more independent
   video groups per source than currently exist (FF++ test has 50 split groups).
3. **Identity-disjoint Celeb-DF/DFDC splits:** requires linking fake videos to source actors (not possible from DFDC file
   names).
4. **Higher-resolution FF++ faces** (raw/c23 frames with tighter crops) to tell resolution limits apart from method
   difficulty for NeuralTextures/Face2Face.
