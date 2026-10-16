# VeriFace AI — Baseline audit (Phase A)

Generated 2026-10-16. Scripts: `scripts/phase_ab_audit.py` (read-only). Machine-readable: `reports/phase_a_audit.json`.
Protected-file hashes recorded before any experiment: `data/protected_hashes_before.sha256`.

## 1. Artifacts verified

| Item | Value |
|---|---|
| Baseline checkpoint | `best_face_model.pth`, SHA-256 `c9a5eb5ea0d082ad99b2039fb63290f6c94c28372fc2ee88c75c2837a1d0eb11`, epoch 7, `model_type=vision`, `backbone=efficientnet_b0`, `positive_class=FAKE` |
| Architecture match | loads into `model.build_model("vision","efficientnet_b0")` with strict `load_state_dict` (done by `evaluate.py` and `scripts/infer.py`) |
| Locked test predictions | `evaluation_reports/predictions_test_20261016_091737.csv` (16,635 rows) + `evaluation_test_20261016_091737.json`, exit code 0 |
| Prediction ↔ metadata join | by `image_id`: 16,635/16,635 matched, 0 path mismatches, 0 label mismatches, 0 non-test rows |
| Label convention | FAKE = 1, REAL = 0 in metadata, training, `evaluate.py`, checkpoint (`positive_class="FAKE"`) |

## 2. Metadata schema (columns actually present)

`image_id, local_image_path, raw_path, label, label_numeric, dataset_split, source_dataset, generator, generator_family,
group_id, orig_split, width, height, aspect_ratio, file_size, jpeg_luma_q, md5, phash, face_detected, num_faces,
crop_side, upscaled, jpeg_quality, proc_file_size, proc_width, proc_height, proc_aspect_ratio`

* label: `label` / `label_numeric`; split: `dataset_split`; source: `source_dataset` (kaggle_140k, gravex_200k);
  manipulation: `generator`, `generator_family`.
* **No explicit identity / source-video / original↔fake column.** `group_id` is the split group (connected component
  built by `audit_dataset.py` over shared identity/video keys + verified near-duplicates). Video and identity keys are
  recoverable from `raw_path` file names for FF++ (`real_<vid>`, `fake_<method>_<target>_<source>`), DeepFakeDetection
  (actor ids), Celeb-DF (`id<target>_id<source>`), Celeb-DF YouTube (clip id) and DFDC (video id). FFHQ / StyleGAN /
  diffusion images have no identity metadata.

## 3. Counts

| split | REAL | FAKE | total |
|---|---:|---:|---:|
| train | 65,993 | 65,691 | 131,684 |
| val | 8,465 | 7,976 | 16,441 |
| test | 8,522 | 8,113 | 16,635 |

| generator | label | train | val | test |
|---|---|---:|---:|---:|
| ffhq | REAL | 55,717 | 7,075 | 7,091 |
| stylegan | FAKE | 46,328 | 5,790 | 5,785 |
| stable_diffusion | FAKE | 6,000 | 750 | 749 |
| celebdf_real | REAL | 4,257 | 616 | 598 |
| celebdf_synthesis | FAKE | 3,200 | 385 | 253 |
| celebdf_youtube_real | REAL | 2,235 | 258 | 252 |
| dfdc_fake | FAKE | 3,383 | 240 | 442 |
| dfdc_real | REAL | 1,192 | 156 | 230 |
| ffpp_original | REAL | 2,013 | 286 | 278 |
| ffpp_Deepfakes / Face2Face / FaceShifter / FaceSwap / NeuralTextures | FAKE | ≈ 990 each | ≈ 120 each | ≈ 130 each |
| ffpp_DeepFakeDetection | FAKE | 230 | **0** | 23 |
| unattributed_ai | FAKE | 1,200 | 149 | 150 |
| text_to_image_unknown | FAKE | 407 | 49 | 50 |
| web_photo_real | REAL | 579 | 74 | 73 |

**Composition risk.** FFHQ + StyleGAN (Kaggle 140k, one GAN family, one real source) are 77 % of every split. Aggregate
metrics are therefore dominated by one easy, single-generator pairing.

**Within-source label priors (train).** FF++ 72 % FAKE, DFDC 74 % FAKE, Celeb-DF 33 % FAKE. A model that recognises the
*source* can lower its loss by predicting that source's majority label.

**Effective sample sizes are small for the weak groups.** Test FF++ originals: 278 frames from 100 videos in 50 split
groups. Test DFDC real: 230 frames, 12 videos, **4 split groups**. Val DFDC: 9 split groups in total. Val has no
DeepFakeDetection images.

## 4. Preprocessing verification

* Training: PIL `convert("RGB")` → `RandomCrop(224)` of the 256 px face crop → light augmentation → ImageNet normalisation.
  Evaluation/validation: `CenterCrop(224)` → same normalisation (`train.get_eval_transforms`, imported by `evaluate.py`).
  **Consistent.**
* All images were re-encoded by `preprocess_data.py` (face crop, resize to 256, JPEG q~U[60,100] per image). The JPEG
  quality factor carries no label signal (REAL FPR is flat across quality bins on test: 3.5 / 3.6 / 3.2 / 2.9 %).
* **Deployment mismatch (finding):** `app.py` scores uploads with `transforms.Resize((224,224))` on the *whole* image and
  does not call `face_utils.crop_face_square`, contrary to the face_utils docstring. App predictions are therefore made on
  differently framed and resampled inputs than every reported metric. `app.py` also does not check
  `positive_class`. Not modified (out of scope); recommended fix in `research_summary.md`.

### Crop-size trace (a potential shortcut)

| generator | upscaled | median crop side (px) |
|---|---:|---:|
| ffpp_original (REAL) | 84 % | 119 |
| ffpp fakes (5 methods) | 77–83 % | 119–124 |
| stable_diffusion (FAKE) | 77 % | 191 |
| unattributed_ai (FAKE) | 75 % | 187 |
| ffhq / stylegan / Celeb-DF / DFDC / web | 0.3–2 % | 256 |

In train, P(FAKE | upscaled) = 0.775 vs P(FAKE | not upscaled) = 0.469. FF++ originals are the only large REAL group that
is mostly upscaled. This was tested causally on validation images (Phase B probe) — see `error_analysis.md` §4: upscaling
alone does **not** reproduce the FF++ false positives.

## 5. Leakage audit

| check | result |
|---|---|
| split `group_id` shared across splits | 0 |
| identical MD5 across splits | 0 |
| identical raw pHash across splits | 0 |
| raw pHash Hamming ≤ 4 across splits (pigeonhole banding, all pairs) | **97 pairs**: 77 FFHQ↔StyleGAN (cross-label, the audit's NCC check classified FFHQ/StyleGAN pHash links as collisions), 11 DFDC real↔DFDC fake, 9 Celeb-DF. `data/phash_le4_pairs_across_splits.csv` |
| FF++ video ids (original + both target and source of every fake) across splits | 0 — FF++ is fully video-disjoint |
| DFDC video ids across splits | 0 (but DFDC fakes cannot be linked to their source real video by file name, so the same actor can appear across splits — audit caveat) |
| **Celeb-DF identity ids across splits** | **49 identities** cross splits (via the *source* identity of synthesis videos, and real videos of that person). Touches 486 test celebdf_real and 506 test celebdf_synthesis images. |

Interpretation: there is no image-, hash- or FF++-video-level leakage. Celeb-DF has **identity-level leakage** by design
(the audit grouped synthesis videos by target identity only, to avoid collapsing Celeb-DF into one group), so Celeb-DF
test/val numbers are optimistic as a measure of generalisation to unseen people. The 11 DFDC near-duplicate pairs across
splits (real in one split, fake in another) are consistent with the known actor overlap.

## 6. Limitations that block strong generalisation claims

1. No held-out generator: every test generator also appears in train (except that val lacks DeepFakeDetection). The
   existing split measures **in-distribution** generalisation to new images/videos of seen generators, not unseen-generator
   generalisation.
2. Celeb-DF identity overlap across splits; DFDC actor overlap cannot be ruled out.
3. DFDC and FF++ effective sample sizes are tens of video groups, so subgroup CIs are wide.
4. Strong aggregate metrics are dominated by FFHQ vs StyleGAN (same Kaggle source, identical raw JPEG history).
   File-level cues are removed by the preprocessing re-encode (the audit's shortcut probe is near chance), but an
   aggregate AUC mixes easy cross-source contrasts with hard within-source ones, so within-source discrimination must be
   measured separately (see `error_analysis.md`).
