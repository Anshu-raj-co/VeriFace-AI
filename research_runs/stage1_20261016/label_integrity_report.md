# VeriFace AI — Dataset & label-integrity audit (Stage 1)

Scripts: `scripts/label_audit.py`, `scripts/label_review_sheets.py`; logs `logs/label_audit.log`,
`logs/dfdc_video_level.log`. Outputs: `reports/label_audit_summary.json`, `reports/label_provenance_conflicts.csv`,
`reports/cross_label_pairs_pixel_check.csv`, `reports/label_model_disagreement_candidates.csv`,
`reports/dfdc_suspicious_videos.csv`, `plots/label_review_*.png`. Nothing in the dataset, metadata or splits was changed.

**Bottom line:** no label error could be *confirmed*. There are *suspicious, unverifiable* groups — most notably a
handful of DFDC videos — and structural risks (identity overlap, small groups). This audit cannot certify that all
labels are correct: visual review covered ~70 images, and several checks rely on file provenance, which is only as good
as the upstream datasets.

## 1. What was checked

| check | scope | result |
|---|---|---|
| Label vs. provenance tokens in the raw path (`/real/`, `/fake/`, `ai_images/`, `real_`, `fake_`, `ai_detect_real/fake`, `_SD_`, `_men_`) | all 164,760 | **0 conflicts, 0 ambiguous** (FAKE 81,780 / REAL 82,980 all consistent) |
| GRAVEX label CSV vs. folder (original audit) | 200,000 raw | 0 conflicts |
| Identical files (MD5) across labels / anywhere | all | 0 / 0 |
| Cross-label near-duplicate pairs (pHash ≤ 4, from the original audit), **pixel-verified on raw images** | 7,903 pairs where both images were kept | **0 pixel copies.** Every pair differs by ≥ 22 grey levels at some pixel; see §2 |
| Model–label confident disagreement (P(FAKE) > 0.98 for REAL, < 0.02 for FAKE) | val (all), train video sources, existing test predictions (descriptive) | 106 candidates (42 val, 15 train, 49 test) (§3) |
| Manual review of the most confident disagreements | 6 contact sheets, ≈ 70 images | no image is visibly mislabelled; several are unverifiable (§3) |
| Split leakage (re-confirmed) | all | no shared groups / MD5 / exact pHash / FF++ or DFDC video ids across splits; **49 Celeb-DF identities in > 1 split** (previous study) |

## 2. Can a FAKE be a genuine image (or vice versa)? — pixel check of cross-label pairs

Face-swap and reenactment fakes share their frame with an original video, so pHash pairs REAL↔FAKE are expected. If a
"FAKE" were pixel-identical to its "REAL" partner, it would carry no manipulation.

| fake generator | pairs | median PSNR | max PSNR | median share of pixels changed > 8 levels | min share |
|---|---:|---:|---:|---:|---:|
| ffpp_Deepfakes | 1,110 | 33.9 dB | 41.0 dB | 5.2 % | 1.2 % |
| ffpp_Face2Face | 1,146 | 35.1 dB | 45.4 dB | 5.5 % | 0.8 % |
| ffpp_FaceShifter | 1,162 | 35.5 dB | 41.4 dB | 7.0 % | 1.4 % |
| ffpp_FaceSwap | 1,146 | 34.8 dB | 41.6 dB | 5.5 % | 1.3 % |
| ffpp_NeuralTextures | 1,142 | 35.8 dB | 45.6 dB | 5.6 % | 0.5 % |
| celebdf_synthesis | 1,778 | 28.1 dB | 36.4 dB | 36 % | 10 % |
| dfdc_fake | 152 | 23.0 dB | 36.5 dB | 48 % | 9 % |
| stylegan (vs FFHQ) | 267 | 13.1 dB | 17.9 dB | 98 % | 73 % |

* No pair is a pixel copy. The two most similar pairs (FF++ video 387, Face2Face and NeuralTextures, 45 dB) still differ
  by up to 34–35 grey levels, concentrated in the centre (face) box (MAE centre 1.1–1.2 vs border 0.2–0.3) — a small but
  real manipulation.
* FF++ manipulations touch a median of only ~5–7 % of the 256 px frame (the face is ≈ 120 px). This quantifies why FF++
  reenactment is hard: the signal is small and has been up-scaled.
* FFHQ↔StyleGAN pHash pairs are completely different images (PSNR ≤ 18 dB) — hash collisions, not leaked copies.

## 3. Model–label disagreement candidates (suspicious, not confirmed)

| set | generator | candidates / n | reading after review |
|---|---|---:|---|
| val | dfdc_real | 23 / 156 (14.7 %) | 9 frames from one video (`ehccixxzoe`): heavily blurred, low-resolution face, no visible manipulation; others from videos of an actor who also appears in DFDC fake videos. **Unverifiable.** |
| test (descr.) | dfdc_fake | 20 / 442 | mostly one actor across several "fake" videos with no visible swap. DFDC contains audio-only fakes and multi-person videos where only one face is swapped — such frames are visually genuine. **Plausible label noise, unverifiable** without DFDC's per-video manipulation metadata. |
| train | dfdc_fake | 12 / 3,383 | same pattern |
| val/test | celebdf_synthesis | 12 / 638 | blurry / profile frames; no visible evidence either way |
| val/test | ffpp_original | 12 / 564 | genuine-looking broadcast frames (model weakness, see error analysis) |
| val/test | ffhq | 6 / 14,166 | genuine photos (one strong colour cast) — no evidence of mislabelling |
| val/test | stylegan | 7 / 11,575 | photorealistic GAN outputs with typical background artefacts — label plausible |

Video-level view (DFDC, `reports/dfdc_suspicious_videos.csv`): 5 of 276 DFDC fake videos have **every** frame scored
< 0.10 (7 frames); 3 of 64 DFDC real videos average > 0.90 (11 frames). DFDC fakes whose file name ends in face index 1
(a second person in the video) are detected less often (88 % vs 97 % for index 0) — consistent with, but not proof of,
unswapped second faces being labelled FAKE.

## 4. Structural risks (not label errors)

* **Celeb-DF identity overlap** (49 identities in more than one split) → Celeb-DF val/test are optimistic.
* **DFDC actor overlap** cannot be ruled out (fake videos are not linkable to their source by name); DFDC val/test have
  only 9 / 4 independent split groups.
* **Upstream provenance**: `web_photo_real`, `unattributed_ai`, `text_to_image_unknown` come from an aggregated
  "AI detect" collection whose own labelling process is not documented here; their labels are accepted on provenance.
* **Frame-level correlation**: up to hundreds of frames per video; any per-frame metric overstates the sample size.

## 5. Recommendations

1. Do **not** relabel anything on the basis of this audit — nothing was confirmed.
2. Keep a **review list** (`reports/label_model_disagreement_candidates.csv`, `reports/dfdc_suspicious_videos.csv`) and,
   if DFDC's original `metadata.json` (per-video label + original-video mapping) can be obtained, verify the DFDC
   candidates against it; that is the single most informative check left.
3. For future data: store identity / source-video / original↔fake ids as columns, not only in file names.
4. Report DFDC and Celeb-DF metrics with group-bootstrap CIs and the leakage caveats.
