# VeriFace AI -- Phase 1 Dataset Audit Report

Generated: 2026-10-15 14:59:17  |  pHash threshold: Hamming <= 4  |  seed: 42

Label convention: **FAKE = 1 (positive class)**, REAL = 0.

## Raw inventory

| source | label | images |
|---|---|---:|
| gravex_200k | FAKE | 100,000 |
| gravex_200k | REAL | 100,000 |
| kaggle_140k | FAKE | 70,000 |
| kaggle_140k | REAL | 70,000 |

### Per generator (raw)

| source | generator | label | images |
|---|---|---|---:|
| gravex_200k | augmented_copy | FAKE | 4,895 |
| gravex_200k | augmented_copy | REAL | 14 |
| gravex_200k | celebdf_real | REAL | 7,235 |
| gravex_200k | celebdf_synthesis | FAKE | 11,878 |
| gravex_200k | celebdf_youtube_real | REAL | 3,421 |
| gravex_200k | dfdc_fake | FAKE | 20,499 |
| gravex_200k | dfdc_real | REAL | 9,448 |
| gravex_200k | ffhq | REAL | 68,037 |
| gravex_200k | ffpp_DeepFakeDetection | FAKE | 884 |
| gravex_200k | ffpp_Deepfakes | FAKE | 2,000 |
| gravex_200k | ffpp_Face2Face | FAKE | 2,000 |
| gravex_200k | ffpp_FaceShifter | FAKE | 2,000 |
| gravex_200k | ffpp_FaceSwap | FAKE | 2,000 |
| gravex_200k | ffpp_NeuralTextures | FAKE | 2,000 |
| gravex_200k | ffpp_original | REAL | 11,101 |
| gravex_200k | stable_diffusion | FAKE | 7,500 |
| gravex_200k | stylegan | FAKE | 41,844 |
| gravex_200k | text_to_image_unknown | FAKE | 1,000 |
| gravex_200k | unattributed_ai | FAKE | 1,500 |
| gravex_200k | web_photo_real | REAL | 744 |
| kaggle_140k | ffhq | REAL | 70,000 |
| kaggle_140k | stylegan | FAKE | 70,000 |

### Resolution / format

- **kaggle_140k**: 256x256 (140,000)
- **gravex_200k**: 256x256 (200,000)
- format/mode: JPEG/RGB (340,000)

### File size & JPEG quantisation by generator (raw)

| generator | n | median KB | median luma quant |
|---|---:|---:|---:|
| augmented_copy | 4,909 | 8.6 | 29.03 |
| celebdf_real | 7,235 | 5.2 | 29.03 |
| celebdf_synthesis | 11,878 | 5.1 | 29.03 |
| celebdf_youtube_real | 3,421 | 5.6 | 29.03 |
| dfdc_fake | 20,499 | 5.0 | 29.03 |
| dfdc_real | 9,448 | 5.1 | 29.03 |
| ffhq | 138,037 | 20.7 | 5.77 |
| ffpp_DeepFakeDetection | 884 | 11.7 | 29.03 |
| ffpp_Deepfakes | 2,000 | 10.8 | 29.03 |
| ffpp_Face2Face | 2,000 | 10.8 | 29.03 |
| ffpp_FaceShifter | 2,000 | 10.8 | 29.03 |
| ffpp_FaceSwap | 2,000 | 10.9 | 29.03 |
| ffpp_NeuralTextures | 2,000 | 10.8 | 29.03 |
| ffpp_original | 11,101 | 11.0 | 29.03 |
| stable_diffusion | 7,500 | 8.5 | 29.03 |
| stylegan | 111,844 | 24.5 | 5.77 |
| text_to_image_unknown | 1,000 | 10.4 | 29.03 |
| unattributed_ai | 1,500 | 7.6 | 29.03 |
| web_photo_real | 744 | 7.8 | 29.03 |

## GRAVEX label CSV cross-check

- CSV code meaning REAL: `1` (VeriFace AI uses FAKE = 1; CSV labels are not used)
- folder vs CSV label conflicts: 0

## Cross-dataset overlap (same filename in Kaggle 140k and GRAVEX)

- **ffhq**: {"shared_filenames": 68037, "identical_bytes": 0, "phash_hamming_median": 0.0, "phash_hamming_le_4": 68037, "phash_hamming_le_10": 68037}
- **stylegan**: {"shared_filenames": 41844, "identical_bytes": 0, "phash_hamming_median": 0.0, "phash_hamming_le_4": 41844, "phash_hamming_le_10": 41844}
- **random_pairs_baseline**: {"median": 30.0, "p0_1": 14.0, "frac_le_4": 0.0, "frac_le_10": 0.00025}

## Deduplication

| step | images removed / pairs |
|---|---:|
| unreadable | 0 |
| augmented_copy | 4,909 |
| unattributed_generator | 0 |
| csv_folder_label_conflict | 0 |
| md5_label_conflict | 0 |
| md5_duplicate | 3 |
| phash_same_generator_pairs | 441,710 |
| phash_cross_generator_same_label_pairs (kept, grouped) | 33,290 |
| phash_cross_label_pairs (kept, grouped) | 120,501 |
| phash_near_duplicate | 170,328 |
| phash_edges_crossing_identity_groups | 247,887 |
|   verified duplicates (NCC >= 0.95) | 69,292 |
|   rejected as hash collisions | 178,595 |

Total removed: **175,240** of 340,000; kept **164,760**.

## Leak-free split (StratifiedGroupKFold, 10 folds -> 8/1/1)

- split groups: 138,953; largest group: 474 images
- leakage checks (group / MD5 / pHash shared across splits): {'group_id': 0, 'md5': 0, 'verified_near_dup_pairs_across_splits': 0}

| split | REAL | FAKE | total |
|---|---:|---:|---:|
| train | 65,993 | 65,691 | 131,684 |
| val | 8,465 | 7,976 | 16,441 |
| test | 8,522 | 8,113 | 16,635 |

| generator | train | val | test |
|---|---:|---:|---:|
| celebdf_real | 4,257 | 616 | 598 |
| celebdf_synthesis | 3,200 | 385 | 253 |
| celebdf_youtube_real | 2,235 | 258 | 252 |
| dfdc_fake | 3,383 | 240 | 442 |
| dfdc_real | 1,192 | 156 | 230 |
| ffhq | 55,717 | 7,075 | 7,091 |
| ffpp_DeepFakeDetection | 230 | 0 | 23 |
| ffpp_Deepfakes | 995 | 123 | 133 |
| ffpp_Face2Face | 988 | 123 | 131 |
| ffpp_FaceShifter | 988 | 120 | 132 |
| ffpp_FaceSwap | 987 | 125 | 135 |
| ffpp_NeuralTextures | 985 | 122 | 130 |
| ffpp_original | 2,013 | 286 | 278 |
| stable_diffusion | 6,000 | 750 | 749 |
| stylegan | 46,328 | 5,790 | 5,785 |
| text_to_image_unknown | 407 | 49 | 50 |
| unattributed_ai | 1,200 | 149 | 150 |
| web_photo_real | 579 | 74 | 73 |

## Metadata shortcut probe (train -> test, raw files)

| feature set | model | accuracy | balanced acc | ROC-AUC | verdict |
|---|---|---:|---:|---:|---|
| metadata (w, h, file size, aspect) | logistic_regression | 0.5583 | 0.5536 | 0.5689 | ok (near chance) |
| metadata (w, h, file size, aspect) | decision_tree | 0.5800 | 0.5756 | 0.6075 | ok (near chance) |
| metadata + JPEG quant table | logistic_regression | 0.5662 | 0.5595 | 0.5501 | ok (near chance) |
| metadata + JPEG quant table | decision_tree | 0.5800 | 0.5755 | 0.6074 | ok (near chance) |

## Caveats

- pHash cannot see horizontal flips or crops; GRAVEX pre-augmented copies are therefore dropped by filename provenance (augmentation_log.csv lists their originals), not by hashing.
- Celeb-DF fakes are grouped by target identity only; the source-face identity may appear (as REAL video of that person) in another split. Linking both collapses Celeb-DF into a single group. Celeb-DF YouTube real clips carry no identity label, so a celebrity can appear in them and in another split.
- DFDC fake videos cannot be linked to their real source video by filename; only pairs that are pixel-level duplicates (pHash + NCC check) are grouped, so a DFDC actor can still appear in multiple splits.
- pHash edges linking two different identity groups are trusted only when NCC >= --min-ncc; weaker matches were verified to be collisions between different people (Celeb-DF, FFHQ vs StyleGAN).
- Cross-label near-duplicate pairs are kept (both labels) but forced into the same split; review audit_reports/cross_label_near_duplicates.csv for possible label noise.
- Diffusion coverage is small (stable_diffusion + text_to_image_unknown + unattributed_ai); per-generator metrics for these will have wide confidence intervals.
