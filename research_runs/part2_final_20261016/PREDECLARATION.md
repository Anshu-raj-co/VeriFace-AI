# Part 2 — pre-declaration (written 2026-10-16 before any Part-2 test-set inference)

## Model selection (decided on validation, now — not revisable by test results)

* Pre-registered rule (`research_runs/stage1_20261016/experiment_plan.md`, "Selection for Part 2"): a candidate replaces
  the baseline only if, on **validation**, it beats the baseline on its target weakness **by the pre-declared margin**
  without guardrail violations.
* J1's pre-declared margin (S1-B), re-verified from the raw prediction files (`reports/verify_handoff.json`):
  C1 StyleGAN-TPR drop under q35 ≤ ½ of E0's → **false** (ratio 0.92);
  C2 FFHQ-FPR rise under q35 ≤ ½ of E0's → **false** (ratio 0.59);
  C3 clean EER ≤ E0 + 0.3 pp → true; C4 within-FF++ AUC ≥ E0 − 0.02 → true. **Criteria not met.**
* **Decision: the selected model is the baseline `best_face_model.pth`** (SHA-256
  `c9a5eb5ea0d082ad99b2039fb63290f6c94c28372fc2ee88c75c2837a1d0eb11`). This decision is final for this stage. No
  test-set result below can change it, change a threshold, or change the temperature.

## Test-set evaluation plan (one shot, no iteration)

1. **Baseline, clean (primary):** reuse the existing `evaluate.py` run
   (`evaluation_reports/evaluation_test_20261016_091737.json` + predictions CSV; recomputation verified identical).
   Additional metrics (app policy, CIs, subgroups) are computed from those stored predictions.
2. **Secondary, descriptive comparison — baseline vs J1** on the test split under identical conditions: clean, JPEG
   q50, JPEG q35 (decode → JPEG Q, 4:2:0 → decode → `get_eval_transforms`), fp32, batch 64, one pass per model × arm,
   script `scripts/eval_test_arms.py`. The baseline clean arm must reproduce the stored `evaluate.py` predictions
   (sanity check for the pipeline).
3. Metrics, each labelled with its threshold: ROC-AUC, PR-AUC, EER (threshold-free); accuracy, precision, recall, F1,
   confusion matrix, genuine-image FPR, fake miss rate at the **conventional 0.5**; three-way verdict shares at the
   **validation-derived app policy** (raw score < 0.30 authentic, ≥ 0.90 manipulated, else inconclusive — fitted for the
   baseline; applied unchanged to J1 for comparability, which is noted as a limitation). Per-generator / per-source
   subgroup rates; group-bootstrap 95 % CIs over `group_id`; paired J1 − baseline CIs.
4. Validity: J1 was trained on the train split only and selected on validation; the test split was never used for its
   development. **Limitation:** the baseline's test predictions and subgroup errors were already examined in the
   generalization audit and the Stage-1 label audit (descriptive), so this test split is not an untouched benchmark,
   and the comparison is descriptive — it cannot serve as an independent confirmation of J1.
