"""
Part 2, step 1: independently re-verify the Stage-1 handoff from raw artifacts (no GPU, no test inference).
Recomputes LOFO held-out AUCs from each run's val_predictions.csv, the J1 pre-registered criteria from the JPEG
prediction files, the baseline test metrics from the stored test predictions, and checks split/policy consistency.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

BASE = r"D:\PROJECT\VeriFace AI"
S1 = os.path.join(BASE, r"research_runs\stage1_20261016")
OUT = os.path.join(BASE, r"research_runs\part2_final_20261016\reports")
FAMILIES = {"diffusion_t2i": ["stable_diffusion", "text_to_image_unknown", "unattributed_ai"], "gan": ["stylegan"],
            "faceswap_video": ["celebdf_synthesis", "dfdc_fake", "ffpp_Deepfakes", "ffpp_FaceSwap", "ffpp_FaceShifter",
                               "ffpp_DeepFakeDetection"], "reenactment": ["ffpp_Face2Face", "ffpp_NeuralTextures"]}
MATCHED = {"diffusion_t2i": ["ffhq"], "gan": ["ffhq"],
           "faceswap_video": ["celebdf_real", "celebdf_youtube_real", "dfdc_real", "ffpp_original"],
           "reenactment": ["ffpp_original"]}


def eer(y, s):
    f, t, _ = roc_curve(y, s)
    fn = 1 - t
    i = np.nanargmin(np.abs(fn - f))
    return float((f[i] + fn[i]) / 2)


res = {"issues": []}
# --- LOFO ---
rows = []
for arm in ["ALL_control", "diffusion_t2i", "faceswap_video", "reenactment", "gan"]:
    d = os.path.join(S1, "experiments", f"LOFO_{arm}")
    sm = json.load(open(os.path.join(d, "summary.json")))
    cf = json.load(open(os.path.join(d, "config.json")))
    vp = pd.read_csv(os.path.join(d, "val_predictions.csv"))
    fam = arm if arm in FAMILIES else None
    r = {"run": f"LOFO_{arm}", "epochs": sm["epochs"], "steps_done": sm["steps_done"], "steps_planned": sm["steps_planned"],
         "steps_per_epoch": cf.get("steps_per_epoch"),
         "complete_2_epochs": sm["steps_done"] == sm["steps_planned"] == 2 * cf.get("steps_per_epoch"),
         "smoke_test": sm.get("smoke_test"), "n_val": len(vp),
         "val_auc_recomputed": roc_auc_score(vp.label_numeric, vp.prob_fake), "val_auc_summary": sm["val_auc_final"],
         "excluded_from_train": ";".join(sorted(cf.get("excluded_from_train", [])))}
    if fam:
        s = vp.loc[vp.generator.isin(FAMILIES[fam]), "prob_fake"]
        m = vp.loc[vp.generator.isin(MATCHED[fam]), "prob_fake"]
        r["heldout_auc_vs_matched_real"] = roc_auc_score(np.r_[np.zeros(len(m)), np.ones(len(s))], np.r_[m, s])
        r["heldout_tpr@0.5"] = float((s >= .5).mean())
        r["no_heldout_generator_in_train"] = not (set(cf["train_generators"]) & set(FAMILIES[fam]))
    rows.append(r)
L = pd.DataFrame(rows)
L.to_csv(os.path.join(OUT, "verify_lofo.csv"), index=False)
V = pd.read_csv(os.path.join(S1, "reports", "lofo_verdicts_all_models.csv"))
V = V[V.features == "efficientnet_b0_2ep"].set_index("eval_family")
for r in rows:
    f = r["run"].replace("LOFO_", "")
    if f in V.index and abs(V.loc[f, "auc_vs_matched_real"] - r["heldout_auc_vs_matched_real"]) > 1e-9:
        res["issues"].append(f"LOFO {f}: report AUC {V.loc[f, 'auc_vs_matched_real']} != recomputed {r['heldout_auc_vs_matched_real']}")
    if not r["complete_2_epochs"]:
        res["issues"].append(f"{r['run']} not 2 complete epochs")
    if f in FAMILIES and not r["no_heldout_generator_in_train"]:
        res["issues"].append(f"{r['run']}: held-out generator present in train_generators")
res["lofo"] = rows

# --- J1 pre-registered criteria, recomputed from the jpeg_eval prediction files ---
J = os.path.join(S1, "data", "jpeg_eval")
g = {}
for m in ["baseline", "E0", "J1"]:
    for a in ["clean", "q50", "q35"]:
        d = pd.read_csv(os.path.join(J, f"{m}_{a}.csv"))
        g[(m, a)] = {"eer": eer(d.label_numeric, d.prob_fake),
                     "ffhq_fpr": float((d.loc[d.generator == "ffhq", "prob_fake"] >= .5).mean()),
                     "stylegan_tpr": float((d.loc[d.generator == "stylegan", "prob_fake"] >= .5).mean()), "n": len(d)}
crit = {}
for m in ["E0", "J1"]:
    crit[f"{m}_stylegan_drop_q35"] = g[(m, "clean")]["stylegan_tpr"] - g[(m, "q35")]["stylegan_tpr"]
    crit[f"{m}_ffhq_rise_q35"] = g[(m, "q35")]["ffhq_fpr"] - g[(m, "clean")]["ffhq_fpr"]
crit["ratio_ffhq_rise_J1_over_E0"] = crit["J1_ffhq_rise_q35"] / crit["E0_ffhq_rise_q35"]
crit["ratio_stylegan_drop_J1_over_E0"] = crit["J1_stylegan_drop_q35"] / crit["E0_stylegan_drop_q35"]
crit["C1_stylegan_drop_le_half_E0"] = bool(crit["ratio_stylegan_drop_J1_over_E0"] <= 0.5)
crit["C2_ffhq_rise_le_half_E0"] = bool(crit["ratio_ffhq_rise_J1_over_E0"] <= 0.5)
crit["C3_clean_eer_le_E0+0.3pp"] = bool(g[("J1", "clean")]["eer"] <= g[("E0", "clean")]["eer"] + 0.003)
dec = json.load(open(os.path.join(S1, "reports", "jpeg_decision.json")))
crit["C4_within_ffpp_from_report"] = dec["J1_within_ffpp_ge_E0_minus_0.02"]
crit["ALL_MET"] = all([crit["C1_stylegan_drop_le_half_E0"], crit["C2_ffhq_rise_le_half_E0"],
                       crit["C3_clean_eer_le_E0+0.3pp"], crit["C4_within_ffpp_from_report"]])
crit["matches_jpeg_decision_json"] = (crit["C1_stylegan_drop_le_half_E0"] == dec["J1_stylegan_drop_le_half_E0"]
                                      and crit["C2_ffhq_rise_le_half_E0"] == dec["J1_ffhq_rise_le_half_E0"]
                                      and crit["ALL_MET"] == dec["J1_ALL_CRITERIA_MET"])
crit["eer_table"] = {f"{m}_{a}": v["eer"] for (m, a), v in g.items()}
res["j1_criteria"] = crit
if not crit["C1_stylegan_drop_le_half_E0"]:
    res["issues"].append("Handoff said J1 missed the margin only on the FFHQ criterion; the StyleGAN-drop criterion is ALSO "
                         f"formally false (J1/E0 ratio {crit['ratio_stylegan_drop_J1_over_E0']:.2f}); the report called it 'uninformative'.")
res["issues"].append(f"FFHQ-rise ratio J1/E0 = {crit['ratio_ffhq_rise_J1_over_E0']:.4f} (reported as '58 %'; rounds to 59 %).")

# --- baseline test metrics recomputed from stored predictions ---
tp = pd.read_csv(os.path.join(BASE, "evaluation_reports", "predictions_test_20261016_091737.csv"))
js = json.load(open(os.path.join(BASE, "evaluation_reports", "evaluation_test_20261016_091737.json")))
y, p = tp.label_numeric.to_numpy(), tp.prob_fake.to_numpy()
cm = [int(v) for v in confusion_matrix(y, (p >= .5).astype(int)).ravel()]
res["baseline_test_recomputed"] = {"n": len(tp), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                                   "confusion@0.5": cm, "json_roc_auc": js["roc_auc"],
                                   "json_confusion@0.5": js["at_0.5"]["confusion [tn, fp, fn, tp]"]}
if cm != js["at_0.5"]["confusion [tn, fp, fn, tp]"] or abs(roc_auc_score(y, p) - js["roc_auc"]) > 1e-9:
    res["issues"].append("baseline test JSON does not match its predictions file")

# --- split / policy consistency ---
md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "dataset_split", "label_numeric"])
res["split_counts"] = {k: int(v) for k, v in md.dataset_split.value_counts().items()}
res["test_ids_match_predictions_file"] = set(md.loc[md.dataset_split == "test", "image_id"]) == set(tp.image_id)
j1cfg = json.load(open(os.path.join(S1, "experiments", "J1_jpeg_aug", "config.json")))
res["J1_n_train_equals_train_split"] = j1cfg["n_train"] == res["split_counts"]["train"]
res["J1_n_val_equals_val_split"] = j1cfg["n_val"] == res["split_counts"]["val"]
pol = json.load(open(os.path.join(S1, "reports", "app_policy.json")))
src = open(os.path.join(BASE, "inference.py"), encoding="utf-8").read()
res["inference_constants_match_policy"] = all(s in src for s in [f"REAL_BELOW = {pol['real_below']}",
                                                                 f"FAKE_AT_OR_ABOVE = {pol['fake_at_or_above']}",
                                                                 f"TEMPERATURE = {pol['temperature']}"])
res["app_policy_decides_on"] = "raw sigmoid score (inference.decide(raw)); temperature-scaled score is display-only"
json.dump(res, open(os.path.join(OUT, "verify_handoff.json"), "w"), indent=2, default=float)
print(L.round(4).to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k != "lofo"}, indent=1, default=float))
