"""
Stage-1 tests for the fixed app inference path (inference.py) -- validation images only.

1. Parity: inference.predict on RAW source images (the kind of file a user uploads) vs. the stored
   baseline prediction for the same image's processed crop (the evaluation pipeline). The processed file
   was additionally JPEG re-encoded (q 60-100), so small differences are expected; large ones would mean
   the app pipeline diverges from evaluation.
2. Old vs new: the pre-fix app transform (Resize((224,224)) of the whole image) on the same raw files.
3. Edge cases: grayscale, RGBA PNG, tiny image, very large image, blank (no face) image.
4. Label convention: a checkpoint copy with positive_class='REAL' must be refused.
"""
import io
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

BASE = r"D:\PROJECT\VeriFace AI"
OUT = os.path.join(BASE, r"research_runs\stage1_20261016")
sys.path.insert(0, BASE)
import inference  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, meta = inference.load_checkpoint(inference.MODEL_PATH, dev)
print("[test] loaded", meta, flush=True)
results = {"meta": meta}

# 1 + 2: parity on validation raw images
md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"))
vp = pd.read_csv(os.path.join(BASE, r"research_runs\generalization_audit_20261016\data\preds_baseline_val.csv"))[["image_id", "prob_fake"]]
v = md[md.dataset_split == "val"].merge(vp, on="image_id")
sample = v.groupby("generator").sample(15, random_state=0).reset_index(drop=True)
old_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
rows, t0 = [], time.time()
for r in sample.itertuples():
    with Image.open(os.path.join(BASE, r.raw_path)) as im:
        img = im.convert("RGB")
    out = inference.predict(model, img, dev)
    with torch.no_grad():
        old = float(torch.sigmoid(model(old_tf(img).unsqueeze(0).to(dev))).item())
    rows.append({"image_id": r.image_id, "generator": r.generator, "label": r.label_numeric,
                 "eval_pipeline_score": r.prob_fake, "new_app_score": out["raw_score_fake"], "old_app_score": old,
                 "new_verdict": out["verdict"], "face_detected": out["crop_info"]["face_detected"],
                 "crop_side": out["crop_info"]["crop_side"], "stored_crop_side": r.crop_side})
p = pd.DataFrame(rows)
p.to_csv(os.path.join(OUT, "app_tests", "parity_val_sample.csv"), index=False)
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
for col in ["new_app_score", "old_app_score"]:
    results[col] = {
        "spearman_vs_eval_pipeline": float(p[[col, "eval_pipeline_score"]].corr("spearman").iloc[0, 1]),
        "median_abs_logit_diff_vs_eval": float(np.median(np.abs(lg(p[col]) - lg(p.eval_pipeline_score)))),
        "decision_agreement@0.5_vs_eval": float(((p[col] >= .5) == (p.eval_pipeline_score >= .5)).mean()),
        "accuracy@0.5": float(((p[col] >= .5) == p.label).mean()),
        "auc": float(roc_auc_score(p.label, p[col])),
    }
results["eval_pipeline_on_sample"] = {"accuracy@0.5": float(((p.eval_pipeline_score >= .5) == p.label).mean()),
                                      "auc": float(roc_auc_score(p.label, p.eval_pipeline_score))}
results["crop_side_match_rate"] = float((p.crop_side == p.stored_crop_side).mean())
results["new_verdict_counts"] = p.groupby(["label", "new_verdict"]).size().unstack(fill_value=0).to_dict()
results["n_images"] = len(p)
results["seconds"] = round(time.time() - t0, 1)
results["by_generator"] = p.groupby("generator").agg(
    eval=("eval_pipeline_score", "mean"), new=("new_app_score", "mean"), old=("old_app_score", "mean")).round(3).to_dict(orient="index")

# 3: edge cases
src = Image.open(os.path.join(BASE, sample.raw_path.iloc[0])).convert("RGB")
cases = {
    "grayscale_L": src.convert("L"),
    "rgba_png": (lambda b: (src.convert("RGBA").save(b, "PNG"), Image.open(io.BytesIO(b.getvalue())))[1])(io.BytesIO()),
    "tiny_64px": src.resize((64, 64)),
    "large_3000px": src.resize((3000, 3000)),
    "wide_1920x1080": src.resize((1920, 1080)),
    "blank_gray": Image.new("RGB", (512, 512), (128, 128, 128)),
}
edge = {}
for k, im in cases.items():
    try:
        o = inference.predict(model, im, dev)
        edge[k] = {"ok": True, "tensor_shape": list(o["tensor"].shape), "score": round(o["raw_score_fake"], 4),
                   "verdict": o["verdict"], "face_detected": o["crop_info"]["face_detected"]}
    except Exception as e:  # noqa: BLE001
        edge[k] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
results["edge_cases"] = edge

# 4: label convention guard
ck = torch.load(inference.MODEL_PATH, map_location="cpu", weights_only=False)
ck["positive_class"] = "REAL"
with tempfile.TemporaryDirectory() as td:
    bad = os.path.join(td, "bad.pth"); torch.save(ck, bad)
    try:
        inference.load_checkpoint(bad, dev); results["refuses_REAL_positive_checkpoint"] = False
    except ValueError as e:
        results["refuses_REAL_positive_checkpoint"] = True; results["refusal_message"] = str(e)
results["policy"] = {"REAL_BELOW": inference.REAL_BELOW, "FAKE_AT_OR_ABOVE": inference.FAKE_AT_OR_ABOVE, "T": inference.TEMPERATURE}
json.dump(results, open(os.path.join(OUT, "app_tests", "inference_tests.json"), "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in results.items() if k != "by_generator"}, indent=1, default=str))
print(pd.DataFrame(results["by_generator"]).T.to_string())
