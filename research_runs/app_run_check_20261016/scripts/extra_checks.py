"""
Additional application checks run before the manual browser session (validation images only, no test images).
Does not modify app.py / inference.py / the checkpoint. Output: app_tests/extra_checks.json

1. Label convention: checkpoint positive_class, metadata label <-> label_numeric mapping on all rows.
2. Preprocessing: inference.preprocess == face_utils.crop_face_square(256) -> CenterCrop(224) -> ToTensor ->
   ImageNet Normalize (recomputed independently); model_view equals the de-normalised tensor; crop geometry on raw
   validation files equals the crop_side stored by preprocess_data.py.
3. Calibration: calibrated score = sigmoid(logit / T), 0.5 at logit 0, monotone, verdict uses the raw score.
4. Grad-CAM (the app's own GradCAM class, extracted from app.py by AST): hooks the last Conv2d of the backbone,
   uses the same tensor as predict (same logit), 7x7 finite map in [0, 1], hooks removed, overlay size 224.
5. Unusual inputs: grayscale, RGBA, palette, 16-bit, tiny and very large images through inference.predict;
   AppTest of app.py with a corrupt (non-image) upload and with an RGBA PNG of a face.
"""
import ast
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

BASE = r"D:\PROJECT\VeriFace AI"
OUTD = os.path.join(BASE, r"research_runs\app_run_check_20261016\app_tests")
os.chdir(BASE); sys.path.insert(0, BASE)
import inference  # noqa: E402
from face_utils import crop_face_square  # noqa: E402

res, fail = {}, []


def check(section, name, ok, **info):
    res.setdefault(section, {})[name] = {"pass": bool(ok), **info}
    if not ok:
        fail.append(f"{section}:{name}")


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, meta = inference.load_checkpoint(inference.MODEL_PATH, dev)
ck = torch.load(inference.MODEL_PATH, map_location="cpu", weights_only=False)
res["checkpoint"] = {"sha256": hashlib.sha256(open(inference.MODEL_PATH, "rb").read()).hexdigest(), "meta": meta,
                     "keys": sorted(k for k in ck if k != "state_dict"),
                     "n_params": int(sum(p.numel() for p in model.parameters()))}

# ---------- 1. label convention ----------
md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"),
                 usecols=["image_id", "raw_path", "label", "label_numeric", "dataset_split", "generator", "crop_side", "face_detected"])
pairs = md.groupby(["label", "label_numeric"]).size().to_dict()
check("labels", "checkpoint_positive_class_FAKE", ck.get("positive_class") == "FAKE", value=ck.get("positive_class"))
check("labels", "metadata_REAL0_FAKE1", set(pairs) == {("REAL", 0), ("FAKE", 1)}, pairs={str(k): v for k, v in pairs.items()})
app_src = open(os.path.join(BASE, "app.py"), encoding="utf-8").read()
# meter: pointer = calibrated FAKE score, left legend AUTHENTIC -> gradient must start green (REAL) and end red (FAKE)
check("labels", "score_meter_orientation", "linear-gradient(to right, {ACCENT_REAL} 0%, {ACCENT_WARN} 50%, {ACCENT_FAKE} 100%)" in app_src
      and "marker_pct = max(0.0, min(100.0, score * 100))" in app_src and "<span>0% &mdash; AUTHENTIC</span>" in app_src)
res["labels"]["n_rows"] = int(len(md))
res["labels"]["split_counts"] = md.dataset_split.value_counts().to_dict()

# ---------- 2. preprocessing ----------
ref_tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
v = md[md.dataset_split == "val"]
sample = v.groupby("generator").head(3).sort_values("image_id")
geom_ok, tens_ok, view_ok, rows = 0, 0, 0, []
for r in sample.itertuples():
    with Image.open(os.path.join(BASE, r.raw_path)) as im:
        img = im.convert("RGB")
    tensor, view, crop, info = inference.preprocess(img)
    fc, _ = crop_face_square(img, out_size=256)
    t_ok = torch.equal(tensor, ref_tf(fc).unsqueeze(0)) and tuple(tensor.shape) == (1, 3, 224, 224)
    den = (tensor[0] * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None])
    v_ok = float((den - T.ToTensor()(view)).abs().max()) < 1e-5 and view.size == (224, 224) and crop.size == (256, 256)
    g_ok = int(info["crop_side"]) == int(r.crop_side) and bool(info["face_detected"]) == bool(r.face_detected)
    tens_ok += t_ok; view_ok += v_ok; geom_ok += g_ok
    rows.append({"image_id": r.image_id, "generator": r.generator, "crop_side": info["crop_side"],
                 "stored_crop_side": int(r.crop_side), "face_detected": info["face_detected"]})
n = len(sample)
check("preprocess", "tensor_equals_independent_eval_transform", tens_ok == n, n=n, n_ok=int(tens_ok))
check("preprocess", "model_view_is_denormalised_input", view_ok == n, n=n, n_ok=int(view_ok))
check("preprocess", "crop_geometry_matches_preprocess_data", geom_ok == n, n=n, n_ok=int(geom_ok),
      mismatches=[x for x in rows if x["crop_side"] != x["stored_crop_side"]])
check("preprocess", "eval_transform_has_no_resize", "Resize" not in repr(inference._EVAL_TRANSFORM), transform=repr(inference._EVAL_TRANSFORM))

# ---------- 3. calibration ----------
lg = np.linspace(-12, 12, 241)
cal = np.array([inference.calibrated_score(x) for x in lg])
check("calibration", "formula_sigmoid_logit_over_T", np.allclose(cal, 1 / (1 + np.exp(-lg / inference.TEMPERATURE))), T=inference.TEMPERATURE)
check("calibration", "half_at_zero_and_monotone", abs(inference.calibrated_score(0.0) - .5) < 1e-12 and np.all(np.diff(cal) > 0))
check("calibration", "verdict_bands_on_raw_score",
      inference.decide(0.2999) == "LIKELY AUTHENTIC" and inference.decide(0.30) == "INCONCLUSIVE"
      and inference.decide(0.8999) == "INCONCLUSIVE" and inference.decide(0.90) == "LIKELY MANIPULATED")
# raw 0.90 -> logit 2.197 -> calibrated 0.805; raw 0.30 -> calibrated 0.369: displayed % differs from the band edges
res["calibration"]["calibrated_at_band_edges"] = {"raw_0.30": inference.calibrated_score(np.log(.3 / .7)),
                                                  "raw_0.90": inference.calibrated_score(np.log(.9 / .1))}

# ---------- 4. Grad-CAM from app.py ----------
tree = ast.parse(open(os.path.join(BASE, "app.py"), encoding="utf-8").read())
keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"find_last_conv_layer", "GradCAM", "overlay_heatmap"}]
ns = {"nn": nn, "torch": torch, "np": np, "Image": Image}
exec(compile(ast.Module(body=keep, type_ignores=[]), "app.py(GradCAM extract)", "exec"), ns)
last_conv = ns["find_last_conv_layer"](model.backbone)
named = {m: n for n, m in model.backbone.named_modules()}
for r in sample.head(6).itertuples():
    with Image.open(os.path.join(BASE, r.raw_path)) as im:
        img = im.convert("RGB")
    p = inference.predict(model, img, dev)
    hooks_before = (len(last_conv._forward_hooks), len(last_conv._backward_hooks))
    gc = ns["GradCAM"](model)
    cam, logit = gc.generate(p["tensor"].to(dev))
    gc.remove()
    hooks_after = (len(last_conv._forward_hooks), len(last_conv._backward_hooks))
    ov = ns["overlay_heatmap"](p["model_view"], cam)
    ok = (cam.shape == (7, 7) and np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1 + 1e-6
          and abs(logit - p["logit"]) < 1e-3 and hooks_after == hooks_before and ov.size == (224, 224) and not model.training)
    check("gradcam", r.image_id, ok, target_layer=named[last_conv], target_out_channels=last_conv.out_channels,
          cam_shape=list(cam.shape), cam_max=float(cam.max()), logit_gradcam=logit, logit_predict=p["logit"],
          hooks_removed=hooks_after == hooks_before)

# ---------- 5. unusual inputs ----------
face_path = os.path.join(BASE, v[(v.generator == "ffhq") & v.face_detected].sort_values("image_id").raw_path.iloc[0])
face = Image.open(face_path).convert("RGB")
variants = {
    "grayscale_L": face.convert("L"), "rgba": face.convert("RGBA"), "palette_P": face.convert("P"),
    "16bit_I;16": Image.fromarray((np.asarray(face.convert("L")).astype(np.uint16) * 257)).convert("I;16"),
    "tiny_8x8": face.resize((8, 8)), "very_large_4096": face.resize((4096, 4096), Image.BICUBIC),
    "wide_panorama_2000x200": face.resize((2000, 200)),
}
for k, im in variants.items():
    try:
        o = inference.predict(model, im, dev)
        check("unusual_inputs", k, np.isfinite(o["logit"]), verdict=o["verdict"], raw=round(o["raw_score_fake"], 4),
              face_detected=o["crop_info"]["face_detected"], crop_side=o["crop_info"]["crop_side"])
    except Exception as e:  # noqa: BLE001
        check("unusual_inputs", k, False, error=f"{type(e).__name__}: {e}")

from streamlit.testing.v1 import AppTest  # noqa: E402

corrupt = os.path.join(OUTD, "_corrupt.jpg"); open(corrupt, "wb").write(os.urandom(4096))
rgba_png = os.path.join(OUTD, "_face_rgba.png"); face.convert("RGBA").save(rgba_png)


def apptest(tag, img_path):
    w = os.path.join(OUTD, f"_wrapper_{tag}.py")
    open(w, "w", encoding="utf-8").write("\n".join([
        "import io, runpy, streamlit as st", "st.cache_resource.clear()",
        "class _Up(io.BytesIO):", "    def __init__(self, p):",
        "        super().__init__(open(p, 'rb').read()); self.name = p.rsplit('\\\\', 1)[-1]; self.size = len(self.getvalue())",
        f"st.file_uploader = lambda *a, **k: _Up(r'{img_path}')",
        f"runpy.run_path(r'{os.path.join(BASE, 'app.py')}', run_name='__main__')"]) + "\n")
    try:
        t = AppTest.from_file(w, default_timeout=180).run()
        out = {"button_disabled": t.button[0].disabled if t.button else None}
        if t.button and not t.button[0].disabled:
            t.button[0].click().run()
        out.update(exceptions=[str(e.value) for e in t.exception], errors=[e.value for e in t.error],
                   banners=[m.value for m in t.markdown if 'class="verdict-banner' in m.value],
                   images=len(t.get("imgs")), metrics={m.label: m.value for m in t.metric})
    finally:
        os.remove(w)
    return out


a = apptest("corrupt", corrupt)
check("apptest", "corrupt_upload_refused", not a["exceptions"] and a["button_disabled"] and any("Could not read image" in e for e in a["errors"]), **a)
a = apptest("rgba_png", rgba_png)
check("apptest", "rgba_png_face_runs_with_gradcam", not a["exceptions"] and not a["button_disabled"] and len(a["banners"]) == 1
      and len(a["metrics"]) == 3, **a)
for f in (corrupt, rgba_png):
    os.remove(f)

res["failures"] = fail
res["all_pass"] = not fail
json.dump(res, open(os.path.join(OUTD, "extra_checks.json"), "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "preprocess"}, indent=1, default=str)[:6000])
print("FAILURES:", fail)
sys.exit(0 if not fail else 1)
