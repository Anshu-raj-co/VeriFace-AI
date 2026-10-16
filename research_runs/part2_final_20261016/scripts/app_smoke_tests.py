"""
Part 2 application verification (validation images only -- no test images are used here).
1. Static: app.py / inference.py load the selected checkpoint (path + SHA-256), use the eval preprocessing, and the
   policy constants equal reports/app_policy.json; score wording is not presented as a guaranteed probability.
2. inference.predict on: confident genuine, confident manipulated, an inconclusive case, no-face images.
3. Incompatible checkpoints: REAL-positive, bare state_dict, wrong architecture, missing file -> refused.
4. Streamlit AppTest of the real app.py driven with the same images, plus a REAL-positive checkpoint injected.
Outputs: app_tests/app_smoke_tests.json
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

BASE = r"D:\PROJECT\VeriFace AI"
P2 = os.path.join(BASE, r"research_runs\part2_final_20261016")
OUTD = os.path.join(P2, "app_tests")
SELECTED_SHA = "c9a5eb5ea0d082ad99b2039fb63290f6c94c28372fc2ee88c75c2837a1d0eb11"
os.chdir(BASE); sys.path.insert(0, BASE)
import inference  # noqa: E402

res = {"static": {}, "inference": {}, "checkpoints": {}, "apptest": {}}
fail = []

# ---------- 1. static checks ----------
app_src = open(os.path.join(BASE, "app.py"), encoding="utf-8").read()
pol = json.load(open(os.path.join(BASE, r"research_runs\stage1_20261016\reports\app_policy.json")))
sha = hashlib.sha256(open(inference.MODEL_PATH, "rb").read()).hexdigest()
st_ = res["static"]
st_["inference_MODEL_PATH"] = inference.MODEL_PATH
st_["model_sha256"] = sha
st_["sha_matches_selected"] = sha == SELECTED_SHA
st_["app_MODEL_PATH_is_best_face_model"] = 'MODEL_PATH = os.path.join(BASE_DIR, "best_face_model.pth")' in app_src
st_["app_uses_inference_preprocess_and_load"] = all(s in app_src for s in ["load_checkpoint(MODEL_PATH", "preprocess(", "from inference import"])
st_["app_has_no_whole_image_resize"] = "Resize((224, 224))" not in app_src and "Resize((224,224))" not in app_src
st_["policy_matches_app_policy_json"] = (inference.REAL_BELOW == pol["real_below"] and inference.FAKE_AT_OR_ABOVE == pol["fake_at_or_above"]
                                        and inference.TEMPERATURE == pol["temperature"])
st_["not_a_guaranteed_probability_wording"] = "not a guaranteed probability" in app_src
st_["no_confidence_percent_wording_in_ui_text"] = "Confidence" not in app_src
for k, v in st_.items():
    if v is False:
        fail.append(f"static:{k}")

# ---------- 2. inference on validation images ----------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, meta = inference.load_checkpoint(inference.MODEL_PATH, dev)
res["meta"] = meta
md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "raw_path", "generator", "label_numeric", "dataset_split"])
vp = pd.read_csv(os.path.join(BASE, r"research_runs\generalization_audit_20261016\data\preds_baseline_val.csv"))[["image_id", "prob_fake"]]
v = md[md.dataset_split == "val"].merge(vp, on="image_id")
assert set(v.dataset_split) == {"val"}


def run(path):
    with Image.open(path) as im:
        img = im.convert("RGB")
    o = inference.predict(model, img, dev)
    return {"file": os.path.relpath(path, BASE), "raw_score": round(o["raw_score_fake"], 4),
            "calibrated_score": round(o["calibrated_score_fake"], 4), "verdict": o["verdict"],
            "face_detected": o["crop_info"]["face_detected"], "num_faces": o["crop_info"]["num_faces"]}


picks = {}
for tag, gen, want in [("genuine_ffhq", "ffhq", "LIKELY AUTHENTIC"), ("manipulated_stylegan", "stylegan", "LIKELY MANIPULATED"),
                       ("manipulated_stable_diffusion", "stable_diffusion", "LIKELY MANIPULATED"),
                       ("genuine_celebdf_real", "celebdf_real", "LIKELY AUTHENTIC")]:
    # run 1 took the first image per generator; its first FFHQ image is genuinely borderline (eval pipeline 0.49,
    # app 0.57 -> INCONCLUSIVE, kept in app_smoke_tests_run1_naive_pick.json). Run 2 picks images whose stored
    # evaluation-pipeline score is clearly on the expected side (< 0.10 genuine, > 0.95 fake).
    clear = v[(v.generator == gen) & ((v.prob_fake < .10) if want == "LIKELY AUTHENTIC" else (v.prob_fake > .95))]
    for r in clear.sort_values("image_id").itertuples():
        o = run(os.path.join(BASE, r.raw_path))
        if o["face_detected"]:
            picks[tag] = (os.path.join(BASE, r.raw_path), want, o); break
# inconclusive: first FF++ original (sorted id) with stored eval score in the band whose app verdict is INCONCLUSIVE
band = v[(v.generator == "ffpp_original") & (v.prob_fake >= .35) & (v.prob_fake <= .85)].sort_values("image_id")
for r in band.itertuples():
    o = run(os.path.join(BASE, r.raw_path))
    if o["verdict"] == "INCONCLUSIVE" and o["face_detected"]:
        picks["inconclusive_ffpp_original"] = (os.path.join(BASE, r.raw_path), "INCONCLUSIVE", o); break
blank = os.path.join(OUTD, "_noface_blank.png"); Image.new("RGB", (512, 512), (128, 128, 128)).save(blank)
rng = np.random.default_rng(0)
noise = os.path.join(OUTD, "_noface_noise.png"); Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(noise)
for tag, p_ in [("noface_blank", blank), ("noface_noise", noise)]:
    o = run(p_); picks[tag] = (p_, "NO FACE (app refuses)", o)
for tag, (p_, want, o) in picks.items():
    ok = (not o["face_detected"]) if tag.startswith("noface") else (o["verdict"] == want)
    res["inference"][tag] = {**o, "expected": want, "pass": ok}
    if not ok:
        fail.append(f"inference:{tag}")

# ---------- 3. incompatible checkpoints ----------
ck = torch.load(inference.MODEL_PATH, map_location="cpu", weights_only=False)
with tempfile.TemporaryDirectory() as td:
    bad_real = os.path.join(td, "real_positive.pth"); c2 = dict(ck); c2["positive_class"] = "REAL"; torch.save(c2, bad_real)
    bare = os.path.join(td, "bare_state_dict.pth"); torch.save(ck["state_dict"], bare)
    wrong = os.path.join(td, "wrong_backbone.pth"); c3 = dict(ck); c3["backbone"] = "resnet18"; torch.save(c3, wrong)
    cases = {"REAL_positive": bad_real, "bare_state_dict": bare, "wrong_architecture": wrong,
             "missing_file": os.path.join(td, "does_not_exist.pth")}
    for k, pth in cases.items():
        try:
            inference.load_checkpoint(pth, torch.device("cpu"))
            res["checkpoints"][k] = {"refused": False}; fail.append(f"checkpoint:{k}")
        except Exception as e:  # noqa: BLE001
            res["checkpoints"][k] = {"refused": True, "error": f"{type(e).__name__}: {str(e)[:160]}"}

    # ---------- 4. Streamlit AppTest of the real app.py ----------
    from streamlit.testing.v1 import AppTest

    def apptest(tag, img_path=None, ckpt=None):
        w = os.path.join(OUTD, f"_wrapper_{tag}.py")
        lines = ["import io, runpy, streamlit as st", "st.cache_resource.clear()"]
        if ckpt:
            lines += ["import inference", "_orig = inference.load_checkpoint",
                      f"inference.load_checkpoint = lambda p, d=None: _orig(r'{ckpt}', d)"]
        if img_path:
            lines += ["class _Up(io.BytesIO):", "    def __init__(self, p):",
                      "        super().__init__(open(p, 'rb').read()); self.name = p.rsplit('\\\\', 1)[-1]; self.size = len(self.getvalue())",
                      f"st.file_uploader = lambda *a, **k: _Up(r'{img_path}')"]
        lines += [f"runpy.run_path(r'{os.path.join(BASE, 'app.py')}', run_name='__main__')"]
        open(w, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        try:
            t = AppTest.from_file(w, default_timeout=180).run()
            out = {"exceptions_before_click": [str(e.value) for e in t.exception],
                   "button_disabled": t.button[0].disabled if t.button else None}
            if t.button and not t.button[0].disabled:
                t.button[0].click().run()
            out["exceptions_after_click"] = [str(e.value) for e in t.exception]
            out["errors"] = [e.value for e in t.error]
            out["warnings"] = [x.value for x in t.warning]
            out["banners"] = [m.value for m in t.markdown if 'class="verdict-banner' in m.value]
            out["score_captions"] = [m.value for m in t.markdown if "guaranteed probability" in m.value]
            out["metrics"] = {m.label: m.value for m in t.metric}
        finally:
            os.remove(w)
        return out

    for tag, (p_, want, o) in picks.items():
        a = apptest(tag, p_)
        if tag.startswith("noface"):
            ok = (not a["exceptions_after_click"]) and a["button_disabled"] and any("No human face" in e for e in a["errors"])
        else:
            ok = (not a["exceptions_after_click"]) and any(want in b for b in a["banners"]) and len(a["score_captions"]) > 0
        res["apptest"][tag] = {**a, "expected": want, "pass": bool(ok)}
        if not ok:
            fail.append(f"apptest:{tag}")
    a = apptest("REAL_positive_ckpt", picks["genuine_ffhq"][0], ckpt=bad_real)
    # the app shows a 'MODEL UNAVAILABLE' placeholder banner; what must not appear is a verdict
    ok = (any("Model Unavailable" in e and "positive_class" in e for e in a["errors"])
          and not any(w in b for b in a["banners"] for w in ("AUTHENTIC", "MANIPULATED", "INCONCLUSIVE")))
    res["apptest"]["REAL_positive_ckpt"] = {**a, "expected": "Model Unavailable error, no verdict", "pass": bool(ok)}
    if not ok:
        fail.append("apptest:REAL_positive_ckpt")

for f in (blank, noise):
    os.remove(f)
res["failures"] = fail
res["all_pass"] = not fail
json.dump(res, open(os.path.join(OUTD, "app_smoke_tests.json"), "w"), indent=2, default=str)
print(json.dumps(res, indent=1, default=str))
sys.exit(0 if not fail else 1)
