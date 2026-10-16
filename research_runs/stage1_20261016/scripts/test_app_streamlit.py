"""Streamlit AppTest of the real app.py: loads, renders, model available, no exceptions; then drives the
analysis path by injecting an image through a tiny wrapper script that patches st.file_uploader."""
import json, os, sys
from streamlit.testing.v1 import AppTest
BASE = r"D:\PROJECT\VeriFace AI"; OUT = os.path.join(BASE, r"research_runs\stage1_20261016")
os.chdir(BASE); sys.path.insert(0, BASE)
res = {}
at = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=180).run()
res["initial_exceptions"] = [str(e.value) for e in at.exception]
res["initial_errors"] = [e.value for e in at.error]
res["run_button_disabled_without_upload"] = at.button[0].disabled
res["banners"] = [m.value for m in at.markdown if "verdict-banner" in m.value]

# Wrapper: same app, but st.file_uploader returns a validation image (raw source file).
def run_with(img_path, tag):
    wrapper = os.path.join(OUT, "app_tests", f"_wrapper_{tag}.py")
    with open(wrapper, "w", encoding="utf-8") as fh:
        fh.write(f'''
import io, runpy, streamlit as st
class _Up(io.BytesIO):
    def __init__(self, p):
        super().__init__(open(p, "rb").read()); self.name = p.rsplit("\\\\", 1)[-1]; self.size = len(self.getvalue())
_p = r"{img_path}"
st.file_uploader = lambda *a, **k: _Up(_p)
runpy.run_path(r"{os.path.join(BASE, 'app.py')}", run_name="__main__")
''')
    t = AppTest.from_file(wrapper, default_timeout=180).run()
    out = {"exceptions_before_click": [str(e.value) for e in t.exception],
           "button_disabled": t.button[0].disabled if t.button else None}
    if t.button and not t.button[0].disabled:
        t.button[0].click().run()
    out["exceptions_after_click"] = [str(e.value) for e in t.exception]
    out["errors"] = [e.value for e in t.error]
    out["banners"] = [m.value for m in t.markdown if "verdict-banner" in m.value]
    out["metrics"] = {m.label: m.value for m in t.metric}
    out["warnings"] = [w.value for w in t.warning]
    out["images_rendered"] = len(t.get("imgs")) if hasattr(t, "get") else None
    os.remove(wrapper)
    return out

import pandas as pd
md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"))
v = md[md.dataset_split == "val"]
picks = {"ffhq_real": v[v.generator == "ffhq"].raw_path.iloc[0], "stylegan_fake": v[v.generator == "stylegan"].raw_path.iloc[0],
         "ffpp_original_real": v[v.generator == "ffpp_original"].raw_path.iloc[0], "stable_diffusion_fake": v[v.generator == "stable_diffusion"].raw_path.iloc[0]}
for tag, rp in picks.items():
    res[tag] = {"file": rp, **run_with(os.path.join(BASE, rp.replace("/", "\\")), tag)}
json.dump(res, open(os.path.join(OUT, "app_tests", "streamlit_apptest.json"), "w"), indent=2, default=str)
print(json.dumps(res, indent=1, default=str))
