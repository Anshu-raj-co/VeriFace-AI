"""Contact sheets of label-disagreement candidates for manual review (raw source images, unchanged files)."""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from PIL import Image
BASE = r"D:\PROJECT\VeriFace AI"; OUT = os.path.join(BASE, r"research_runs\stage1_20261016")
d = pd.read_csv(os.path.join(OUT, "reports", "label_model_disagreement_candidates.csv"))
d["vid"] = d.raw_path.str.extract(r"_([a-z]{10})_\d+_\d+\.jpg$")[0]
sheets = {
 "dfdc_real_flagged_val": d[(d.generator == "dfdc_real") & (d.pred_set == "val")].sort_values("prob_fake", ascending=False),
 "dfdc_fake_missed": d[(d.generator == "dfdc_fake")].sort_values("prob_fake"),
 "ffhq_flagged": d[(d.generator == "ffhq")].sort_values("prob_fake", ascending=False),
 "stylegan_missed": d[(d.generator == "stylegan")].sort_values("prob_fake"),
 "celebdf_synthesis_missed": d[(d.generator == "celebdf_synthesis")].sort_values("prob_fake"),
 "ffpp_original_flagged": d[(d.generator == "ffpp_original")].sort_values("prob_fake", ascending=False),
}
for name, s in sheets.items():
    s = s.head(12)
    if not len(s): continue
    fig, axs = plt.subplots(2, 6, figsize=(13, 5))
    for ax, (_, r) in zip(axs.ravel(), s.iterrows()):
        with Image.open(os.path.join(BASE, r.raw_path)) as im: ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r.image_id[-26:]}\n{r.label} p(F)={r.prob_fake:.3f} [{r.pred_set[:5]}]", fontsize=6); ax.axis("off")
    for ax in axs.ravel()[len(s):]: ax.axis("off")
    fig.suptitle(f"label review: {name} (raw source images)", fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", f"label_review_{name}.png")); plt.close(fig); print(name, len(s))
