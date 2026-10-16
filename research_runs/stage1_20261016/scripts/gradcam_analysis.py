"""
Grad-CAM for the baseline on VALIDATION images (S1 section 6). Same Grad-CAM as app.py (last Conv2d of the
backbone, gradient-weighted activations, evidence toward FAKE), applied to the exact 224 px model input.

Quantitative check: share of positive CAM mass inside the central 50 % box (the face region of the 1.8x face
crop) vs the border (hair / background / letterbox). Groups: FF++ originals scored FAKE (false positives) vs
scored REAL, NeuralTextures detected vs missed, FFHQ, StyleGAN, DFDC real flagged. Descriptive, not causal.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

BASE = r"D:\PROJECT\VeriFace AI"
S1 = os.path.join(BASE, r"research_runs\stage1_20261016")
sys.path.insert(0, BASE)
import inference  # noqa: E402
from train import get_eval_transforms  # noqa: E402


class GradCAM:
    def __init__(self, model):
        self.model = model
        last = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)][-1]
        self.h1 = last.register_forward_hook(lambda m, i, o: setattr(self, "act", o.detach()))
        self.h2 = last.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            lg = self.model(x)
            lg.backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.act).sum(1)).squeeze().cpu().numpy()
        return cam, float(lg.item())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = inference.load_checkpoint(inference.MODEL_PATH, dev)
    gc = GradCAM(model)
    tf = get_eval_transforms()
    md = pd.read_csv(os.path.join(BASE, "cleaned_metadata.csv"), usecols=["image_id", "local_image_path", "generator"])
    vp = pd.read_csv(os.path.join(BASE, r"research_runs\generalization_audit_20261016\data\preds_baseline_val.csv"))[["image_id", "prob_fake"]]
    v = md.merge(vp, on="image_id")
    groups = {
        "ffpp_original_FP": v[(v.generator == "ffpp_original") & (v.prob_fake >= .5)],
        "ffpp_original_TN": v[(v.generator == "ffpp_original") & (v.prob_fake < .5)],
        "NeuralTextures_TP": v[(v.generator == "ffpp_NeuralTextures") & (v.prob_fake >= .5)],
        "NeuralTextures_FN": v[(v.generator == "ffpp_NeuralTextures") & (v.prob_fake < .5)],
        "dfdc_real_FP": v[(v.generator == "dfdc_real") & (v.prob_fake >= .5)],
        "ffhq_TN": v[(v.generator == "ffhq") & (v.prob_fake < .5)],
        "stylegan_TP": v[(v.generator == "stylegan") & (v.prob_fake >= .5)],
        "stable_diffusion_TP": v[(v.generator == "stable_diffusion") & (v.prob_fake >= .5)],
    }
    rows, examples = [], {}
    for name, g in groups.items():
        g = g.sample(min(120, len(g)), random_state=0)
        for r in g.itertuples():
            with Image.open(os.path.join(BASE, r.local_image_path)) as im:
                img = im.convert("RGB")
            x = tf(img).unsqueeze(0).to(dev)
            cam7, lg = gc(x)
            # 7x7 map -> 224 px (bilinear) so the central box is exactly 112x112 = 25 % of the area
            cam = torch.nn.functional.interpolate(torch.tensor(cam7)[None, None], size=(224, 224), mode="bilinear",
                                                  align_corners=False)[0, 0].numpy()
            tot = cam.sum() + 1e-12
            h = 224; a, b = 56, 168
            rows.append({"group": name, "image_id": r.image_id, "logit": lg, "cam_center_share": float(cam[a:b, a:b].sum() / tot),
                         "cam_top_rows_share": float(cam[:a].sum() / tot), "cam_bottom_rows_share": float(cam[b:].sum() / tot)})
            if len(examples.setdefault(name, [])) < 6:
                w, hh = img.size
                view = img.crop(((w - 224) // 2, (hh - 224) // 2, (w - 224) // 2 + 224, (hh - 224) // 2 + 224))
                examples[name].append((view, cam / (cam.max() + 1e-12), r.prob_fake))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(S1, "reports", "gradcam_center_share_val.csv"), index=False)
    s = d.groupby("group").agg(n=("image_id", "size"), center_share_mean=("cam_center_share", "mean"),
                               center_share_median=("cam_center_share", "median"), top_rows=("cam_top_rows_share", "mean"),
                               bottom_rows=("cam_bottom_rows_share", "mean"))
    s.to_csv(os.path.join(S1, "reports", "gradcam_center_share_summary_val.csv"))
    print(s.round(3).to_string())
    print("uniform-attention reference: centre box = 25 % of area", flush=True)
    cmap = matplotlib.colormaps["jet"]
    fig, axs = plt.subplots(len(examples), 6, figsize=(12, 2.1 * len(examples)))
    for i, (name, ex) in enumerate(examples.items()):
        for j in range(6):
            ax = axs[i, j]; ax.axis("off")
            if j < len(ex):
                view, cam, p = ex[j]
                c = cam
                ax.imshow(np.asarray(view)); ax.imshow(cmap(c)[..., :3], alpha=0.45)
                ax.set_title(f"{name}\nP(FAKE)={p:.2f}", fontsize=6)
    fig.suptitle("Baseline Grad-CAM on VALIDATION images (evidence toward FAKE; 224 px model input)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(S1, "plots", "gradcam_val_examples.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
