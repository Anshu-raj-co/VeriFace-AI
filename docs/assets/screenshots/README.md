# Screenshot provenance

Captured on 2026-10-16. The app was the local `app.py`, running the selected checkpoint (`best_face_model.pth`, SHA-256
`c9a5eb5e…a1d0eb11`). It was driven by headless Chrome at 1360 px width on a separate local port. Scores are exactly
what the app displayed; nothing was edited except the privacy redaction described below and trimming of empty margins.

| file | input | verdict shown | matches recorded smoke test |
|---|---|---|---|
| `app_interface.png` | none | AWAITING IMAGE | — |
| `result_no_face_rejected.png` | locally generated gradient / checkerboard test pattern (no face) | CANNOT VERIFY — NO FACE | `noface_*` cases: refused |
| `result_likely_authentic.png` | genuine FFHQ photo from the validation split | LIKELY AUTHENTIC, uncalibrated 0.6 %, calibrated 3.5 % | `genuine_ffhq`: raw 0.0058, calibrated 0.035 |
| `result_inconclusive.png` | genuine FaceForensics++ frame from the validation split | INCONCLUSIVE, uncalibrated 45.2 % | `inconclusive_ffpp_original`: raw 0.4516 |
| `result_likely_manipulated.png` | StyleGAN-generated face from the validation split | LIKELY MANIPULATED, logit +18.47 | `manipulated_stylegan`: raw 1.0 |
| `gradcam_panel.png` | crop of the previous screenshot | — | — |

Smoke-test records: `research_runs/app_run_check_20261016/app_tests/app_smoke_tests.json`.

## Privacy and rights

* **Photos of real people are not shown.** In the FFHQ and FF++ screenshots, every image element is hidden and replaced
  by a placeholder before capture.
* **The StyleGAN face is blurred.** It depicts no real person, but it comes from a third-party dataset whose
  redistribution terms were not established. The large upload preview is hidden, and the two Grad-CAM tiles are
  strongly blurred. Grad-CAM here is a 7 × 7 map, so the blur keeps the attribution pattern readable while making the
  face unrecoverable.
* Uploads were renamed (`sample_photo_a.jpg`, …) so no dataset identifiers appear. No personal photographs, local
  paths, usernames or credentials appear in any screenshot.

To replace these with unredacted examples, use only images you own or that are licensed for redistribution, and that
show no identifiable real person without consent.
