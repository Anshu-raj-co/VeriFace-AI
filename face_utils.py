"""
VeriFace AI -- face detection utilities: a lightweight face-presence gate to
run BEFORE an image reaches the classifier, and the canonical square face
crop (`crop_face_square`) shared by preprocessing and inference.

Rationale: a binary REAL/FAKE authenticity score is only meaningful for face
images. Feeding it a photo of a dog, a landscape, a blank/solid-color image,
or a heavily cropped non-face region will still produce a confident-looking
sigmoid output, which reads to a user as a real verdict ("94% authentic")
about content the model was never evaluating in the first place. Gating on
face presence turns that into an explicit rejection instead of a silent
false signal.

Uses OpenCV's bundled Haar cascade (`opencv-python` ships the .xml file, so
no extra download/model weights are required). This is intentionally cheap
and approximate -- it is a **gate**, not a face-quality judgement:
  - False negatives (missed real faces) are the failure mode to watch for;
    tune `scale_factor` / `min_neighbors` / `min_size` down if you see
    legitimate portraits being rejected on your dataset.
  - It will not catch a well-cropped fake face-shaped image that isn't
    actually a face -- it only rejects the obviously-not-a-face case.

For higher accuracy at the cost of a heavier dependency, swap this out for
MTCNN (`facenet-pytorch`) or MediaPipe Face Detection -- both expose a
similar "did I find at least one face, and where" API, so `detect_face()`'s
return shape below is written to make that swap a drop-in change.
"""

import cv2
import numpy as np
from PIL import Image

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

if _face_cascade.empty():
    raise RuntimeError(
        f"Failed to load Haar cascade from {_CASCADE_PATH}. "
        "Check your opencv-python installation."
    )


def detect_face(
    image: Image.Image,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple = (60, 60),
) -> dict:
    """
    Run a Haar-cascade face check on a PIL image.

    Returns:
        {
            "has_face": bool,
            "num_faces": int,
            "largest_box": (x, y, w, h) or None,   # in original image pixel coords
        }
    """
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection under poor lighting/contrast

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    has_face = len(faces) > 0
    largest_box = None
    if has_face:
        # (x, y, w, h) with the largest area, in case of multiple detections
        largest_box = tuple(max(faces, key=lambda f: f[2] * f[3]))

    return {
        "has_face": has_face,
        "num_faces": int(len(faces)),
        "largest_box": largest_box,
    }


def crop_face_square(
    image: Image.Image,
    out_size: int = 256,
    expand: float = 1.8,
    detect_max_side: int = 512,
    min_size: tuple = (40, 40),
) -> tuple:
    """
    VeriFace AI canonical face crop. preprocess_data.py (training data) and
    app.py (inference) must both go through this function so the model sees
    identically framed inputs at train and test time.

    1. Detect the largest face on a copy downscaled to `detect_max_side`
       (Haar cost grows with resolution; detection doesn't need the detail).
    2. Square box centred on the face, side = `expand` x the detector box
       (Haar boxes are brow-to-chin; 1.8x takes in the whole head + a little
       background). No face -> centred square of side min(W, H).
    3. The side is capped at min(W, H) and the box is shifted back inside the
       image, so the crop never needs padding: no synthetic borders whose
       presence/size could differ between sources and become a shortcut.
    4. Resize to `out_size` (INTER_AREA when shrinking, INTER_CUBIC when
       enlarging).

    Returns (PIL RGB image of out_size x out_size, info dict).
    """
    rgb = image.convert("RGB")
    W, H = rgb.size
    scale = min(1.0, detect_max_side / max(W, H))
    det_img = rgb.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.BILINEAR) if scale < 1 else rgb
    det = detect_face(det_img, min_size=min_size)

    max_side = min(W, H)
    if det["has_face"]:
        x, y, w, h = (v / scale for v in det["largest_box"])
        cx, cy = x + w / 2, y + h / 2
        side = min(max_side, expand * max(w, h))
    else:
        cx, cy = W / 2, H / 2
        side = max_side

    side = int(round(side))
    left = int(round(min(max(cx - side / 2, 0), W - side)))
    top = int(round(min(max(cy - side / 2, 0), H - side)))
    crop = np.asarray(rgb)[top:top + side, left:left + side]

    interp = cv2.INTER_AREA if side >= out_size else cv2.INTER_CUBIC
    out = cv2.resize(crop, (out_size, out_size), interpolation=interp)

    info = {
        "face_detected": bool(det["has_face"]),
        "num_faces": det["num_faces"],
        "crop_box": (left, top, side, side),
        "crop_side": side,
        "upscaled": side < out_size,
    }
    return Image.fromarray(out), info


def validate_face_or_raise(image: Image.Image, **kwargs) -> dict:
    """
    Convenience wrapper for non-interactive call sites (batch scripts,
    APIs): raises ValueError instead of returning a rejectable dict, so a
    caller that forgets to check `has_face` fails loudly rather than
    silently scoring a non-face image.
    """
    result = detect_face(image, **kwargs)
    if not result["has_face"]:
        raise ValueError(
            "No face detected in the uploaded image. Authenticity scoring "
            "requires a visible face and will not run on this input."
        )
    return result


if __name__ == "__main__":
    # Quick smoke test on a blank (non-face) image -- should reject.
    blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = detect_face(blank)
    print("Blank image face check:", result)
    assert result["has_face"] is False, "Expected no face on a blank image."
    print("Face gate correctly rejects a non-face input.")
