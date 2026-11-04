"""Shared blendshape vocabulary.

Names follow Apple ARKit / Google MediaPipe face blendshapes, so that in Phase 2
we can use MediaPipe's per-frame output on real video as training labels and
drive the Blender mouth with the model's predictions directly.

Animation files exchanged between the "brain" (rule-based or neural) and
Blender look like:

    {
      "fps": 30,
      "names": ["jawOpen", ...],          # subset/order of MOUTH_SHAPES
      "frames": [[0.0, 0.1, ...], ...]    # one row per frame, values in [0, 1]
    }
"""

import json

FPS = 30

MOUTH_SHAPES = [
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
]


def save_animation(path, frames, names=MOUTH_SHAPES, fps=FPS):
    rows = [[round(float(v), 4) for v in row] for row in frames]
    with open(path, "w") as f:
        json.dump({"fps": fps, "names": list(names), "frames": rows}, f)


def load_animation(path):
    with open(path) as f:
        data = json.load(f)
    return data["fps"], data["names"], data["frames"]
