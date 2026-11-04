"""A simple 2D face drawn from blendshape weights, for a live preview window.

Not the avatar - just enough to see at a glance what the model is sending
(jaw, lip shapes, smiles, blinks, brows) without Unreal running.
"""

import cv2
import numpy as np

INK = (40, 40, 40)
SKIN = (205, 222, 240)
LIP = (90, 90, 190)
MOUTH = (40, 30, 60)


def draw_face(img, w, cx, cy, size, roll=0.0):
    """img: BGR canvas; w: dict name -> weight; size: face half-width in px."""
    g = lambda n: float(w.get(n, 0.0))  # noqa: E731
    layer = np.zeros_like(img)
    s = size
    cv2.ellipse(layer, (cx, cy), (s, int(s * 1.3)), 0, 0, 360, SKIN, -1)

    for side, sign in (("Left", -1), ("Right", 1)):
        # MediaPipe "Left" = the person's left = viewer's right (mirror view)
        ex, ey = cx - sign * int(0.42 * s), cy - int(0.25 * s)
        open_ = (1 - g(f"eyeBlink{side}")) * (1 + 0.6 * g(f"eyeWide{side}") - 0.4 * g(f"eyeSquint{side}"))
        eh = max(1, int(0.13 * s * open_))
        cv2.ellipse(layer, (ex, ey), (int(0.2 * s), eh), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(layer, (ex, ey), (int(0.2 * s), eh), 0, 0, 360, INK, 2)
        if eh > 3:
            look_x = g(f"eyeLookOut{side}") - g(f"eyeLookIn{side}")
            look_y = g(f"eyeLookDown{side}") - g(f"eyeLookUp{side}")
            cv2.circle(layer, (ex - sign * int(0.1 * s * look_x), ey + int(0.06 * s * look_y)),
                       max(2, min(eh, int(0.08 * s))), INK, -1)
        lift = 0.18 * s * (g("browInnerUp") + g(f"browOuterUp{side}")) - 0.12 * s * g(f"browDown{side}")
        inner = (cx - sign * int(0.2 * s), int(ey - 0.25 * s - lift - 0.12 * s * g("browInnerUp")))
        outer = (cx - sign * int(0.62 * s), int(ey - 0.22 * s - lift))
        cv2.line(layer, inner, outer, INK, max(2, s // 18))

    # mouth: corners + upper/lower lip curves
    mw = 0.32 * s * (1 + 0.35 * (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
                     + 0.25 * (g("mouthStretchLeft") + g("mouthStretchRight")) / 2
                     - 0.45 * g("mouthPucker") - 0.3 * g("mouthFunnel"))
    my = cy + 0.62 * s
    shift = 0.15 * s * (g("mouthRight") - g("mouthLeft"))
    opening = max(0.0, g("jawOpen") - 0.7 * g("mouthClose")) + 0.5 * g("mouthFunnel")
    up = 0.06 * s + 0.1 * s * (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2
    down = 0.06 * s + 0.5 * s * opening + 0.08 * s * (g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 2
    press = 1 - 0.7 * (g("mouthPressLeft") + g("mouthPressRight")) / 2
    xs = np.linspace(-1, 1, 21)
    corner_l = 0.12 * s * (g("mouthSmileRight") - g("mouthFrownRight"))  # viewer's left
    corner_r = 0.12 * s * (g("mouthSmileLeft") - g("mouthFrownLeft"))
    corner = np.where(xs < 0, corner_l, corner_r) * np.abs(xs) ** 2
    bulge = 1 - xs ** 2
    top = np.stack([cx + shift + xs * mw, my - corner - up * bulge * press], 1)
    bot = np.stack([cx + shift + xs * mw, my - corner + down * bulge * press], 1)
    poly = np.concatenate([top, bot[::-1]]).astype(np.int32)
    cv2.fillPoly(layer, [poly], MOUTH)
    cv2.polylines(layer, [poly], True, LIP, max(2, s // 14))

    if abs(roll) > 1e-3:
        M = cv2.getRotationMatrix2D((cx, cy), np.degrees(roll), 1.0)
        layer = cv2.warpAffine(layer, M, (img.shape[1], img.shape[0]))
    mask = layer.any(axis=2)
    img[mask] = layer[mask]
    return img


def bars(img, w, names, x, y, width=160, row=14):
    for k, n in enumerate(names):
        v = float(w.get(n, 0.0))
        yy = y + k * row
        cv2.putText(img, n, (x, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.rectangle(img, (x + 130, yy + 2), (x + 130 + width, yy + row - 3), (70, 70, 70), -1)
        cv2.rectangle(img, (x + 130, yy + 2), (x + 130 + int(width * v), yy + row - 3), (80, 200, 255), -1)
    return img
