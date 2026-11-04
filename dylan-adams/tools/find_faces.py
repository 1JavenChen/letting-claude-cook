"""Scan a video for faces and report where they are big enough to train on.

Samples the video twice a second, finds every face on the full frame, and
records its box and mouth width in pixels (distance between mouth corners).
Small facecams (~15 px mouths) give labels that don't follow speech; this
finds the stretches - e.g. fullscreen-webcam endings - with big faces.

    python tools/find_faces.py data/raw/video.mp4 [--sheet out/video_faces.jpg]

Writes <video>.faces.json next to the processed data with every detection,
and prints the time ranges where faces are large.
"""

import argparse
import json
import os
import sys

import av
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
MOUTH_L, MOUTH_R = 61, 291  # MediaPipe face mesh mouth-corner landmarks


def tiles(w, h):
    """Whole frame plus overlapping half-size tiles (3 across x 3 down)."""
    tw, th = w // 2, h // 2
    out = [(0, 0, w, h)]
    for y in (0, h // 4, h // 2):
        for x in (0, w // 4, w // 2):
            out.append((x, y, x + tw, y + th))
    return out


def _same_face(a, b):
    """Two boxes are the same face if their centres are within half a face width."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 0.5 * max(a[2] - a[0], b[2] - b[0])


def background_brightness(img, box):
    """Mean brightness of the area around a face (excluding the face itself),
    used to tell people apart by their rooms."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    fw, fh = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, x0 - fw)), int(max(0, y0 - 0.5 * fh))
    X1, Y1 = int(min(w, x1 + fw)), int(min(h, y1 + 0.5 * fh))
    region = img[Y0:Y1, X0:X1].astype(np.float32).mean(axis=2)
    mask = np.ones(region.shape, bool)
    mask[int(y0) - Y0:int(y1) - Y0, int(x0) - X0:int(x1) - X0] = False
    return float(region[mask].mean()) if mask.any() else float(region.mean())


def scan(video_path, every_s=0.5):
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=4,
        min_face_detection_confidence=0.5)
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    samples, next_t = [], 0.0
    with vision.FaceLandmarker.create_from_options(options) as lm:
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < next_t:
                continue
            next_t = t + every_s
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            # MediaPipe's detector expects faces to fill a good part of the
            # image, so scan overlapping tiles rather than the whole frame.
            faces = []
            for tx0, ty0, tx1, ty1 in tiles(w, h):
                tile = np.ascontiguousarray(img[ty0:ty1, tx0:tx1])
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile))
                for pts in res.face_landmarks:
                    xs = np.array([p.x for p in pts]) * (tx1 - tx0) + tx0
                    ys = np.array([p.y for p in pts]) * (ty1 - ty0) + ty0
                    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                    if any(_same_face(box, f["box_px"]) for f in faces):
                        continue
                    mouth = float(np.hypot(xs[MOUTH_L] - xs[MOUTH_R], ys[MOUTH_L] - ys[MOUTH_R]))
                    faces.append({"box_px": box, "mouth_px": mouth,
                                  "bg": background_brightness(img, box)})
            for f in faces:
                b = f.pop("box_px")
                f["box"] = [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
            samples.append({"t": t, "faces": faces})
    container.close()
    return samples, (w, h)


def large_ranges(samples, min_mouth, min_len_s=10):
    """Time ranges where at least one face has a mouth >= min_mouth px."""
    good = np.array([any(f["mouth_px"] >= min_mouth for f in s["faces"]) for s in samples])
    times = np.array([s["t"] for s in samples])
    ranges, start = [], None
    for t, g in zip(times, good):
        if g and start is None:
            start = t
        if not g and start is not None:
            if t - start >= min_len_s:
                ranges.append((start, t))
            start = None
    if start is not None and times[-1] - start >= min_len_s:
        ranges.append((start, float(times[-1])))
    return ranges


def contact_sheet(video_path, out_path, n=12):
    container = av.open(video_path)
    stream = container.streams.video[0]
    dur = float(stream.duration * stream.time_base) if stream.duration else float(container.duration / 1e6)
    tiles = []
    for t in np.linspace(5, dur - 5, n):
        container.seek(int(t / stream.time_base), stream=stream)
        frame = next(container.decode(stream))
        img = cv2.resize(frame.to_ndarray(format="bgr24"), (320, 180))
        cv2.putText(img, f"{t / 60:.1f}m", (5, 18), 0, 0.55, (0, 255, 255), 2)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)]
    cv2.imwrite(out_path, np.vstack(rows))
    container.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--min-mouth", type=float, default=35, help="mouth width in px to count as large")
    p.add_argument("--sheet", help="also save a contact sheet image here")
    args = p.parse_args()

    samples, (w, h) = scan(args.video)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, stem + ".faces.json"), "w") as f:
        json.dump({"size": [w, h], "samples": samples}, f)

    mouths = [f["mouth_px"] for s in samples for f in s["faces"]]
    counts = np.bincount([len(s["faces"]) for s in samples], minlength=4)
    print(f"{stem}: {w}x{h}, {len(samples)} samples | faces per sample: "
          + ", ".join(f"{k}: {c}" for k, c in enumerate(counts) if c))
    if mouths:
        print(f"   mouth width px: median {np.median(mouths):.0f}, 90th pct {np.percentile(mouths, 90):.0f}")
    for a, b in large_ranges(samples, args.min_mouth):
        sub = [s for s in samples if a <= s["t"] < b and s["faces"]]
        n_faces = np.median([sum(f["mouth_px"] >= args.min_mouth for f in s["faces"]) for s in sub])
        m = np.median([max(f["mouth_px"] for f in s["faces"]) for s in sub])
        print(f"   large faces {a / 60:5.2f}-{b / 60:5.2f} min ({b - a:4.0f}s): ~{n_faces:.0f} large face(s), mouth ~{m:.0f}px")
    if args.sheet:
        contact_sheet(args.video, args.sheet)


if __name__ == "__main__":
    main()
