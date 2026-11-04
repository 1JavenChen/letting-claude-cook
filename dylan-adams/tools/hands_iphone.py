"""Hand-over-face flags for the iPhone view of a session (cached): face solvers read a hand as a face.

    .venv\\Scripts\\python tools/hands_iphone.py data/raw/session_<...> [--workers 3]

For every frame in track_iphone.npz (30 fps), MediaPipe's hand landmarker on the iPhone frame
(unpacked PNGs of a depth take, or the .mov of a video-only take, unmirrored for ARKit mode);
-> <session>/hands_iphone.npz: t, cover [N] = share of the face box (landmark extent) covered by
the box of any detected hand (0 = no hand on the face).
"""

import argparse
import glob
import json
import multiprocessing as mproc
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")


def read_frames(frames):
    """BGR images: PNG paths, or ("mov path", [frame indices], mirrored) decoded in order."""
    import cv2
    if isinstance(frames, list):
        for png in frames:
            yield cv2.imread(png)
        return
    import av
    path, want, flip = frames
    want = list(want)
    i = 0
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for k, f in enumerate(c.decode(s)):
            while i < len(want) and want[i] == k:
                img = f.to_ndarray(format="bgr24")
                yield np.ascontiguousarray(img[:, ::-1]) if flip else img
                i += 1
            if i == len(want):
                return


def run(job):
    idx, frames, boxes = job
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=HAND_MODEL), num_hands=2,
                                        running_mode=vision.RunningMode.IMAGE)
    cover = np.zeros(len(boxes), np.float32)
    with vision.HandLandmarker.create_from_options(opts) as hl:
        for i, (img, box) in enumerate(zip(read_frames(frames), boxes)):
            h, w = img.shape[:2]
            small = cv2.resize(img, (w // 2, h // 2))
            r = hl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            fx0, fy0, fx1, fy1 = box
            area = max((fx1 - fx0) * (fy1 - fy0), 1)
            for hand in r.hand_landmarks or []:
                xs = np.array([q.x * w for q in hand]); ys = np.array([q.y * h for q in hand])
                ix = max(0, min(xs.max(), fx1) - max(xs.min(), fx0))
                iy = max(0, min(ys.max(), fy1) - max(ys.min(), fy0))
                cover[i] = max(cover[i], ix * iy / area)
            if idx == 0 and i % 2000 == 0:
                print(f"  {i}/{len(boxes)}", flush=True)
    return idx, cover


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    S = args.session
    with open(os.path.join(S, "iphone.json")) as f:
        info = json.load(f)
    pngs = sorted(glob.glob(os.path.join(info["media"], "Video", "*", "*.png"))) if info.get("media") else None
    vt = np.load(os.path.join(S, "iphone_times.npz"))["video_t"]
    tr = np.load(os.path.join(S, "track_iphone.npz"))
    t, lmk, ok = tr["t"], tr["lmk"], tr["ok"]
    k = np.clip(np.searchsorted(vt, t - 1e-4), 0, len(vt) - 1)  # the frames track_views.py used
    # face box: this frame's landmarks, else the last seen (a hand often hides the face entirely)
    boxes = np.zeros((len(t), 4))
    last = np.array([0, 0, 1, 1.0])
    for i in range(len(t)):
        if ok[i]:
            last = np.r_[lmk[i, :, :2].min(0), lmk[i, :, :2].max(0)]
        boxes[i] = last
    chunks = np.array_split(np.arange(len(t)), args.workers)
    jobs = [(c, [pngs[j] for j in k[ix]] if pngs else (info["mov"], k[ix], info.get("mirrored", False)), boxes[ix])
            for c, ix in enumerate(chunks)]
    cover = [None] * len(jobs)
    with mproc.Pool(args.workers) as pool:
        for c, cv in pool.imap_unordered(run, jobs):
            cover[c] = cv
    cover = np.concatenate(cover)
    np.savez_compressed(os.path.join(S, "hands_iphone.npz"), t=t, cover=cover)
    print(f"{len(t)} frames: hand on the face (>10% of it) in {np.mean(cover > 0.1):.1%}")


if __name__ == "__main__":
    sys.exit(main())
