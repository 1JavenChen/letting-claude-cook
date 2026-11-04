"""Triangulate face landmarks from a calibrated two-camera session into true 3D.

    .venv\\Scripts\\python tools/triangulate.py data/raw/session_<...> [--start 90] [--seconds 20]

For frames captured at the same moment by the front and side cameras (shared
clock, within half a frame), MediaPipe finds the 468 face landmarks in each
view; with the calibration (calib.json) each landmark is triangulated into a
3D point in millimetres, in the front camera's coordinate frame. Unlike
MediaPipe's own depth guess, this depth is measured.

Reports how well the 3D points re-project into both views (pixels) and a
sanity check of real distances (outer eye corners are typically ~9 cm apart).
Writes <session>/landmarks3d.npz (times, points [T,468,3] mm, reprojection error).
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


def frames(path, start, seconds):
    """[(t, rgb)] from one camera file, shared-clock times, between start and start+seconds
    (counted from the file's first frame)."""
    out, first = [], None
    with av.open(path) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            first = t if first is None else first
            if t - first < start:
                continue
            if t - first > start + seconds:
                break
            out.append((t, f.to_ndarray(format="rgb24")))
    return out


def landmarks(images):
    """[468,2] pixel landmarks per image (None where no face)."""
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=FACE_MODEL),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    out = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for k, (t, img) in enumerate(images):
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), int(t * 1000) + k)
            if r.face_landmarks:
                h, w = img.shape[:2]
                out.append(np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0][:468]]))
            else:
                out.append(None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--start", type=float, default=90, help="seconds after the first frame (after the board part)")
    p.add_argument("--seconds", type=float, default=20)
    args = p.parse_args()

    with open(os.path.join(args.session, "calib.json")) as f:
        cal = json.load(f)["cameras"]
    K1, d1 = np.array(cal["front"]["K"]), np.array(cal["front"]["dist"])
    K2, d2 = np.array(cal["side"]["K"]), np.array(cal["side"]["dist"])
    R, T = np.array(cal["side"]["R"]), np.array(cal["side"]["t_mm"]).reshape(3, 1)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # normalized cameras: points are undistorted first
    P2 = np.hstack([R, T])

    fa = frames(os.path.join(args.session, "front.mkv"), args.start, args.seconds)
    fb = frames(os.path.join(args.session, "side.mkv"), args.start - 1, args.seconds + 2)
    tb = np.array([t for t, _ in fb])
    pairs = []
    for t, img in fa:
        j = int(np.argmin(np.abs(tb - t)))
        if abs(tb[j] - t) < 0.017:
            pairs.append((t, img, fb[j][1]))
    la = landmarks([(t, a) for t, a, _ in pairs])
    lb = landmarks([(t, b) for t, _, b in pairs])

    times, pts, errs = [], [], []
    for (t, _, _), xa, xb in zip(pairs, la, lb):
        if xa is None or xb is None:
            continue
        ua = cv2.undistortPoints(xa.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
        ub = cv2.undistortPoints(xb.reshape(-1, 1, 2), K2, d2).reshape(-1, 2)
        X = cv2.triangulatePoints(P1, P2, ua.T, ub.T)
        X = (X[:3] / X[3]).T  # [468,3] mm, front-camera frame
        ra, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K1, d1)
        rb, _ = cv2.projectPoints(X, cv2.Rodrigues(R)[0], T, K2, d2)
        err = (np.linalg.norm(ra.reshape(-1, 2) - xa, axis=1).mean() +
               np.linalg.norm(rb.reshape(-1, 2) - xb, axis=1).mean()) / 2
        times.append(t)
        pts.append(X)
        errs.append(err)
    if not pts:
        sys.exit("no moments where both cameras saw the face")
    pts = np.array(pts, np.float32)
    eye = np.linalg.norm(pts[:, 33] - pts[:, 263], axis=1)  # outer eye corners
    depth = pts[:, 1, 2]  # nose tip distance from the front camera
    np.savez_compressed(os.path.join(args.session, "landmarks3d.npz"), times=np.array(times), points=pts,
                        reproj_px=np.array(errs, np.float32))
    print(f"{len(pts)} of {len(pairs)} paired frames with a face in both cameras")
    print(f"re-projection error: median {np.median(errs):.2f} px")
    print(f"outer eye corners: {np.median(eye):.1f} mm apart (typical adult ~85-100 mm), "
          f"frame-to-frame spread {np.std(eye):.1f} mm")
    print(f"nose tip is {np.median(depth) / 10:.0f} cm from the front camera")


if __name__ == "__main__":
    main()
