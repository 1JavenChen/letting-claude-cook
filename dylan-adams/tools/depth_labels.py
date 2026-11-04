"""Measured face labels from an iPhone TrueDepth take, compared with the current labels.

    .venv\\Scripts\\python tools/depth_labels.py data/raw/session_<...> [--start 60 --duration 40]

MediaPipe finds the face landmarks in the iPhone video (2D); the TrueDepth image
(same viewpoint) gives each landmark its measured distance, so every landmark
becomes a true 3D point in mm. The geometric retargeter (realtime/retarget.py)
fits ARKit-style weights to those points - every shape from geometry, including
lip push (pucker/funnel), which a single camera can't measure.

Where the depth can't be trusted at a landmark (inside an open mouth, at the face
edge) MediaPipe's own depth guess is kept, rescaled to match the measured ones.

Writes <session>/depth_labels.npz and a side-by-side video
out/depth_labels_<session>.mp4: iPhone | depth | current labels | measured labels.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import OpenEXR
from mediapipe.tasks.python import BaseOptions, vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.head3d import Head3D  # noqa: E402
from realtime.retarget import LandmarkRetargeter, hybrid, resting_offset  # noqa: E402

FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
TRUST_MM = 12  # measured depth this far from MediaPipe's (rescaled) guess is treated as a miss
MIN_USED = 0.6  # frames where fewer landmarks than this got real depth (face out of range) aren't measured


def read_depth(path):
    with OpenEXR.File(path) as e:
        d = e.channels()["Y"].pixels.astype(np.float32) * 10  # cm -> mm
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def lift(px, mp_z, depth, K):
    """px [468,2] video pixels, mp_z [468] MediaPipe relative depth (px units, + = away),
    depth [h,w] mm at half the video resolution -> [468,3] mm camera coords and a used-depth mask."""
    u = np.clip(np.round(px[:, 0] / 2).astype(int), 2, depth.shape[1] - 3)
    v = np.clip(np.round(px[:, 1] / 2).astype(int), 2, depth.shape[0] - 3)
    win = np.stack([depth[v + dv, u + du] for dv in (-1, 0, 1) for du in (-1, 0, 1)], 1)
    z = np.nanmedian(win, 1) if np.isfinite(win).any() else np.full(len(px), np.nan)
    ok = np.isfinite(z)
    # MediaPipe's depth guess, mapped onto the measured one by a robust line fit
    a, b = np.polyfit(mp_z[ok], z[ok], 1) if ok.sum() > 50 else (0.0, np.nanmedian(z))
    for _ in range(2):
        res = np.abs(a * mp_z + b - z)
        good = ok & (res < TRUST_MM)
        if good.sum() > 50:
            a, b = np.polyfit(mp_z[good], z[good], 1)
    good = ok & (np.abs(a * mp_z + b - z) < TRUST_MM)
    z = np.where(good, z, a * mp_z + b)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = np.stack([(px[:, 0] - cx) / fx * z, (px[:, 1] - cy) / fy * z, z], 1)
    return X, good


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--start", type=float, default=0, help="seconds into the iPhone take (video)")
    p.add_argument("--duration", type=float, default=40, help="length of the comparison video")
    args = p.parse_args()

    with open(os.path.join(args.session, "iphone.json")) as f:
        info = json.load(f)
    media = info["media"]
    K = np.array(info["video"]["K"])
    video = sorted(glob.glob(os.path.join(media, "Video", "*", "*.png")))
    depth = sorted(glob.glob(os.path.join(media, "Depth", "*", "*.exr")))
    tt = np.load(os.path.join(args.session, "iphone_times.npz"))
    vt, dt = tt["video_t"], tt["depth_t"]
    import torch
    names = list(torch.load(os.path.join(ROOT, "models", "fusion.pt"), map_location="cpu")["names"])

    # one sample per depth frame (30 fps), with the nearest video frame
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=FACE_MODEL),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    n = len(depth)
    mp_lmk = np.full((n, 468, 3), np.nan)
    meas = np.full((n, 468, 3), np.nan)
    scores = np.zeros((n, len(names)), np.float32)
    heads = np.zeros((n, 3), np.float32)
    used = np.zeros(n)
    vidx = np.array([int(np.argmin(np.abs(vt - t))) for t in dt])
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for k in range(n):
            img = cv2.cvtColor(cv2.imread(video[vidx[k]]), cv2.COLOR_BGR2RGB)
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img),
                                    int((dt[k] - dt[0]) * 1000))
            if not r.face_landmarks:
                continue
            h, w = img.shape[:2]
            q = np.array([[a.x * w, a.y * h, a.z * w] for a in r.face_landmarks[0][:468]])
            d = {c.category_name: c.score for c in r.face_blendshapes[0]}
            scores[k] = [d.get(x, 0.0) for x in names]
            R = np.array(r.facial_transformation_matrixes[0])[:3, :3]
            heads[k] = (np.arctan2(R[0, 2], R[2, 2]), np.arcsin(-np.clip(R[1, 2], -1, 1)),
                        np.arctan2(R[1, 0], R[1, 1]))
            mp_lmk[k] = q * [1, -1, -1]  # retargeter frame: x right, y up, z toward camera
            X, good = lift(q[:, :2], q[:, 2], read_depth(depth[k]), K)
            meas[k] = X * [1, -1, -1]
            used[k] = good.mean()
            if k % 500 == 0:
                print(f"  {k}/{n} frames")
    face = np.isfinite(mp_lmk[:, 0, 0])
    ok = face & (used >= MIN_USED)
    # TrueDepth is noisy frame to frame: average each point's depth over 3 frames (100 ms)
    zs = meas[..., 2].copy()
    for k in np.flatnonzero(ok[1:-1] & ok[:-2] & ok[2:]) + 1:
        zs[k] = meas[k - 1:k + 2, :, 2].mean(0)
    meas[..., 2] = zs
    print(f"face found in {face.mean():.0%} of {n} frames; face within depth range in "
          f"{ok.mean():.0%} (TrueDepth reaches ~50-80 cm)")

    # resting face: the most relaxed frames by MediaPipe's expression scores
    cols = [names.index(x) for x in ("jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
                                     "mouthFunnel", "browInnerUp", "eyeBlinkLeft", "eyeBlinkRight")]
    fi = np.flatnonzero(face)
    relaxed = fi[np.argsort(scores[fi][:, cols].sum(1))[:60]]

    # current labels: hybrid from MediaPipe's landmarks (its guessed depth) + scores
    rt_cur = LandmarkRetargeter(names)
    rt_cur.set_neutral(mp_lmk[relaxed])
    geo = rt_cur(mp_lmk[fi], scores[fi])
    rel = np.isin(fi, relaxed)
    cur = np.zeros_like(scores)
    cur[fi] = hybrid(scores[fi], geo, names, resting_offset(scores[fi], geo, names, rel))
    # measured labels: every shape from geometry, depth trusted as much as width/height
    mi = np.flatnonzero(ok)
    m_relaxed = mi[np.argsort(scores[mi][:, cols].sum(1))[:60]]
    rt_m = LandmarkRetargeter(names, z_weight=0.6)
    rt_m.set_neutral(meas[m_relaxed])
    geo_m = rt_m(meas[mi], scores[mi])
    off = np.nanmedian(geo_m[np.isin(mi, m_relaxed)], 0)
    look = np.array([x.startswith("eyeLook") for x in names])
    measured = np.zeros_like(scores)
    measured[mi] = np.where(look, geo_m, np.clip((geo_m - off) / np.maximum(1 - off, 0.2), 0, 1))
    np.savez_compressed(os.path.join(args.session, "depth_labels.npz"), times=dt, face=face, names=names,
                        measured=measured, measured_ok=ok, current=cur, mp_scores=scores, landmarks_mm=meas.astype(np.float32),
                        head=heads)

    print("shape            current  measured  (mean over measured frames)")
    for x in ("jawOpen", "mouthClose", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthFrownLeft",
              "mouthStretchLeft", "mouthRollLower", "mouthPressLeft", "eyeBlinkLeft", "browInnerUp"):
        i = names.index(x)
        c = np.corrcoef(cur[ok, i], measured[ok, i])[0, 1]
        print(f"  {x:16s} {cur[ok, i].mean():.2f}     {measured[ok, i].mean():.2f}    (agree r={c:.2f})")

    # side-by-side video
    sess = os.path.basename(os.path.normpath(args.session))
    out = os.path.join(ROOT, "out", f"depth_labels_{sess}.mp4")
    tmp = out[:-4] + "_noaudio.mp4"
    a = int(np.searchsorted(dt - dt[0], args.start))
    b = min(n, a + int(args.duration * 30))
    S = 480
    head = Head3D(names, size=S)
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (270 + 270 + 2 * S, S))
    for k in range(a, b):
        img = cv2.resize(cv2.imread(video[vidx[k]]), (270, S))
        d = read_depth(depth[k])
        dd = np.nan_to_num(d, nan=0)
        near = np.nanpercentile(d, 2) if np.isfinite(d).any() else 300
        vis = np.where(dd > 0, 255 * (1 - np.clip((dd - near) / 250, 0, 1)), 0).astype(np.uint8)
        vis = cv2.resize(cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO), (270, S))
        panels = [img, vis]
        for lab, title in ((cur, "current labels"), (measured, "measured (depth)")):
            head.update(lab[k], tuple(heads[k]) if face[k] else (0, 0, 0))
            r = head.render()
            if lab is measured and not ok[k]:
                r = (r * 0.35).astype(np.uint8)
                cv2.putText(r, "face out of depth range", (90, S // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
            cv2.putText(r, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
            panels.append(r)
        cv2.putText(panels[0], "iPhone", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panels[1], "TrueDepth", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        wr.write(np.hstack(panels))
    wr.release()
    wav = glob.glob(os.path.join(media, "Audio", "*", "*.wav"))[0]
    t_audio = dt[a] - info["shift_s"]  # iPhone audio clock
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-ss", f"{t_audio:.3f}",
                    "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", out], check=True)
    os.remove(tmp)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
