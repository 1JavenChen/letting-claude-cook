"""Label a talking-head video with SMIRK FLAME parameters (run with .venv-smirk).

Per video frame: MediaPipe finds the largest face, the frame is cropped the way
SMIRK expects (landmark box * 1.4 -> 224x224), and SMIRK's encoder predicts FLAME
parameters on the GPU in batches. The result is resampled onto the same 30 fps
grid as tools/prepare_data.py so it lines up with data/processed/<name>.npz.

Usage:
    .venv-smirk\\Scripts\\python tools/smirk_track.py VIDEO --out data/processed/<name>_flame.npz
        [--crop x0,y0,x1,y1] [--start S --duration D] [--match data/processed/<name>.npz]

Output npz (n = grid frames, N = video frames):
    expression [n,50]  FLAME expression coefficients
    jaw        [n,3]   jaw pose, axis-angle (jaw[:,0] = opening in radians, >= 0)
    eyelid     [n,2]   SMIRK eyelid closure (left, right) in [0,1]
    pose       [n,3]   global head rotation, axis-angle
    cam        [n,3]   orthographic camera (scale, tx, ty) relative to the crop
    face_ok    [n]     a face was found within 0.1 s of this grid frame
    shape_mean [300]   median identity shape over the clip
    start, start_frame grid frame 0 = video time `start` = frame `start_frame` of a
                       full-video prepare_data npz (when --match is used the arrays
                       are already padded to that npz's length and start_frame=0)
    frame_times [N], frame_params [N,61] (pose,cam,exp,eyelid,jaw), frame_face [N],
    crop_affine [N,2,3] (frame px -> 224 crop px)  - raw per-video-frame values
    speed_fps          end-to-end frames per second
"""

import argparse
import os
import sys
import threading
import queue
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS  # noqa: E402
from prepare_data import SR, extract_audio, resample_labels  # noqa: E402
from smirk_flame import CROP_SIZE, crop_transform, load_encoder  # noqa: E402

FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
PARAM_SLICES = {"pose": (0, 3), "cam": (3, 6), "expression": (6, 56), "eyelid": (56, 58), "jaw": (58, 61)}


def _decode(video_path, start, end, crop, out_q):
    """Background thread: decoded RGB frames (+ exact PTS) -> queue."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        if start:
            container.seek(int(max(0, start - 2) / tb), stream=stream)  # earlier keyframe
        for frame in container.decode(stream):
            t = float(frame.pts * tb)
            if t < start:
                continue
            if end is not None and t >= end:
                break
            rgb = frame.to_ndarray(format="rgb24")
            if crop:
                h, w = rgb.shape[:2]
                rgb = rgb[int(crop[1] * h): int(crop[3] * h), int(crop[0] * w): int(crop[2] * w)]
            out_q.put((t - start, np.ascontiguousarray(rgb)))
        container.close()
    finally:
        out_q.put(None)


def track(video_path, crop=None, start=0.0, duration=None, max_width=640, batch=64,
          device="cuda", fp16=False):
    """Returns frame_times [N], params [N,61] (NaN rows = no face), shape [N,300], affine [N,2,3]."""
    encoder = load_encoder(device)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=3)
    with av.open(video_path) as c:
        s = c.streams.video[0]
        span = duration or (float(s.duration * s.time_base) - start if s.duration else None)
        total = int(span * float(s.average_rate)) if span and s.average_rate else None
    end = start + duration if duration else None

    q = queue.Queue(maxsize=64)
    threading.Thread(target=_decode, args=(video_path, start, end, crop, q), daemon=True).start()

    times, affines, params, shapes = [], [], [], []
    pending_idx, pending_img = [], []

    def flush():
        if not pending_img:
            return
        x = torch.from_numpy(np.stack(pending_img)).to(device).permute(0, 3, 1, 2).float() / 255.0
        with torch.inference_mode(), torch.autocast("cuda", enabled=fp16):
            o = encoder(x)
        p = torch.cat([o["pose_params"], o["cam"], o["expression_params"], o["eyelid_params"],
                       o["jaw_params"]], 1).float().cpu().numpy()
        sh = o["shape_params"].float().cpu().numpy()
        for k, i in enumerate(pending_idx):
            params[i] = p[k]
            shapes[i] = sh[k]
        pending_idx.clear()
        pending_img.clear()

    last_ms = -1
    progress = tqdm(total=total, desc="  SMIRK", unit="frame", leave=False)
    t0 = time.time()
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while (item := q.get()) is not None:
            t, rgb = item
            ms = max(int(round(t * 1000)), last_ms + 1)  # MediaPipe needs increasing ms
            last_ms = ms
            h, w = rgb.shape[:2]
            small = cv2.resize(rgb, (max_width, int(h * max_width / w))) if w > max_width else rgb
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=small), ms)
            i = len(times)
            times.append(t)
            params.append(np.full(61, np.nan, np.float32))
            shapes.append(np.full(300, np.nan, np.float32))
            affines.append(np.full((2, 3), np.nan))
            if res.face_landmarks:
                face = max(res.face_landmarks, key=lambda L: (max(p.x for p in L) - min(p.x for p in L))
                           * (max(p.y for p in L) - min(p.y for p in L)))
                lm = np.array([[p.x * w, p.y * h] for p in face])  # full-res pixel coords
                A = crop_transform(lm)
                affines[i] = A
                pending_img.append(cv2.warpAffine(rgb, A, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR))
                pending_idx.append(i)
                if len(pending_img) >= batch:
                    flush()
            progress.update()
        flush()
    progress.close()
    elapsed = time.time() - t0
    return (np.array(times), np.array(params, np.float32), np.array(shapes, np.float32),
            np.array(affines, np.float32), len(times) / max(elapsed, 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="default: data/processed/<video name>_flame.npz")
    ap.add_argument("--crop", help="region as fractions x0,y0,x1,y1 (same as prepare_data.py)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--max-width", type=int, default=640, help="downscale for MediaPipe detection only; "
                    "SMIRK always crops from the full-resolution frame")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", help="autocast the encoder (slightly faster, tiny drift)")
    ap.add_argument("--match", help="prepare_data npz of the same (full) video: pad the output onto its "
                    "frame grid so arrays line up index-for-index")
    a = ap.parse_args()
    crop = tuple(float(v) for v in a.crop.split(",")) if a.crop else None
    name = os.path.splitext(os.path.basename(a.video))[0]
    out = a.out or os.path.join(ROOT, "data", "processed", name + "_flame.npz")

    audio = extract_audio(a.video, a.start, a.duration)
    times, params, shapes, affines, fps = track(a.video, crop, a.start, a.duration, a.max_width,
                                                a.batch, fp16=a.fp16)
    face = ~np.isnan(params[:, 0])
    if face.sum() < 2:
        sys.exit("no face found")
    n = min(int(len(audio) / SR * FPS), int(times[-1] * FPS))
    grid, face_ok = resample_labels(times, params, n)
    start_frame = int(round(a.start * FPS))

    if a.match:
        ref_n = len(np.load(a.match)["face_ok"])
        pad = lambda x: np.concatenate([np.zeros((start_frame,) + x.shape[1:], x.dtype), x,  # noqa: E731
                                        np.zeros((max(0, ref_n - start_frame - n),) + x.shape[1:], x.dtype)])[:ref_n]
        grid, face_ok = pad(grid), pad(face_ok)
        start_frame = 0

    res = {k: grid[:, s:e] for k, (s, e) in PARAM_SLICES.items()}
    np.savez_compressed(
        out, **res, face_ok=face_ok, fps=FPS, start=a.start, start_frame=start_frame,
        shape_mean=np.nanmedian(shapes[face], 0).astype(np.float32),
        frame_shape=shapes[:, :50].astype(np.float16),  # per-frame identity, for "is this the same person?"
        frame_times=times, frame_params=params, frame_face=face, crop_affine=affines,
        param_layout=np.array([f"{k}:{s}:{e}" for k, (s, e) in PARAM_SLICES.items()]),
        speed_fps=fps, source=os.path.basename(a.video))
    print(f"{len(times)} frames ({face.mean():.0%} with a face) at {fps:.1f} frames/s -> {out}")
    print(f"grid: {len(face_ok)} frames @ {FPS} fps, face_ok {face_ok.mean():.0%}")


if __name__ == "__main__":
    main()
