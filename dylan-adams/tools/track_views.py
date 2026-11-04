"""Face landmarks for every frame of every camera in a multi-camera session (cached).

    .venv\\Scripts\\python tools/track_views.py data/raw/session_<...> [--start 90]

One process per camera (front.mkv, side.mkv, the iPhone .mov at 30 fps) runs
MediaPipe and saves <session>/track_<camera>.npz:
  t       frame time on the session clock (webcams: as stamped, before delay correction)
  lmk     [N,468,3] MediaPipe landmarks in pixels (x, y, z with z in the same px scale)
  scores  [N,52] MediaPipe blendshape scores (names in `names`)
  ok      [N] a face was found
Frames before --start (seconds after the session's first webcam frame: the board part) are skipped.
"""

import argparse
import glob
import json
import multiprocessing as mproc
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")


def track(job):
    name, path, times, t_from, out, flip = job
    import av
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=FACE_MODEL),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                        output_face_blendshapes=True)
    T, L, S, OK = [], [], [], []
    names = None
    last_ms = -1

    def frames():  # (index, time or None, rgb image)
        if isinstance(path, list):  # depth take: the video was unpacked to PNGs by Capture Manager
            import cv2
            for k, png in enumerate(path):
                if k % 2 == 0 and times[k] >= t_from:
                    yield k, None, cv2.cvtColor(cv2.imread(png), cv2.COLOR_BGR2RGB)
            return
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for k, f in enumerate(c.decode(s)):
                yield k, float(f.pts * s.time_base), f

    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for k, t, f in frames():
            if times is not None:  # iPhone: times from the frame log; keep every 2nd frame (30 fps)
                if k >= len(times):
                    break
                t = float(times[k])
                if k % 2:
                    continue
            if t < t_from:
                continue
            img = f if isinstance(f, np.ndarray) else np.ascontiguousarray(f.to_ndarray(format="rgb24"))  # 1080-wide rows are padded
            if flip:  # ARKit-mode takes are saved mirrored
                img = np.ascontiguousarray(img[:, ::-1])
            ms = max(int(t * 1000), last_ms + 1)
            last_ms = ms
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), ms)
            T.append(t)
            if r.face_landmarks:
                h, w = img.shape[:2]
                L.append(np.array([[q.x * w, q.y * h, q.z * w] for q in r.face_landmarks[0][:468]], np.float32))
                names = names or [b.category_name for b in r.face_blendshapes[0]]
                S.append(np.array([b.score for b in r.face_blendshapes[0]], np.float32))
                OK.append(True)
            else:
                L.append(np.full((468, 3), np.nan, np.float32))
                S.append(np.zeros(52, np.float32))
                OK.append(False)
            if len(T) % 3000 == 0:
                print(f"  {name}: {len(T)} frames (t={t:.0f} s), face in {np.mean(OK):.0%}", flush=True)
    np.savez_compressed(out, t=np.array(T), lmk=np.stack(L), scores=np.stack(S), ok=np.array(OK),
                        names=np.array(names or []), size=np.array(img.shape[1::-1]))
    return f"{name}: {len(T)} frames, face found in {np.mean(OK):.0%}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--start", type=float, default=90, help="skip the board part (s after the first webcam frame)")
    p.add_argument("--only", help="comma-separated cameras")
    args = p.parse_args()

    import av
    with av.open(os.path.join(args.session, "front.mkv")) as c:
        s = c.streams.video[0]
        t0 = float(next(c.demux(s)).pts * s.time_base)
    t_from = t0 + args.start
    with open(os.path.join(args.session, "iphone.json")) as f:
        info = json.load(f)
    vt = np.load(os.path.join(args.session, "iphone_times.npz"))["video_t"]
    jobs = [("front", os.path.join(args.session, "front.mkv"), None),
            ("side", os.path.join(args.session, "side.mkv"), None),
            ("iphone", info.get("mov") or sorted(glob.glob(os.path.join(info["media"], "Video", "*", "*.png"))), vt)]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only.split(",")]
    jobs = [(n, pth, tm, t_from, os.path.join(args.session, f"track_{n}.npz"), n == "iphone" and info.get("mirrored", False))
            for n, pth, tm in jobs]
    with mproc.Pool(len(jobs)) as pool:
        for msg in pool.imap_unordered(track, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
