"""Short three-panel video for class: front webcam | answer key | model output, with the mic audio.

    .venv\\Scripts\\python tools/class_video.py multicam_<date> data/raw/session_<...> \\
        --model models/fusion_arkit_v1.pt --target labels_arkit --start 998.7 --duration 20 --out out/class.mp4

--start in seconds into the clip. A caption strip under each panel names it.
"""

import argparse
import json
import os
import subprocess
import sys

import av
import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compare_models import run_model  # noqa: E402
from realtime.head3d import Head3D  # noqa: E402

FPS, S, BAR = 30, 400, 56


def caption(img, top, sub):
    bar = np.full((BAR, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, top, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (17, 17, 17), 2, cv2.LINE_AA)
    cv2.putText(bar, sub, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (102, 102, 102), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("session")
    p.add_argument("--model", required=True)
    p.add_argument("--target", default="labels_arkit")
    p.add_argument("--start", type=float, default=998.7)
    p.add_argument("--duration", type=float, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = np.load(os.path.join(ROOT, "data", "processed", args.clip + ".npz"))
    feats = np.load(os.path.join(ROOT, "data", "features", "wav2vec2", args.clip + ".npy")).astype(np.float32)
    names = list(d["names"])
    pred = run_model(args.model, d, feats)
    key, ok = d[args.target], d[args.target.replace("labels", "face_ok")]
    with open(os.path.join(args.session, "iphone.json")) as f:
        delay = json.load(f)["webcams"]["front"]["delay_s"]
    grid = d["grid_t"]
    a = int(args.start * FPS)
    b = min(len(grid), a + int(args.duration * FPS))
    tr = np.load(os.path.join(args.session, "track_front.npz"))  # a steady square crop around the face
    win = (tr["t"] >= grid[a]) & (tr["t"] <= grid[b - 1] + 1) & tr["ok"]
    lm = tr["lmk"][win, :, :2]
    cx, cy = np.median((lm.min(1) + lm.max(1)) / 2, 0)
    rad = int(np.median((lm.max(1) - lm.min(1)).max(1)) * 1.35)
    head = Head3D(names, size=S)
    tmp = args.out[:-4] + "_noaudio.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (S * 3, S + BAR))
    with av.open(os.path.join(args.session, "front.mkv")) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        c.seek(int((grid[a] + delay - 1) / st.time_base), stream=st)
        frames = c.decode(st)
        cur = next(frames)
        for k in range(a, b):
            while float(cur.pts * st.time_base) < grid[k] + delay - 0.5 / FPS:
                cur = next(frames)
            img = cur.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            rad = min(rad, h // 2)
            x0, y0 = int(np.clip(cx - rad, 0, w - 2 * rad)), int(np.clip(cy - rad * 0.9, 0, h - 2 * rad))
            cam = cv2.resize(img[y0:y0 + 2 * rad, x0:x0 + 2 * rad], (S, S))
            head.update(key[k])
            rk = head.render()
            if not ok[k]:
                rk = (rk * 0.4).astype(np.uint8)
            head.update(pred[k])
            rm = head.render()
            wr.write(np.hstack([caption(cam, "Webcam", "the live input (plus the mic)"),
                                caption(rk, "Answer key", "Apple ARKit, iPhone depth camera"),
                                caption(rm, "Model output", "webcam + voice, never saw this part")]))
    wr.release()
    wav = args.out[:-4] + ".wav"
    seg = d["audio"][int(a / FPS * 16000):int(b / FPS * 16000)]
    sf.write(wav, seg * (0.9 / max(np.abs(seg).max(), 1e-6)), 16000)  # playback only: the mic records quietly
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", wav,
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", args.out],
                   check=True)
    os.remove(tmp)
    os.remove(wav)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
