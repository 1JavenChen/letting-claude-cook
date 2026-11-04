"""Side-by-side video of answer keys against the iPhone footage they came from.

    .venv\\Scripts\\python tools/key_video.py multicam_<date> data/raw/session_<...> \\
        --keys labels_iph,labels_mha --start 600 --duration 60

Panels: iPhone close-up | one avatar head per label set (dimmed where its face_ok is false); mic audio.
--start in seconds into the clip.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.head3d import Head3D  # noqa: E402

FPS = 30
TITLES = {"labels_iph": "old key (iPhone MediaPipe)", "labels_mha": "new key (MetaHuman depth)",
          "labels_meas_lo": "webcam alone", "labels_arkit": "new key (Apple ARKit)"}


def iphone_frames(info, idx):
    """BGR iPhone frames at the given (non-decreasing) video frame indices: unpacked PNGs of a depth
    take, or the .mov decoded from the start (unmirrored in ARKit mode)."""
    if info.get("media"):
        pngs = sorted(glob.glob(os.path.join(info["media"], "Video", "*", "*.png")))
        for k in idx:
            yield cv2.imread(pngs[min(k, len(pngs) - 1)])
        return
    import av
    with av.open(info["mov"]) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        i, img = 0, None
        for k, f in enumerate(c.decode(s)):
            while i < len(idx) and idx[i] <= k:
                if img is None or idx[i] == k:
                    img = f.to_ndarray(format="bgr24")
                    img = np.ascontiguousarray(img[:, ::-1]) if info.get("mirrored") else img
                yield img
                i += 1
            img = None
            if i == len(idx):
                return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("session")
    p.add_argument("--keys", default="labels_iph,labels_mha")
    p.add_argument("--start", type=float, default=600)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--out")
    args = p.parse_args()

    d = np.load(os.path.join(ROOT, "data", "processed", args.clip + ".npz"))
    names = list(d["names"])
    keys = args.keys.split(",")
    with open(os.path.join(args.session, "iphone.json")) as f:
        info = json.load(f)
    vt = np.load(os.path.join(args.session, "iphone_times.npz"))["video_t"]
    grid = d["grid_t"]
    a = int(args.start * FPS)
    b = min(len(grid), a + int(args.duration * FPS))
    S = 400
    head = Head3D(names, size=S)
    out = args.out or os.path.join(ROOT, "out", f"keys_{args.clip}_{a // FPS}.mp4")
    tmp = out[:-4] + "_noaudio.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (225 + S * len(keys), S))
    tr = np.load(os.path.join(args.session, "track_iphone.npz"))
    win = (tr["t"] >= grid[a]) & (tr["t"] <= grid[b - 1]) & tr["ok"]
    lm = tr["lmk"][win, :, :2]
    c = np.median((lm.min(1) + lm.max(1)) / 2, 0)  # a steady crop around the face for the window
    rad = np.median((lm.max(1) - lm.min(1)).max(1)) * 0.8
    x0, y0 = int(max(c[0] - rad * 0.5625, 0)), int(max(c[1] - rad, 0))
    frames = iphone_frames(info, [int(np.searchsorted(vt, grid[k])) for k in range(a, b)])
    for k in range(a, b):
        img = next(frames)
        img = img[y0:y0 + int(2 * rad), x0:x0 + int(2 * rad * 0.5625)]
        panels = [cv2.resize(img, (225, S))]
        for key in keys:
            head.update(d[key][k])
            r = head.render()
            ok = d.get(key.replace("labels", "face_ok"))
            if ok is not None and not ok[k]:
                r = (r * 0.4).astype(np.uint8)
            cv2.putText(r, TITLES.get(key, key), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
            panels.append(r)
        wr.write(np.hstack(panels))
    wr.release()
    wav = out[:-4] + ".wav"
    seg = d["audio"][int(a / FPS * 16000):int(b / FPS * 16000)]
    sf.write(wav, seg * (0.9 / max(np.abs(seg).max(), 1e-6)), 16000)  # playback only: the mic records quietly
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", wav,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], check=True)
    os.remove(tmp)
    os.remove(wav)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
