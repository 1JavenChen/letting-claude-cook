"""Side-by-side video + scores: fusion models against the measured labels of a multi-camera clip.

    .venv\\Scripts\\python tools/compare_models.py multicam_<date> data/raw/session_<...> \\
        --models models/fusion.pt,models/fusion_meas_v1.pt --start 0.9 --duration 40

--start is a fraction of the clip (0.9 = the held-out last 10%) or seconds if > 1.
Panels: front webcam | answer key | webcam alone | one head per model; with the mic audio.
Scores (held-out last 10%): mean abs error vs the measured labels, overall and mouth only.
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
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.fusion import FusionHead  # noqa: E402
from realtime.head3d import Head3D  # noqa: E402

FPS = 30


def run_model(path, d, feats):
    ck = torch.load(path, map_location="cuda")
    head = FusionHead(**ck["config"]).cuda().eval()
    head.load_state_dict(ck["state_dict"])
    names = list(d["names"])
    assert ck["names"] == names
    video = d["labels_meas_lo"]
    face = d["face_ok_meas_lo"].astype(np.float32)
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None].cuda()  # noqa: E731
    with torch.no_grad():
        return head(t(feats), t(video * face[:, None]), t(face), t(face))[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("session")
    p.add_argument("--models", required=True)
    p.add_argument("--start", type=float, default=0.9)
    p.add_argument("--duration", type=float, default=40)
    p.add_argument("--target", default="labels_iph", help="label set to score against / show")
    p.add_argument("--out")
    args = p.parse_args()

    d = np.load(os.path.join(ROOT, "data", "processed", args.clip + ".npz"))
    feats = np.load(os.path.join(ROOT, "data", "features", "wav2vec2", args.clip + ".npy")).astype(np.float32)
    names = list(d["names"])
    models = args.models.split(",")
    preds = {m: run_model(m, d, feats) for m in models}
    meas, ok, cam = d[args.target], d[args.target.replace("labels", "face_ok")], d["labels_meas_lo"]
    n = len(meas)
    held = np.arange(n) >= int(0.9 * n)
    mouth = np.array([x.startswith(("jaw", "mouth")) for x in names])
    m = held & ok & d["face_ok_meas_lo"]
    print(f"held-out last 10% ({m.sum() / FPS:.0f} s): mean abs error vs {args.target}")
    print(f"  {'webcam alone (live input)':34s} all {np.abs(cam[m] - meas[m]).mean():.3f}  "
          f"mouth {np.abs(cam[m][:, mouth] - meas[m][:, mouth]).mean():.3f}")
    for k, pr in preds.items():
        print(f"  {os.path.basename(k):34s} all {np.abs(pr[m] - meas[m]).mean():.3f}  "
              f"mouth {np.abs(pr[m][:, mouth] - meas[m][:, mouth]).mean():.3f}")
    print("  per shape (r with measured):  " + "  ".join(os.path.basename(k)[:-3] for k in models))
    for s in ("jawOpen", "mouthClose", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthFrownLeft",
              "mouthStretchLeft", "mouthPressLeft", "browInnerUp", "eyeBlinkLeft"):
        i = names.index(s)
        rs = [np.corrcoef(pr[m, i], meas[m, i])[0, 1] for pr in preds.values()]
        print(f"    {s:16s} webcam {np.corrcoef(cam[m, i], meas[m, i])[0, 1]:.2f}   " +
              "   ".join(f"{r:.2f}" for r in rs))

    # video
    with open(os.path.join(args.session, "iphone.json")) as f:
        info = json.load(f)
    delay = info["webcams"]["front"]["delay_s"]
    grid = d["grid_t"]
    a = int(args.start * n) if args.start <= 1 else int(args.start * FPS)
    b = min(n, a + int(args.duration * FPS))
    S = 400
    head = Head3D(names, size=S)
    out = args.out or os.path.join(ROOT, "out", f"compare_{args.clip}_{a // FPS}.mp4")
    tmp = out[:-4] + "_noaudio.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (S * (2 + len(models)) + 300, S))
    with av.open(os.path.join(args.session, "front.mkv")) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        c.seek(int((grid[a] + delay - 1) / st.time_base), stream=st)
        frames = c.decode(st)
        cur = next(frames)
        for k in range(a, b):
            want = grid[k] + delay  # the stamp of the frame showing this moment
            while float(cur.pts * st.time_base) < want - 0.5 / FPS:
                cur = next(frames)
            img = cur.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            crop = img[:, (w - h * 3 // 4) // 2:(w + h * 3 // 4) // 2]
            panels = [cv2.resize(crop, (300, S))]
            for lab, title in [(meas, "answer key (iPhone)")] + [(preds[mm], os.path.basename(mm)[:-3]) for mm in models]:
                head.update(lab[k])
                r = head.render()
                if lab is meas and not ok[k]:
                    r = (r * 0.4).astype(np.uint8)
                cv2.putText(r, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
                panels.append(r)
            head.update(cam[k])
            r = head.render()
            cv2.putText(r, "webcam alone", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            panels.insert(2, r)
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
