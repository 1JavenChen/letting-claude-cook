"""Side-by-side demo video: real footage | fused face | audio-only face (+ sound).

    python tools/demo_video.py data/processed/<name>.npz data/raw/<video> --start 82 --duration 30

Uses the trained fusion model on the cached features. The fused panel gets the
weak (low-res) camera tracking; press-style comparisons:
  --hide-mouth   fused panel gets the camera with the mouth hidden (audio fills it in)
  --view 2d      cartoon faces instead of the 3D head (realtime/head3d.py)
Output: out/demo_<name>_<start>.mp4
"""

import argparse
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
from common.blendshapes import FPS  # noqa: E402
from model.fusion import FusionHead  # noqa: E402
from realtime.idle import Blinker  # noqa: E402
from realtime.preview import draw_face  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("video")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--model", default=os.path.join(ROOT, "models", "fusion.pt"))
    p.add_argument("--crop", help="x0,y0,x1,y1 of the source video to show")
    p.add_argument("--hide-mouth", action="store_true")
    p.add_argument("--view", default="3d", choices=["3d", "2d"])
    p.add_argument("--out")
    args = p.parse_args()

    ck = torch.load(args.model)
    names = ck["names"]
    head = FusionHead(**ck["config"]).eval()
    head.load_state_dict(ck["state_dict"])
    d = np.load(args.npz)
    dn = list(d["names"])
    cols = [dn.index(n) for n in names]
    stem = os.path.basename(args.npz)[:-4]
    feats = np.load(os.path.join(ROOT, "data", "features", ck["encoder"], stem + ".npy"), mmap_mode="r")

    a, b = int(args.start * FPS), int((args.start + args.duration) * FPS)
    ctx = 3 * FPS  # warm-up context so the model has history at the first frame
    lo = max(0, a - ctx)
    target = ck.get("target", "labels")  # the label set the model was trained on
    if target != "labels":  # hybrid labels are stored with 51 columns (no "_neutral")
        dn = [n for n in dn if n != "_neutral"]
        cols = [dn.index(n) for n in names]
    ok_key = "face_ok" if target == "labels" else target.replace("labels", "face_ok")
    weak = d[target + "_lo"] if target + "_lo" in d else d[target]
    seen_v = d[ok_key + "_lo"] if ok_key + "_lo" in d else d[ok_key]
    video = weak[lo:b][:, cols] * seen_v[lo:b, None]
    seen = seen_v[lo:b].astype(np.float32)
    mouth_cols = head.mouth_cols.numpy()
    mouth = np.zeros_like(seen) if args.hide_mouth else seen.copy()
    if args.hide_mouth:
        video[:, mouth_cols] = 0
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None]  # noqa: E731
    with torch.no_grad():
        fused = head(t(feats[lo:b]), t(video), t(seen), t(mouth))[0].numpy()[a - lo:]
        z = np.zeros_like(seen)
        audio_only = head(t(feats[lo:b]), t(np.zeros_like(video)), t(z), t(z))[0].numpy()[a - lo:]
    # eyes the camera can't see: open with natural blinks, not the model's "average" half-lid
    bl_f, bl_a = Blinker(names, FPS, seed=0), Blinker(names, FPS, seed=1)
    seen_out = seen[a - lo:]
    fused = np.stack([bl_f(w, s > 0) for w, s in zip(fused, seen_out)])
    audio_only = np.stack([bl_a(w, False) for w in audio_only])

    # frame i of the labels corresponds to source-video time src[i] / FPS
    src = d["drift_src"] if "drift_src" in d else np.arange(len(d["labels"]), dtype=np.float32)
    want = src[a:b] / FPS
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
    container = av.open(args.video)
    stream = container.streams.video[0]
    container.seek(int(max(0, want[0] - 2) / stream.time_base), stream=stream)
    out_path = args.out or os.path.join(ROOT, "out", f"demo_{stem[:20]}_{int(args.start)}.mp4")
    tmp = out_path + ".video.mp4"
    H = 360
    head3d = None
    if args.view == "3d":
        from realtime.head3d import Head3D
        head3d = Head3D(names, size=H)
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (H * 16 // 9 + 2 * H, H))
    k, last = 0, None
    label_mode = "fused (mouth hidden -> from voice)" if args.hide_mouth else "fused: weak camera + voice"
    for frame in container.decode(stream):
        ft = float(frame.pts * stream.time_base)
        while k < len(want) and want[k] <= ft:
            img = last if last is not None else frame.to_ndarray(format="bgr24")
            if crop:
                h, w = img.shape[:2]
                img = img[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
            left = cv2.resize(img, (H * 16 // 9, H))
            panels = []
            for vals, title in ((fused[k], label_mode), (audio_only[k], "voice only")):
                if head3d:
                    head3d.update(vals)
                    pane = head3d.render(bg=(35, 35, 35))
                else:
                    pane = np.full((H, H, 3), 35, np.uint8)
                    draw_face(pane, dict(zip(names, vals)), H // 2, int(H * 0.45), int(H * 0.27))
                cv2.putText(pane, title, (8, 20 if head3d else H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
                panels.append(pane)
            cv2.putText(left, "real", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(np.hstack([left] + panels))
            k += 1
        last = frame.to_ndarray(format="bgr24")
        if k >= len(want):
            break
    writer.release()
    container.close()

    wav = out_path + ".wav"
    sr = int(d["sr"])
    sf.write(wav, d["audio"][a * sr // FPS: b * sr // FPS], sr)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", tmp, "-i", wav,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path],
                   check=True)
    os.remove(tmp)
    os.remove(wav)
    print(f"Saved {out_path} ({k} frames)")


if __name__ == "__main__":
    main()
