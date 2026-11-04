"""Side-by-side video: real mouth (MediaPipe) vs. neural model vs. Phase 1 rules.

Picks the stretch of held-out audio (the last --val-frac of the clip, never
trained on) where the person talks the most, and renders all three mouths in
Blender with the original audio.

    python tools/compare_video.py data/processed/video_bottom.npz --label you
"""

import argparse
import json
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS, MOUTH_SHAPES, save_animation  # noqa: E402
from lipsync import rule_based  # noqa: E402
from model.data import Clip  # noqa: E402
from model.head import MouthHead  # noqa: E402

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
HOP = 16000 // FPS


def display_gain(labels, keep):
    """Per-shape boost so the 99th percentile of real movement reaches ~0.9."""
    p99 = np.percentile(labels[keep], 99, axis=0)
    return np.clip(0.9 / np.maximum(p99, 1e-3), 1.0, 5.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--label", default="")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--ckpt", default=os.path.join(ROOT, "models", "mouth_wav2vec2.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "out"))
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cuda")
    name = os.path.basename(args.npz)[:-4]
    clip = Clip(args.npz, os.path.join(ROOT, "data", "features", ckpt["encoder"], name + ".npy"))
    owner = np.load(args.npz)["voice_owner"] if "voice_owner" in np.load(args.npz) else None
    tag = name.split("_")[-1]
    talking = clip.keep & ((owner == ("a" if tag == "top" else "b")) if owner is not None else True)

    # Busiest window of talking inside the held-out tail.
    n = int(args.seconds * FPS)
    lo = int(len(clip) * (1 - args.val_frac))
    score = np.convolve(talking[lo:].astype(float), np.ones(n), "valid")
    s = lo + int(np.argmax(score))
    e = s + n
    print(f"Held-out segment {s / FPS / 60:.1f}-{e / FPS / 60:.1f} min of {name} "
          f"({score.max() / n:.0%} talking)")

    head = MouthHead(**ckpt["config"]).cuda().eval()
    head.load_state_dict(ckpt["state_dict"])
    with torch.no_grad():
        neural = head(torch.from_numpy(np.asarray(clip.feats[s:e], np.float32))[None].cuda())[0].cpu().numpy()
    real = clip.labels[s:e]
    gain = display_gain(clip.labels, clip.keep)
    audio = clip.audio[s * HOP: e * HOP]
    rules = rule_based.rules(rule_based.frame_features(audio))[:n]

    work = os.path.join(args.out, f"compare_{name}")
    os.makedirs(work, exist_ok=True)
    wav = os.path.join(work, "audio.wav")
    sf.write(wav, audio, 16000)
    panels = [("real", f"Real ({args.label or tag})", np.clip(real * gain, 0, 1)),
              ("neural", "Neural model", np.clip(neural * gain, 0, 1)),
              ("rules", "Phase 1 rules", rules)]
    videos = []
    for key, caption, frames in panels:
        anim = os.path.join(work, key + ".json")
        save_animation(anim, frames)
        video = os.path.join(work, key + ".mp4")
        subprocess.run([BLENDER, "-b", "-P", os.path.join(ROOT, "blender", "animate.py"), "--",
                        "--anim", anim, "--audio", wav, "--video", video, "--resolution", "480",
                        "--label", caption], check=True, capture_output=True)
        videos.append(video)
        print(f"  rendered {caption}")

    out = os.path.join(args.out, f"compare_{name}.mp4")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-v", "error", "-y"] + sum([["-i", v] for v in videos], []) + [
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-map", "0:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out]
    subprocess.run(cmd, check=True)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 0.0
    jaw = MOUTH_SHAPES.index("jawOpen")
    stats = {"neural_jaw_corr": corr(neural[:, jaw], real[:, jaw]),
             "rules_jaw_corr": corr(rules[:, jaw], real[:, jaw])}
    with open(os.path.join(work, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"jawOpen correlation with real on this segment: neural {stats['neural_jaw_corr']:+.2f}, "
          f"rules {stats['rules_jaw_corr']:+.2f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
