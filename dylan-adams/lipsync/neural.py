"""Phase 2 brain: trained model, audio -> blendshape animation JSON.

    python lipsync/neural.py data/test.wav out/test_neural.json

Writes the same format as rule_based.py, so blender/animate.py renders it.
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS, save_animation  # noqa: E402
from lipsync.rule_based import SR, load_mono  # noqa: E402
from model.encoders import ENCODERS  # noqa: E402
from model.head import MouthHead  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("out")
    p.add_argument("--ckpt", default=os.path.join(ROOT, "models", "mouth_wav2vec2.pt"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    head = MouthHead(**ckpt["config"]).to(device).eval()
    head.load_state_dict(ckpt["state_dict"])
    encoder = ENCODERS[ckpt["encoder"]](device=device)

    audio = load_mono(args.audio)
    n_frames = int(len(audio) / SR * FPS)
    feats = torch.from_numpy(encoder.encode(audio, n_frames)).float()[None].to(device)
    with torch.no_grad():
        frames = head(feats)[0].cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_animation(args.out, frames, names=ckpt["names"])
    print(f"Wrote {len(frames)} frames ({len(frames) / FPS:.1f}s) to {args.out}")


if __name__ == "__main__":
    main()
