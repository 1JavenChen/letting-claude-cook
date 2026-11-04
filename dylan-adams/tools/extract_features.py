"""Run the frozen audio encoder over every processed clip once and cache it.

    python tools/extract_features.py                 # wav2vec2 (default)
    python tools/extract_features.py --encoder NAME  # future encoders

Writes data/features/<encoder>/<name>.npy  (float16, [frames, dim], 30 fps,
aligned 1:1 with the labels in data/processed/<name>.npz).
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.encoders import ENCODERS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="wav2vec2", choices=ENCODERS)
    p.add_argument("--processed", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = os.path.join(ROOT, "data", "features", args.encoder)
    os.makedirs(out_dir, exist_ok=True)
    clips = sorted(f[:-4] for f in os.listdir(args.processed) if f.endswith(".npz"))
    todo = [c for c in clips if args.force or not os.path.exists(os.path.join(out_dir, c + ".npy"))]
    print(f"{len(clips)} clips, {len(todo)} to encode with {args.encoder}")
    if not todo:
        return
    encoder = ENCODERS[args.encoder]()
    for name in tqdm(todo, unit="clip"):
        data = np.load(os.path.join(args.processed, name + ".npz"))
        if "audio" not in data.files or "labels" not in data.files:
            continue  # not a training clip (e.g. a *_flame.npz side file)
        feats = encoder.encode(data["audio"], len(data["labels"]))
        np.save(os.path.join(out_dir, name + ".npy"), feats)


if __name__ == "__main__":
    main()
