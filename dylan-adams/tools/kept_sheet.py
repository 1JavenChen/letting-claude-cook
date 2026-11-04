"""Contact sheet of one frame per kept stretch, to confirm by eye that every
kept shot really is the person you want (not b-roll or someone else).

    python tools/kept_sheet.py data/processed/<name>.npz data/raw/<video> out/<name>_kept.jpg
"""

import argparse
import os
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import FPS  # noqa: E402


def runs(mask, min_len):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--cols", type=int, default=6)
    args = p.parse_args()

    keep = np.load(args.npz)["keep"]
    stretches = runs(keep, FPS)
    container = av.open(args.video)
    stream = container.streams.video[0]
    tiles = []
    for a, b in stretches:
        t = (a + b) / 2 / FPS
        container.seek(int(max(0, t - 1) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts * stream.time_base >= t:
                break
        img = cv2.resize(frame.to_ndarray(format="bgr24"), (256, 144))
        cv2.putText(img, f"{a / FPS:.0f}-{b / FPS:.0f}s", (4, 16), 0, 0.5, (0, 255, 255), 1)
        tiles.append(img)
    container.close()
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"{len(stretches)} kept stretches -> {args.out}")


if __name__ == "__main__":
    main()
