"""Track the same video again at low resolution, as a "weak webcam" input.

The fusion model learns: (weak camera tracking + audio) -> full-quality
tracking. This makes the weak half of that pair from the very same frames,
so the only difference between input and target is image quality.

    python tools/degrade_track.py data/processed/<name>.npz data/raw/<video> --width 640

Adds `labels_lo` and `face_ok_lo` to the .npz. Assumes the .npz was made from
the whole video (start 0); pass --crop if prepare_data.py used one. Run fix_drift.py first: its timing
correction is applied here too.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import resample_labels, track_face  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("video")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--crop", help="same crop as prepare_data.py used, x0,y0,x1,y1")
    args = p.parse_args()

    d = dict(np.load(args.npz))
    n = len(d["labels"])
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
    times, blend, names, _ = track_face(args.video, crop, max_width=args.width)
    assert list(names) == list(d["names"]), "blendshape names differ"
    labels_lo, face_ok_lo = resample_labels(times, blend, n)
    if "drift_src" in d:  # fix_drift.py already re-aligned the labels: shift these the same way
        src = d["drift_src"]
        face_ok_lo = face_ok_lo[np.clip(np.round(src).astype(int), 0, n - 1)]
        labels_lo = np.stack([np.interp(src, np.arange(n), labels_lo[:, i])
                              for i in range(labels_lo.shape[1])], 1).astype(np.float32)
    d["labels_lo"], d["face_ok_lo"] = labels_lo, face_ok_lo
    np.savez_compressed(args.npz, **d)

    both = d["face_ok"] & face_ok_lo
    jaw = list(names).index("jawOpen")
    c = np.corrcoef(d["labels"][both, jaw], labels_lo[both, jaw])[0, 1]
    print(f"{os.path.basename(args.npz)}: low-res ({args.width}px wide) face found "
          f"{face_ok_lo[d['face_ok']].mean():.0%} of the time the full-res one was; "
          f"jawOpen agreement corr {c:.2f}")


if __name__ == "__main__":
    main()
