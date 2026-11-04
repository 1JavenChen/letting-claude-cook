"""Drop kept frames whose face isn't the clip's main person (emoji, b-roll, guests).

SMIRK estimates the 3D face *shape* (identity) on every frame, independent of
expression. Frames of the same person cluster tightly; a news anchor, a cartoon
emoji or a movie still land far away. This compares each frame's shape with the
median shape of the frames already kept and drops the outliers from `keep`.

    python tools/identity_filter.py data/processed/<name>.npz [--max-dist 2.0]

Needs <name>_flame.npz from tools/smirk_track.py (with frame_shape). On the
"ai is dumb" video (~100 px mouths) the user's frames sit at distance 0.5-1.8;
the news anchor, emoji graphics and a movie clip at 2.5-4.9. Small faces give
noisier shapes: on the wide desk shot (~37 px mouths) the user reaches 2.7, so
use --max-dist 3.0 there. Check what was dropped before trusting a cutoff.
"""

import argparse
import os

import numpy as np

FPS = 30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--flame", help="default: <npz without .npz>_flame.npz")
    p.add_argument("--max-dist", type=float, default=2.0)
    args = p.parse_args()

    flame_path = args.flame or args.npz[:-4] + "_flame.npz"
    f = np.load(flame_path)
    if "frame_shape" not in f:
        raise SystemExit(f"{flame_path} has no frame_shape - re-run tools/smirk_track.py")
    d = dict(np.load(args.npz))
    n = len(d["labels"])

    shape = f["frame_shape"].astype(np.float32)
    t = f["frame_times"] - float(f["start"])
    ok = f["frame_face"] & ~np.isnan(shape[:, 0])
    # label frame i shows source-video time src[i] / FPS (src = identity unless fix_drift.py shifted it)
    src = d["drift_src"] if "drift_src" in d else np.arange(n, dtype=np.float32)
    nearest = np.clip(np.searchsorted(t[ok], src / FPS), 0, ok.sum() - 1)
    grid_shape = shape[ok][nearest]
    close = np.abs(t[ok][nearest] - src / FPS) < 0.1

    ref = np.median(grid_shape[d["keep"] & close], axis=0)
    dist = np.linalg.norm(grid_shape - ref, axis=1)
    impostor = close & (dist > args.max_dist)

    d.setdefault("keep_before_identity", d["keep"].copy())
    d["keep"] = d["keep_before_identity"] & ~impostor
    d["identity_dist"] = np.where(close, dist, np.nan).astype(np.float32)
    tmp = args.npz[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, args.npz)
    dropped = (d["keep_before_identity"] & impostor).sum()
    print(f"{os.path.basename(args.npz)}: dropped {dropped / FPS:.1f} s of kept frames as a different "
          f"face; {d['keep'].sum() / FPS / 60:.1f} min kept")


if __name__ == "__main__":
    main()
