"""Make geometric ARKit labels for a processed clip from MediaPipe's 3D landmarks.

    .venv\\Scripts\\python tools/retarget_labels.py data/processed/<name>.npz data/raw/<video> \\
        [--crop x0,y0,x1,y1] [--max-width 1920] [--lo-width 320]

Re-tracks the video keeping MediaPipe's landmarks, estimates the person's
neutral face from the clip's most relaxed frames, and retargets every frame onto
the ICT head's ARKit shapes (realtime/retarget.py). Adds to the clip .npz:
  labels_geo [n,51]  geometric labels (same order as the clip's names minus _neutral)
  labels_geo_lo      the same from a low-resolution pass (the "weak webcam" input)
  face_ok_geo, face_ok_geo_lo
Both are shifted by `drift_src` like tools/degrade_track.py. Landmarks are cached
next to the clip as <name>_lmk.npz so re-solving with other settings is fast.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from prepare_data import resample_labels, track_face  # noqa: E402
from realtime.retarget import LandmarkRetargeter  # noqa: E402

RELAX = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
         "browInnerUp", "browDownLeft", "browDownRight", "eyeBlinkLeft", "eyeBlinkRight",
         "mouthUpperUpLeft", "mouthUpperUpRight"]


def track(video, crop, width, cache):
    if os.path.exists(cache):
        c = np.load(cache)
        return c["times"], c["scores"], list(c["names"]), c["lmk"]
    times, scores, names, _, lmk = track_face(video, crop, max_width=width, landmarks=True)
    np.savez(cache, times=times, scores=scores, names=np.array(names), lmk=lmk)
    return times, scores, names, lmk


def neutral_frames(scores, names, ok, n=90):
    """The clip's most relaxed frames, by MediaPipe's expression scores."""
    cols = [names.index(s) for s in RELAX]
    energy = np.where(ok, scores[:, cols].sum(1), np.inf)
    return np.argsort(energy)[:n]


def geo_labels(rt, times, scores, names, lmk, n_grid, neutral_idx=None):
    ok = ~np.isnan(lmk[:, 0, 0])
    if neutral_idx is None:
        neutral_idx = neutral_frames(scores, names, ok)
    rt.set_neutral(lmk[neutral_idx])
    out = np.full((len(times), len(rt.names)), np.nan, np.float32)
    mp = scores[:, [names.index(s) for s in rt.names]]
    idx = np.flatnonzero(ok)
    for i in range(0, len(idx), 4096):
        b = idx[i:i + 4096]
        out[b] = rt(lmk[b], mp[b])
    return resample_labels(times, out, n_grid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("video")
    p.add_argument("--crop")
    p.add_argument("--max-width", type=int, default=1920)
    p.add_argument("--lo-width", type=int, default=320, help="0 = skip the low-res pass")
    args = p.parse_args()
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None

    d = dict(np.load(args.npz))
    n = len(d["labels"])
    names = [s for s in d["names"] if s != "_neutral"]
    rt = LandmarkRetargeter(names)
    stem = args.npz[:-4]

    times, scores, mp_names, lmk = track(args.video, crop, args.max_width, stem + "_lmk.npz")
    geo, ok = geo_labels(rt, times, scores, mp_names, lmk, n)
    results = {"labels_geo": geo, "face_ok_geo": ok}
    neutral_info = f"neutral from {len(neutral_frames(scores, mp_names, ~np.isnan(lmk[:, 0, 0])))} relaxed frames"
    if args.lo_width:
        t2, s2, n2, l2 = track(args.video, crop, args.lo_width, stem + f"_lmk{args.lo_width}.npz")
        geo_lo, ok_lo = geo_labels(rt, t2, s2, n2, l2, n)
        results.update(labels_geo_lo=geo_lo, face_ok_geo_lo=ok_lo)

    if "drift_src" in d:  # the clip's labels were re-aligned by fix_drift.py: shift these too
        src = d["drift_src"]
        for k, v in list(results.items()):
            if v.dtype == bool:
                results[k] = v[np.clip(np.round(src).astype(int), 0, n - 1)]
            else:
                results[k] = np.stack([np.interp(src, np.arange(n), v[:, i]) for i in range(v.shape[1])],
                                      1).astype(np.float32)
    d.update(results)
    tmp = stem + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, args.npz)

    k = d["keep"] & results["face_ok_geo"]
    g, mpl = results["labels_geo"][k], d["labels"][k][:, [list(d["names"]).index(s) for s in names]]
    print(f"{os.path.basename(args.npz)}: {neutral_info}; kept frames {k.sum()}")
    print(f"  {'shape':20s} {'MP rest':>8s} {'geo rest':>8s} {'MP p95':>7s} {'geo p95':>7s} {'corr':>6s}")
    rest = k & (d["labels"][:, list(d["names"]).index("jawOpen")] < np.percentile(mpl[:, names.index("jawOpen")], 20))
    for s in ["eyeBlinkLeft", "eyeSquintLeft", "browDownLeft", "browInnerUp", "jawOpen", "mouthClose",
              "mouthSmileLeft", "mouthStretchLeft", "mouthFrownLeft", "mouthPressLeft", "mouthPucker",
              "mouthFunnel", "mouthUpperUpLeft", "mouthLowerDownLeft", "cheekSquintLeft", "mouthRollLower"]:
        j = names.index(s)
        rg = results["labels_geo"][rest, j]
        rm = d["labels"][rest, list(d["names"]).index(s)]
        c = np.corrcoef(g[:, j], mpl[:, j])[0, 1] if g[:, j].std() > 1e-6 and mpl[:, j].std() > 1e-6 else 0
        print(f"  {s:20s} {np.nanmedian(rm):8.2f} {np.nanmedian(rg):8.2f} {np.percentile(mpl[:, j], 95):7.2f} "
              f"{np.percentile(g[:, j], 95):7.2f} {c:6.2f}")


if __name__ == "__main__":
    main()
