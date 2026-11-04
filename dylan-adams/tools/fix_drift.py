"""Measure and remove slow audio/video drift in edited videos.

Some videos drift out of sync over time (e.g. 23.976 fps camera footage
conformed to a 24 fps timeline, or audio recorded on a separate device). The
drift shows up as the lip-to-sound offset growing steadily through the video.

This tool:
  1. estimates the offset in each block of continuous face tracking (where
     fast mouth movement best matches fast loudness changes, +/-1 s),
  2. smooths those estimates over time (rolling median, weighted towards
     confident blocks) into an offset curve,
  3. resamples the labels so the offset everywhere equals the natural lead of
     mouth over sound (`--lead`, default 50 ms).

    python tools/fix_drift.py data/processed/<name>.npz

Rewrites the .npz (originals of shifted arrays are kept as *_unshifted) and
saves <name>_drift.png. Run before sync_segments.py; it undoes any earlier
sync_segments.py filtering, so run that again afterwards.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import FPS, audio_db, rolling_mean  # noqa: E402
from sync_segments import shots  # noqa: E402

SHIFTED = ["labels", "labels_lo", "face_ok", "face_ok_lo", "keep", "mouth_px"]


def block_offsets(jaw, db, sound, face_ok, mouth_px, block_s, max_lag_s=1.0, around=None):
    """[(centre_s, lag_s, peak_corr)] per block of continuous tracking.

    around: optional function t -> expected lag (s); then only lags within
    +/-max_lag_s of it are searched."""
    MAX_LAG = int(np.ceil((max_lag_s + (3.0 if around else 0)) * FPS))
    j = jaw - rolling_mean(jaw, FPS)
    a = db - rolling_mean(db, FPS)
    out = []
    blk = int(block_s * FPS)
    for s, e in shots(face_ok, mouth_px):
        for b0 in range(s, e, blk):
            b1 = min(e, b0 + blk)
            idx = np.arange(b0, b1)[sound[b0:b1]]
            idx = idx[(idx >= MAX_LAG) & (idx < len(j) - MAX_LAG)]
            if len(idx) < 3 * FPS:
                continue
            c0 = around((b0 + b1) / 2 / FPS) * FPS if around else 0
            lags = np.arange(int(round(c0 - max_lag_s * FPS)), int(round(c0 + max_lag_s * FPS)) + 1)
            cs = np.array([np.corrcoef(j[idx + l], a[idx])[0, 1] for l in lags])
            k = int(np.nanargmax(cs))
            out.append(((b0 + b1) / 2 / FPS, lags[k] / FPS, cs[k]))
    return np.array(out)


def smooth_curve(points, n, window_s, min_corr):
    """Offset (s) for every frame: weighted rolling median of confident blocks."""
    t, lag, c = points.T
    good = c >= min_corr
    t, lag, c = t[good], lag[good], c[good]
    centres = np.arange(0, n / FPS + 10, 10.0)
    vals = []
    for x in centres:
        near = np.abs(t - x) <= window_s / 2
        if near.sum() < 3:
            vals.append(np.nan)
            continue
        o = np.argsort(lag[near])
        cw = np.cumsum(c[near][o])
        vals.append(lag[near][o][np.searchsorted(cw, cw[-1] / 2)])
    vals = np.array(vals)
    ok = ~np.isnan(vals)
    curve = np.interp(np.arange(n) / FPS, centres[ok], vals[ok])
    return curve, (t, lag, c)


def fit_line(t, lag, c, iters=5, tol=0.12):
    """Robust straight-line fit lag = a + b*t (weighted by corr, outliers trimmed)."""
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        b, a = np.polyfit(t[keep], lag[keep], 1, w=c[keep])
        keep = np.abs(lag - (a + b * t)) < tol
        if keep.sum() < 5:
            break
    return a, b, keep


def ransac_line(t, lag, c, tol=0.15, tries=2000, seed=0):
    """Line most confident blocks agree with: try random pairs, score inlier weight."""
    rng = np.random.default_rng(seed)
    best, best_score = (0.0, 0.0), -1
    for _ in range(tries):
        i, k = rng.choice(len(t), 2, replace=False)
        if abs(t[k] - t[i]) < 60:
            continue
        b = (lag[k] - lag[i]) / (t[k] - t[i])
        a = lag[i] - b * t[i]
        score = c[np.abs(lag - (a + b * t)) < tol].sum()
        if score > best_score:
            best, best_score = (a, b), score
    return best


def linear_curve(jaw, db, sound, face_ok, mouth_px, n, block_s, min_corr):
    """Drift as a straight line: fit on blocks found with a wide (+/-3 s) search,
    then refine each block within +/-0.3 s of the line and fit again."""
    pts = block_offsets(jaw, db, sound, face_ok, mouth_px, block_s, max_lag_s=3.0)
    good = pts[:, 2] >= min_corr
    a, b = ransac_line(*pts[good].T)
    ref = block_offsets(jaw, db, sound, face_ok, mouth_px, block_s, max_lag_s=0.3,
                        around=lambda x: a + b * x)
    good = ref[:, 2] >= min_corr
    a, b, inl = fit_line(*ref[good].T)
    t, lag, c = ref[good].T
    print(f"  linear drift: {a:+.3f} s at start, {b * 1000:+.3f} ms per second "
          f"({inl.sum()} of {len(t)} confident blocks within 0.12 s of the line)")
    return a + b * np.arange(n) / FPS, (t[inl], lag[inl], c[inl]), pts


def constant_offset(jaw, db, sound, face_ok, n, min_corr, lo=-0.6, hi=0.4):
    """One A/V offset for a whole (short, uncut) clip: the peak of the clip's sync
    curve within [lo, hi] s. Returns (offset or None, peak corr, contrast)."""
    j = jaw - rolling_mean(jaw, FPS)
    a = db - rolling_mean(db, FPS)
    far = int(2 * FPS)
    idx = np.nonzero(sound & face_ok)[0]
    idx = idx[(idx >= far) & (idx < n - far)]
    if len(idx) < 10 * FPS:
        return None, np.nan, np.nan
    lags = np.arange(-far, far + 1)
    cs = np.array([np.corrcoef(j[idx + l], a[idx])[0, 1] for l in lags])
    win = (lags >= lo * FPS) & (lags <= hi * FPS)
    k = np.nonzero(win)[0][int(np.nanargmax(cs[win]))]
    contrast = (cs[k] - np.nanmedian(cs)) / (np.nanstd(cs) + 1e-9)
    ok = cs[k] >= min_corr and contrast >= 2.5
    return (lags[k] / FPS if ok else None), cs[k], contrast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--block", type=float, default=15, help="seconds per offset estimate")
    p.add_argument("--window", type=float, default=180, help="smoothing window in seconds")
    p.add_argument("--min-corr", type=float, default=0.2, help="ignore blocks below this peak")
    p.add_argument("--lead", type=float, default=-0.05, help="target offset (mouth leads sound)")
    p.add_argument("--mode", default="linear", choices=["linear", "smooth", "constant"],
                   help="linear: one clock-rate drift for the whole video (one continuous "
                   "recording); smooth: rolling median, for videos cut from several takes; "
                   "constant: one fixed offset for a short uncut clip (e.g. public datasets), "
                   "left unchanged when the clip has no confident sync peak")
    args = p.parse_args()

    d = dict(np.load(args.npz))
    if "keep_raw" in d:  # undo sync_segments.py filtering; it must be redone after the shift
        d["keep"] = d.pop("keep_raw")
    if "labels_unshifted" in d:  # already corrected once: start from the originals
        for k in SHIFTED:
            if k + "_unshifted" in d:
                d[k] = d[k + "_unshifted"]
    names = list(d["names"])
    n = len(d["labels"])
    jaw = d["labels"][:, names.index("jawOpen")]
    db = audio_db(d["audio"], n)
    sound = db > np.percentile(db, 95) - 25
    mouth_px = np.nan_to_num(d.get("mouth_px", np.full(n, 100.0)), nan=1.0)

    if args.mode == "constant":
        off, pc, con = constant_offset(jaw, db, sound, d["face_ok"], n, args.min_corr)
        print(f"  constant offset: " + (f"{off:+.2f} s" if off is not None else "none confident (unchanged)")
              + f" (peak corr {pc:+.2f}, stands out {con:.1f} sd)")
        curve = np.full(n, args.lead if off is None else off)
        pts = np.array([[n / 2 / FPS, curve[0], pc if np.isfinite(pc) else 0.0]])
        t, lag, c = pts.T
    elif args.mode == "linear":
        curve, (t, lag, c), pts = linear_curve(jaw, db, sound, d["face_ok"], mouth_px, n,
                                               args.block, args.min_corr)
    else:
        pts = block_offsets(jaw, db, sound, d["face_ok"], mouth_px, args.block)
        curve, (t, lag, c) = smooth_curve(pts, n, args.window, args.min_corr)

    # labels_new[i] = labels[i + curve[i] - lead]  (in frames)
    src = np.arange(n) + (curve - args.lead) * FPS
    for k in SHIFTED:
        if k not in d:
            continue
        x = d[k]
        d[k + "_unshifted"] = x
        if x.dtype == bool:
            d[k] = x[np.clip(np.round(src).astype(int), 0, n - 1)]
        elif x.ndim == 1:
            d[k] = np.interp(src, np.arange(n), x).astype(x.dtype)
        else:
            d[k] = np.stack([np.interp(src, np.arange(n), x[:, i]) for i in range(x.shape[1])],
                            1).astype(x.dtype)
    edge = (src < 0) | (src > n - 1)
    d["keep"] = d["keep"] & ~edge
    d["drift_s"] = curve.astype(np.float32)
    d["drift_src"] = src.astype(np.float32)  # lets later passes (degrade_track.py) apply the same shift
    np.savez_compressed(args.npz, **d)
    if args.mode == "constant":
        print(f"{os.path.basename(args.npz)}: shifted labels by {curve[0] - args.lead:+.2f} s "
              f"(offset {curve[0]:+.2f} s -> {args.lead:+.2f} s)")
        return

    after = block_offsets(d["labels"][:, names.index("jawOpen")], db, sound, d["face_ok"],
                          np.nan_to_num(d.get("mouth_px", np.full(n, 100.0)), nan=1.0), args.block)
    ga = after[:, 2] >= args.min_corr
    print(f"{os.path.basename(args.npz)}: offset ranged {curve.min():+.2f} to {curve.max():+.2f} s "
          f"({len(t)} confident blocks). After correction: median block offset "
          f"{np.median(after[ga, 1]):+.2f} s, median peak corr "
          f"{np.median(pts[:, 2]):.2f} -> {np.median(after[:, 2]):.2f}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(pts[:, 0], pts[:, 1], s=8 + 60 * np.clip(pts[:, 2], 0, 1), c="0.6", label="before (all blocks)")
    ax.scatter(t, lag, s=8 + 60 * c, c="tab:blue", label="before (confident)")
    ax.plot(np.arange(n) / FPS, curve, c="tab:red", label="estimated drift")
    ax.scatter(after[:, 0], after[:, 1], s=8 + 60 * np.clip(after[:, 2], 0, 1), c="tab:green",
               marker="x", label="after correction")
    ax.axhline(args.lead, ls="--", c="k", lw=0.6)
    ax.set_xlabel("seconds")
    ax.set_ylabel("lip-to-sound offset (s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.splitext(args.npz)[0] + "_drift.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
