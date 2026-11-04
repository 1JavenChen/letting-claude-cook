"""Keep only the shots where the face on screen is the one talking.

Edited videos mix talking-head shots with b-roll, timelapses and voiceover.
Face tracking alone can't tell them apart: a timelapse of you still has your
face in it, but the audio isn't coming from that mouth. This tool:

  1. splits the face-tracked frames into shots (a cut = the face disappears or
     its size jumps),
  2. scores each shot by how well fast jaw movement follows loudness while
     there is sound (the same measure as check_labels.py, near-zero lags only),
  3. keeps shots that are clearly in sync and whose mouth is big enough.

Shots with too little sound to judge are dropped, since a silent timelapse
would teach the model "silence = random mouth movement".

    python tools/sync_segments.py data/processed/<name>*.npz [--min-corr 0.25] [--exclude 0-17]

Rewrites `keep` in each .npz (the original is saved once as `keep_raw`) and
writes a <name>_shots.png timeline.
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

LAGS = np.arange(-int(0.3 * FPS), int(0.1 * FPS) + 1)  # mouth leads sound by 0-0.3 s
FAR = np.r_[-2 * FPS:-FPS, FPS:2 * FPS]  # lags where no real relationship exists


def shots(face_ok, mouth_px, jump=0.25, min_len=2 * FPS):
    """[(start, end)] runs of tracked frames without a cut."""
    out, s = [], None
    for i in range(len(face_ok)):
        cut = s is not None and (not face_ok[i] or
                                 abs(mouth_px[i] / mouth_px[i - 1] - 1) > jump)
        if s is not None and (cut or not face_ok[i]):
            if i - s >= min_len:
                out.append((s, i))
            s = None
        if s is None and face_ok[i]:
            s = i
    if s is not None and len(face_ok) - s >= min_len:
        out.append((s, len(face_ok)))
    return out


def corr_at(j, a, idx, lag):
    i = idx[(idx + lag >= 0) & (idx + lag < len(j))]
    if len(i) < FPS:
        return np.nan
    return np.corrcoef(j[i + lag], a[i])[0, 1]


def score(jaw, db, sound, a, b):
    """(sync corr, chance-level corr, seconds of sound) for frames a:b."""
    j = jaw - rolling_mean(jaw, FPS)
    d = db - rolling_mean(db, FPS)
    idx = np.arange(a, b)[sound[a:b]]
    if len(idx) < 2 * FPS:
        return np.nan, np.nan, len(idx) / FPS
    near = np.nanmax([corr_at(j, d, idx, l) for l in LAGS])
    far = np.nanpercentile(np.abs([corr_at(j, d, idx, l) for l in FAR[::3]]), 75)
    return near, far, len(idx) / FPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--min-corr", type=float, default=0.25)
    p.add_argument("--min-mouth", type=float, default=60, help="median mouth width in px")
    p.add_argument("--exclude", help="time ranges in seconds to always drop, e.g. 0-17,300-310 "
                   "(someone else on camera)")
    p.add_argument("--block", type=float, default=20,
                   help="long shots are scored in blocks of this many seconds")
    args = p.parse_args()

    for path in args.npz:
        d = dict(np.load(path))
        names = list(d["names"])
        jaw = d["labels"][:, names.index("jawOpen")]
        n = len(jaw)
        db = audio_db(d["audio"], n)
        sound = db > np.percentile(db, 95) - 25
        face_ok = d["face_ok"]
        mouth_px = np.nan_to_num(d["mouth_px"], nan=1.0) if "mouth_px" in d else np.full(n, 100.0)
        keep_raw = d.get("keep_raw", d["keep"])

        keep = np.zeros(n, bool)
        rows = []
        blk = int(args.block * FPS)
        for s, e in shots(face_ok, mouth_px):
            for a in range(s, e, blk):
                b = min(e, a + blk)
                if b - a < 2 * FPS and a > s:  # fold a short tail into the previous block
                    continue
                if e - b < 2 * FPS:
                    b = e
                near, far, secs = score(jaw, db, sound, a, b)
                mpx = float(np.median(mouth_px[a:b]))
                ok = near >= args.min_corr and near > far + 0.05 and mpx >= args.min_mouth
                keep[a:b] = ok
                rows.append((a, b, near, far, secs, mpx, ok))
        keep &= keep_raw
        for r in (args.exclude.split(",") if args.exclude else []):
            a, b = (float(v) for v in r.split("-"))
            keep[int(a * FPS):int(b * FPS)] = False

        d["keep_raw"] = keep_raw
        d["keep"] = keep
        np.savez_compressed(path, **d)

        kept = [r for r in rows if r[6]]
        print(f"{os.path.basename(path)}: {len(rows)} blocks, kept {len(kept)} | "
              f"{keep.sum() / FPS / 60:.1f} of {n / FPS / 60:.1f} min kept "
              f"({(keep & sound).sum() / FPS / 60:.1f} min with sound)")

        t = np.arange(n) / FPS
        fig, ax = plt.subplots(2, 1, figsize=(16, 4.5), sharex=True)
        ax[0].plot(t, mouth_px, lw=0.5, color="0.4")
        ax[0].set_ylabel("mouth px")
        for a, b, near, far, secs, mpx, ok in rows:
            ax[1].plot([a / FPS, b / FPS], [near, near], color="tab:green" if ok else "tab:red", lw=2)
            ax[1].plot([a / FPS, b / FPS], [far, far], color="0.6", lw=1)
        ax[1].axhline(args.min_corr, ls="--", color="k", lw=0.6)
        ax[1].set_ylabel("sync corr")
        ax[1].set_xlabel("seconds")
        ax[0].fill_between(t, 0, 1, where=keep, transform=ax[0].get_xaxis_transform(),
                           color="tab:green", alpha=0.2, lw=0)
        fig.suptitle(f"{os.path.basename(path)} - green = kept (sync >= {args.min_corr}), "
                     f"grey = chance level")
        fig.tight_layout()
        fig.savefig(os.path.splitext(path)[0] + "_shots.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    main()
