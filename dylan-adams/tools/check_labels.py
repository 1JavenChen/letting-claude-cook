"""Label-quality check: do the mouth labels follow the speech frame by frame?

Run this on a processed clip BEFORE spending time on training. It measures how
well fast jaw movement (slow drift removed) lines up with fast changes in
loudness while the person is talking, across time shifts of +/-1 s.

Good labels show a clear peak near 0 s (mouths lead sound by ~0-0.15 s).
Noise-level labels (e.g. tiny or heavily compressed faces) show no peak.

    python tools/check_labels.py data/processed/<name>.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import FPS, audio_db, rolling_mean  # noqa: E402


def sync_curve(jaw, db, mask, max_lag_s=1.0):
    j = jaw - rolling_mean(jaw, FPS)  # remove posture drift slower than ~1 s
    a = db - rolling_mean(db, FPS)
    lags = np.arange(-int(max_lag_s * FPS), int(max_lag_s * FPS) + 1)
    i = np.nonzero(mask)[0]
    i = i[(i > lags.max()) & (i < len(jaw) - lags.max())]
    return lags / FPS, np.array([np.corrcoef(j[i + l], a[i])[0, 1] for l in lags])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    args = p.parse_args()
    for path in args.npz:
        d = np.load(path)
        names = list(d["names"])
        n = len(d["labels"])
        jaw = d["labels"][:, names.index("jawOpen")]
        db = audio_db(d["audio"], n)
        if "voice_owner" in d:
            me = str(d["role"]) if "role" in d else ("a" if path.endswith("top.npz") else "b")
            mask = d["keep"] & (d["voice_owner"][:n] == me)
        else:
            mask = d["keep"] & (db > np.percentile(db, 95) - 20)
        lags, cs = sync_curve(jaw, db, mask)
        k = int(np.argmax(cs))
        # How far the peak stands out from the typical correlation at other shifts.
        contrast = (cs[k] - np.median(cs)) / (np.std(cs) + 1e-9)
        good = cs[k] >= 0.15 and abs(lags[k]) <= 0.35 and contrast >= 3
        print(f"{os.path.basename(path)}: peak corr {cs[k]:+.2f} at {lags[k]:+.2f}s "
              f"(stands out {contrast:.1f} sd) over {mask.sum() / FPS / 60:.1f} min of speech "
              f"-> {'GOOD: labels follow speech' if good else 'POOR: labels do not follow speech'}")


if __name__ == "__main__":
    main()
