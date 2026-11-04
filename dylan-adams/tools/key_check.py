"""How well each answer key (or model) follows the mouth measured in 3D by the cameras.

    .venv\\Scripts\\python tools/key_check.py multicam_<date> data/raw/session_<...> [--keys labels_iph,labels_mha]
        [--models models/a.pt,...] [--held-out]

Referee: the inner-lip gap and chin drop (mm) triangulated from every camera that saw the face
(<session>/landmarks3d_multi.npz from tools/multicam_labels.py). It is independent of every key
except that it shares MediaPipe's landmark detector with labels_iph, so the fair comparison is how
far a key beats labels_iph on the same session. Scored on frames every key (and the referee) has:
  r gap / r chin   correlation of jawOpen with the lip gap / chin-to-nose distance
  shut@open        share of frames with lips > 10 mm apart where jawOpen < 0.05
  open@shut        share of frames with lips < 2 mm apart where jawOpen > 0.3
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("session")
    p.add_argument("--keys", default="labels_iph")
    p.add_argument("--models", default="")
    p.add_argument("--held-out", action="store_true", help="only the last 10%% of the clip (never trained on)")
    args = p.parse_args()

    d = np.load(os.path.join(ROOT, "data", "processed", args.clip + ".npz"))
    names = list(d["names"])
    jaw = names.index("jawOpen")
    grid = d["grid_t"]
    L = np.load(os.path.join(args.session, "landmarks3d_multi.npz"))
    j = np.clip(np.searchsorted(L["times"], grid), 0, len(L["times"]) - 1)
    near = np.abs(L["times"][j] - grid) < 0.02
    P = L["points"][j]
    gap = np.linalg.norm(P[:, 13] - P[:, 14], axis=1)
    chin = np.linalg.norm(P[:, 152] - P[:, 1], axis=1)
    use = near & L["good"][j] & np.isfinite(gap) & np.isfinite(chin)

    series = {}
    for k in args.keys.split(","):
        series[k] = d[k][:, jaw]
        use &= d[k.replace("labels", "face_ok")]
    if args.models:
        from compare_models import run_model
        feats = np.load(os.path.join(ROOT, "data", "features", "wav2vec2", args.clip + ".npy")).astype(np.float32)
        for m in args.models.split(","):
            series[os.path.basename(m)[:-3]] = run_model(m, d, feats)[:, jaw]
        use &= d["face_ok_meas_lo"]
    series["webcam alone"] = d["labels_meas_lo"][:, jaw]
    if args.held_out:
        use &= np.arange(len(grid)) >= int(0.9 * len(grid))

    g, c = gap[use], chin[use]
    print(f"{args.clip}: {use.sum()} frames ({use.sum() / 30 / 60:.1f} min); lips > 10 mm apart on "
          f"{np.mean(g > 10):.0%}, < 2 mm on {np.mean(g < 2):.0%}")
    print(f"  {'':24s} r gap  r chin  shut@open  open@shut")
    for k, s in series.items():
        s = s[use]
        print(f"  {k:24s} {np.corrcoef(s, g)[0, 1]:.2f}   {np.corrcoef(s, c)[0, 1]:.2f}    "
              f"{np.mean(s[g > 10] < 0.05):5.1%}     {np.mean(s[g < 2] > 0.3):5.1%}")


if __name__ == "__main__":
    main()
