"""Apple's ARKit face tracking from a Live Link (ARKit) take -> labels on a training clip.

    .venv\\Scripts\\python tools/arkit_labels.py data/raw/session_<...> --clip multicam_<date>

Live Link Face in ARKit mode saves Apple's 52 blendshapes per frame (TrueDepth tracking, the face
mesh in the app's preview): *_raw.csv, and *_cal.csv = raw minus the neutral pose captured in the
app (rescaled so 1 stays 1). The frame log's B rows time each CSV row (tools/iphone_take.py ->
iphone_times.npz arkit_t, on the session clock).
1. *_cal.csv resampled onto the clip's 30 fps grid (only between two ARKit frames < 50 ms apart).
2. Frames dropped (face_ok_arkit false): no face in the iPhone view, a hand on the face
   (tools/hands_iphone.py, if run), head yaw beyond --max-yaw (from the iPhone landmarks).
3. labels_arkit / face_ok_arkit (+ _lo: the front-webcam input) added to data/processed/<clip>.npz.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def head_yaw(lmk):
    """Head yaw (deg) from MediaPipe landmarks [N,468,3] in pixels (z on the x scale): the eye-corner line."""
    v = lmk[:, 263] - lmk[:, 33]
    return np.degrees(np.arctan2(v[:, 2], v[:, 0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--clip", required=True)
    p.add_argument("--max-yaw", type=float, default=40, help="deg (same limit as the MHA key)")
    p.add_argument("--hand-cover", type=float, default=0.1)
    p.add_argument("--swap-lr", action="store_true", help="ARKit Left/Right are the other side of ours")
    args = p.parse_args()

    with open(os.path.join(args.session, "iphone.json")) as f:
        info = json.load(f)
    path = glob.glob(os.path.join(os.path.dirname(info["mov"]), "*_cal.csv"))[0]
    rows = list(csv.reader(open(path)))
    head = rows[0][2:]
    vals = np.array([[float(x) for x in r[2:2 + len(head)]] for r in rows[1:]], np.float32)
    at = np.load(os.path.join(args.session, "iphone_times.npz"))["arkit_t"]
    assert len(at) == len(vals), f"frame log has {len(at)} ARKit frames, the CSV {len(vals)}"

    clip_path = os.path.join(ROOT, "data", "processed", args.clip + ".npz")
    Z = dict(np.load(clip_path))
    names = [str(n) for n in Z["names"]]
    grid = Z["grid_t"]
    cols = []
    for n in names:
        src = n[0].upper() + n[1:]
        if args.swap_lr:
            src = src.replace("Left", "#").replace("Right", "Left").replace("#", "Right")
        cols.append(head.index(src))
    hi = np.clip(np.searchsorted(at, grid), 1, len(at) - 1)
    lo = hi - 1
    f = np.clip((grid - at[lo]) / np.maximum(at[hi] - at[lo], 1e-6), 0, 1)[:, None]
    labels = (vals[lo][:, cols] * (1 - f) + vals[hi][:, cols] * f).astype(np.float32)
    ok = (at[hi] - at[lo] < 0.05) & (grid >= at[0]) & (grid <= at[-1])
    print(f"{len(grid)} grid moments, ARKit frames at {ok.mean():.1%}")

    ok &= Z["face_ok_iph"]
    tr = np.load(os.path.join(args.session, "track_iphone.npz"))
    near = lambda t: np.clip(np.searchsorted(t, grid), 0, len(t) - 1)  # noqa: E731
    yaw = np.abs(head_yaw(tr["lmk"]))[near(tr["t"])]
    hp = os.path.join(args.session, "hands_iphone.npz")
    if os.path.exists(hp):
        hands = np.load(hp)
        cover = np.convolve((hands["cover"] > args.hand_cover).astype(float), np.ones(19), "same") > 0  # +-0.3 s
        hand = cover[near(hands["t"])]
    else:
        print("  (no hands_iphone.npz: hand-on-face frames kept)")
        hand = np.zeros(len(grid), bool)
    print(f"  dropped: no iPhone face {np.mean(~Z['face_ok_iph']):.1%}, hand on face {np.mean(ok & hand):.1%}, "
          f"yaw > {args.max_yaw:.0f} deg {np.mean(ok & (yaw > args.max_yaw)):.1%}")
    ok &= ~hand & (yaw <= args.max_yaw)
    labels[~ok] = 0
    print(f"  face_ok_arkit {ok.mean():.1%}")

    Z.update(labels_arkit=labels, face_ok_arkit=ok, labels_arkit_lo=Z["labels_iph_lo"],
             face_ok_arkit_lo=Z["face_ok_iph_lo"])
    np.savez_compressed(clip_path, **Z)
    print("shape              arkit mean/p95   iph mean/p95   r(arkit, iph)")
    both = ok & Z["face_ok_iph"]
    for n in ("jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthSmileLeft", "mouthFrownLeft",
              "mouthStretchLeft", "mouthRollLower", "mouthPressLeft", "eyeBlinkLeft", "browInnerUp", "browDownLeft"):
        i = names.index(n)
        a, b = labels[both, i], Z["labels_iph"][both, i]
        print(f"  {n:16s} {a.mean():.2f}/{np.percentile(a, 95):.2f}       {b.mean():.2f}/{np.percentile(b, 95):.2f}"
              f"      {np.corrcoef(a, b)[0, 1]:.2f}")
    print(f"-> {clip_path}")


if __name__ == "__main__":
    main()
