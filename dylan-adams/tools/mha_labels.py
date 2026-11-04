"""MetaHuman Animator solves of the iPhone depth take -> ARKit-style labels on a training clip.

    .venv\\Scripts\\python tools/mha_labels.py data/raw/session_<...> --clip multicam_<date>

Inputs (unreal/VRFace/Saved, from tools/unreal/mha_chunks.py and mha_take_gui.py):
  vrface_mha_depth_<start>.json  depth solves (the answer key), keyed from their start video frame
  vrface_mha_mono_full.json      video-only solve of the whole take (fills the depth gaps)
  vrface_arkit_mapping.json      Epic's ARKit -> MetaHuman pose asset (tools/unreal/dump_arkit_mapping.py)
1. Each clip grid moment -> iPhone video frame (iphone_times.npz) -> MetaHuman controls
   (CTRL_expressions_*) from the depth solve; the mono solve where depth has no frame, or where
   depth clamps the jaw shut while mono sees it open (depth loses the mouth on strong profiles), or
   while the depth-measured lips are > 10 mm apart (depth_labels.npz from tools/depth_labels.py).
2. Controls -> ARKit weights: Epic's mapping gives the controls each ARKit shape drives at 1.0;
   invert it per frame with bounded least squares (weights 0..1). mouthClose isn't in the mapping:
   lips-together amount, capped by jawOpen (as ARKit defines it).
3. Frames dropped (face_ok_mha false) where MHA was seen to fail: a hand on the face
   (tools/hands_iphone.py), head yaw beyond --max-yaw.
   Brows come from --brows-from (MHA reads hair over the forehead as a lowered brow).
4. labels_mha / face_ok_mha (+ _lo: the front-webcam input) added to data/processed/<clip>.npz.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.optimize import lsq_linear

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED = os.path.join(ROOT, "unreal", "VRFace", "Saved")
VFPS = 60  # iPhone video frames per second (the MHA processing frames)
PREFIX = "ctrl_expressions_"


def load_solve(path):
    with open(path) as f:
        d = json.load(f)
    curves = {k.lower(): v for k, v in d["curves"].items() if k.lower().startswith(PREFIX)}
    span = max((max(v[0]) for v in curves.values() if v[0]), default=0)
    return d["request"]["start"], span, curves


def sample(curves, ctrls, t):
    """[len(t), len(ctrls)] control values at curve times t (an empty curve is constant 0)."""
    out = np.zeros((len(t), len(ctrls)), np.float32)
    for j, c in enumerate(ctrls):
        v = curves.get(c)
        if v and v[0]:
            out[:, j] = np.interp(t, v[0], v[1])
    return out


def controls(solves, ctrls, frame):
    """Controls at (fractional) video frames from a list of solves; first solve covering a frame wins."""
    X = np.zeros((len(frame), len(ctrls)), np.float32)
    have = np.zeros(len(frame), bool)
    for start, span, curves in solves:
        m = ~have & (frame >= start) & (frame <= start + span * VFPS)
        if m.any():
            X[m] = sample(curves, ctrls, (frame[m] - start) / VFPS)
            have |= m
    return X, have


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--clip", required=True, help="data/processed/<clip>.npz to add labels_mha to")
    p.add_argument("--jaw-fill", type=float, default=0.1, help="mono jawOpen above which a depth jaw of 0 is a drop-out")
    p.add_argument("--max-yaw", type=float, default=40, help="deg: MHA loses the far side of the face beyond this")
    p.add_argument("--hand-cover", type=float, default=0.1, help="share of the face box under a hand that drops a frame")
    p.add_argument("--brows-from", default="labels_iph", help="label set to take the brow shapes from ('' = MHA)")
    args = p.parse_args()

    with open(os.path.join(SAVED, "vrface_arkit_mapping.json")) as f:
        mp = json.load(f)
    poses = [s.lower() for s in mp["poses"]]
    clip_path = os.path.join(ROOT, "data", "processed", args.clip + ".npz")
    Z = dict(np.load(clip_path))
    names = [str(n) for n in Z["names"]]
    solved = [n for n in names if n.lower() in poses]  # all but mouthClose
    ctrls = sorted(c for c in mp["curves"] if c.startswith(PREFIX))
    pose_t = lambda name: poses.index(name.lower()) / 24.0  # one pose per frame at 24 fps  # noqa: E731
    base = sample(mp["curves"], ctrls, np.array([pose_t("Default")]))[0]
    A = np.stack([sample(mp["curves"], ctrls, np.array([pose_t(n)]))[0] - base for n in solved], 1)  # [C, S]
    rows = np.abs(A).max(1) > 1e-4  # controls some ARKit shape drives
    print(f"mapping: {len(solved)} ARKit shapes -> {rows.sum()} of {len(ctrls)} MetaHuman controls")

    # grid moments -> fractional iPhone video frames
    vt = np.load(os.path.join(args.session, "iphone_times.npz"))["video_t"]
    frame = np.interp(Z["grid_t"], vt, np.arange(len(vt)))
    depth = []
    for f in glob.glob(os.path.join(SAVED, "vrface_mha_depth_*.json")):
        if re.search(r"depth_\d+\.json$", f):
            depth.append(load_solve(f))
    depth.sort(key=lambda s: s[0])
    Xd, hd = controls(depth, ctrls, frame)
    Xm, hm = controls([load_solve(os.path.join(SAVED, "vrface_mha_mono_full.json"))], ctrls, frame)
    jaw = ctrls.index(PREFIX + "jawopen")
    grid = Z["grid_t"]
    near = lambda t: np.clip(np.searchsorted(t, grid), 0, len(t) - 1)  # noqa: E731
    dl = np.load(os.path.join(args.session, "depth_labels.npz"))
    jd = near(dl["times"])
    gap = np.linalg.norm(dl["landmarks_mm"][jd, 13] - dl["landmarks_mm"][jd, 14], axis=1)
    # depth drop-outs: the jaw clamped shut while mono sees it open, or shut on lips measured > 10 mm
    # apart (mostly a slightly open jaw while talking or laughing, which mono reads better)
    drop = hd & hm & (((Xd[:, jaw] <= 0) & (Xm[:, jaw] > args.jaw_fill)) |
                      ((gap > 10) & (Xd[:, jaw] < 0.05) & (Xm[:, jaw] > Xd[:, jaw])))
    use_mono = (~hd | drop) & hm
    X = np.where(use_mono[:, None], Xm, Xd)
    have = hd | hm
    ok = have & Z["face_ok_iph"]  # MHA always answers; trust it only where the iPhone saw a face
    print(f"{len(frame)} grid moments: depth {np.mean(hd & ~drop):.1%}, mono fill {np.mean(use_mono):.1%} "
          f"(depth drop-outs {drop.mean():.1%}), none {np.mean(~have):.1%}; iPhone face {ok.mean():.1%}")

    # frames MHA gets wrong (frame reviews): a hand on the face (read as jaw/brow/frown at 1.0), strong profiles
    hands = np.load(os.path.join(args.session, "hands_iphone.npz"))
    cover = np.convolve((hands["cover"] > args.hand_cover).astype(float), np.ones(19), "same") > 0  # +-0.3 s
    hand = cover[near(hands["t"])]
    yaw = np.abs(np.degrees(dl["head"][jd, 0]))
    print(f"  dropped: hand on face {np.mean(ok & hand):.1%}, yaw > {args.max_yaw:.0f} deg "
          f"{np.mean(ok & (yaw > args.max_yaw)):.1%}")
    ok &= ~hand & (yaw <= args.max_yaw)
    print(f"  face_ok_mha {ok.mean():.1%}")

    # controls -> ARKit weights
    W = np.zeros((len(frame), len(solved)), np.float32)
    Ar, Xr = A[rows], X[:, rows] - base[rows]
    res = np.zeros(len(frame))
    for i in np.flatnonzero(have):
        r = lsq_linear(Ar, Xr[i], bounds=(0, 1), method="bvls")
        W[i] = r.x
        res[i] = np.linalg.norm(Ar @ r.x - Xr[i]) / max(np.linalg.norm(Xr[i]), 1e-6)
    print(f"fit: median relative residual {np.median(res[ok]):.2f} (p90 {np.percentile(res[ok], 90):.2f})")

    labels = np.zeros((len(frame), len(names)), np.float32)
    for j, n in enumerate(solved):
        labels[:, names.index(n)] = W[:, j]
    together = np.mean([X[:, ctrls.index(PREFIX + f"mouthlipstogether{s}")] for s in ("ul", "ur", "dl", "dr")], 0)
    labels[:, names.index("mouthClose")] = np.clip(np.minimum(together, labels[:, names.index("jawOpen")]), 0, 1)
    if args.brows_from:  # MHA's brows fail under hair over the forehead (browDown at 1.0 on a neutral brow)
        brow = [i for i, n in enumerate(names) if n.startswith("brow")]
        labels[:, brow] = Z[args.brows_from][:, brow]
    labels[~ok] = 0

    Z.update(labels_mha=labels, face_ok_mha=ok, labels_mha_lo=Z["labels_iph_lo"], face_ok_mha_lo=Z["face_ok_iph_lo"],
             mha_source=np.where(use_mono, 1, np.where(hd, 0, -1)).astype(np.int8))
    np.savez_compressed(clip_path, **Z)
    print("shape              mha mean/p95    iph mean/p95   r(mha, iph)")
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
