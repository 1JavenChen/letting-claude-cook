"""Sanity-check SMIRK FLAME labels against the MediaPipe labels (run with .venv-smirk).

    .venv-smirk\\Scripts\\python tools/smirk_check.py FLAME_NPZ MEDIAPIPE_NPZ [VIDEO] [--sheet out/x.jpg]

Prints: jaw opening vs MediaPipe jawOpen (correlation + best lag), how many FLAME
expression dims actually move, and how well each MediaPipe blendshape can be
linearly predicted from the FLAME params (held-out R^2). Compares on raw video
time, i.e. MediaPipe's `labels_unshifted` when the npz has been drift-corrected.
With VIDEO (and FLAME2020 installed) it also renders a contact sheet: video crop |
FLAME mesh | overlay, for a few frames, plus FLAME-vs-MediaPipe landmark error.
"""

import argparse
import os
import sys

import av
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from smirk_flame import CROP_SIZE, FLAME_NPZ, FLAME_PKL, FLAME_ZIP  # noqa: E402


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def best_lag(a, b, max_lag=10):
    """Lag (frames) maximising corr(a[t], b[t+lag])."""
    out = []
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            out.append((corr(a[: len(a) - L], b[L:]), L))
        else:
            out.append((corr(a[-L:], b[: len(b) + L]), L))
    return max(out)


def ridge_r2(X, y, lam=30.0):
    """Two-fold (first half / second half) ridge regression R^2, averaged."""
    n = len(X) // 2
    r2 = []
    for tr, te in ((slice(0, n), slice(n, None)), (slice(n, None), slice(0, n))):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = np.c_[(X[tr] - mu) / sd, np.ones(len(X[tr]))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred = np.c_[(X[te] - mu) / sd, np.ones(len(X[te]))] @ w
        r2.append(1 - ((y[te] - pred) ** 2).sum() / (((y[te] - y[te].mean()) ** 2).sum() + 1e-12))
    return float(np.mean(r2))


def contact_sheet(flame_npz, video, out_path, k=6):
    from smirk_flame import Flame, face_mask_vertices, project, render_mesh
    d = np.load(flame_npz)
    t_rel, P, ok, A = d["frame_times"], d["frame_params"], d["frame_face"], d["crop_affine"]
    start = float(d["start"])
    idx = np.where(ok)[0]
    # spread picks over the clip, preferring a range of jaw openings
    jaw = P[idx, 58]
    picks = sorted(set(idx[np.argsort(jaw)[np.linspace(0, len(idx) - 1, k).astype(int)]]))
    flame = Flame().cuda()
    faces_all = flame.faces.cpu().numpy()
    keep = np.zeros(flame.v_template.shape[0], bool)
    keep[face_mask_vertices()] = True
    faces = faces_all[keep[faces_all].all(1)]
    shape = torch.tensor(d["shape_mean"])[None].cuda()

    want = {round(float(t_rel[i]) + start, 4): i for i in picks}
    rows, errs = [], []
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    lmker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(ROOT, "models", "face_landmarker.task")),
        num_faces=3))
    with av.open(video) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, min(want) - 2) / s.time_base), stream=s)
        for fr in c.decode(s):
            t = round(float(fr.pts * s.time_base), 4)
            if t > max(want) + 0.01:
                break
            i = want.get(t)
            if i is None:
                continue
            rgb = fr.to_ndarray(format="rgb24")
            crop = cv2.warpAffine(rgb, A[i].astype(np.float64), (CROP_SIZE, CROP_SIZE))
            p = torch.tensor(P[i])[None].cuda()
            with torch.no_grad():
                v, lmk = flame(shape, p[:, 6:56], p[:, 0:3], p[:, 58:61], p[:, 56:58])
                px = project(v, p[:, 3:6])[0].cpu().numpy()
                lpx = project(lmk, p[:, 3:6])[0].cpu().numpy()
            mesh = render_mesh(crop, v[0].cpu().numpy(), px, faces)
            over = np.where(mesh.any(2, keepdims=True), (0.5 * crop + 0.5 * mesh).astype(np.uint8), crop)
            # FLAME's 105 MediaPipe-embedded points vs MediaPipe on the same crop
            r = lmker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if r.face_landmarks:
                L = max(r.face_landmarks, key=lambda L: (max(q.x for q in L) - min(q.x for q in L)))
                mpx = np.array([[q.x * CROP_SIZE, q.y * CROP_SIZE] for q in L])[flame.mp_indices]
                errs.append(np.linalg.norm(mpx - lpx, axis=1).mean())
                for x, y in mpx.astype(int):
                    cv2.circle(over, (x, y), 1, (0, 255, 0), -1)
                for x, y in lpx.astype(int):
                    cv2.circle(over, (x, y), 1, (255, 0, 0), -1)
            label = f"t={t:.2f}s jaw={P[i, 58]:.2f}"
            cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            rows.append(np.hstack([crop, mesh, over]))
    sheet = np.vstack(rows)
    cv2.imwrite(out_path, cv2.cvtColor(cv2.resize(sheet, None, fx=1.5, fy=1.5), cv2.COLOR_RGB2BGR))
    return out_path, (float(np.mean(errs)) if errs else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flame_npz")
    ap.add_argument("mp_npz")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--sheet", default=os.path.join(ROOT, "out", "smirk_check.jpg"))
    a = ap.parse_args()
    F = np.load(a.flame_npz)
    M = np.load(a.mp_npz)
    names = list(M["names"])
    lab = M["labels_unshifted"] if "labels_unshifted" in M else M["labels"]
    mok = M["face_ok_unshifted"] if "face_ok_unshifted" in M else M["face_ok"]
    s0 = int(F["start_frame"])
    n = min(len(F["face_ok"]), len(lab) - s0)
    lab, mok = lab[s0:s0 + n], mok[s0:s0 + n]
    ok = F["face_ok"][:n] & mok
    print(f"frames: {n} grid, {ok.sum()} with a face in both trackers")

    jaw = F["jaw"][:n, 0][ok]
    jo = lab[ok, names.index("jawOpen")]
    r, lag = best_lag(jaw, jo)
    print(f"\njaw opening (FLAME jaw[0]) vs MediaPipe jawOpen: r = {corr(jaw, jo):.3f} at lag 0, "
          f"best r = {r:.3f} at lag {lag:+d} frames (+ = MediaPipe later)")
    print(f"  FLAME jaw[0]: mean {jaw.mean():.3f} rad, std {jaw.std():.3f}, p5-p95 "
          f"{np.percentile(jaw, 5):.3f}-{np.percentile(jaw, 95):.3f}; zero (ReLU-clamped) in {np.mean(jaw <= 1e-6):.0%}")

    E = F["expression"][:n][ok]
    sd = E.std(0)
    order = np.argsort(-sd)
    var = sd ** 2
    print("\nFLAME expression dims (unit = PCA std of FLAME's training scans):")
    for th in (0.05, 0.1, 0.2, 0.5):
        print(f"  std > {th}: {(sd > th).sum()} of 50")
    print(f"  participation ratio (effective # dims) = {var.sum() ** 2 / (var ** 2).sum():.1f}")
    print(f"  dims for 90% / 99% of variance: {np.searchsorted(np.cumsum(np.sort(var)[::-1]) / var.sum(), 0.9) + 1}"
          f" / {np.searchsorted(np.cumsum(np.sort(var)[::-1]) / var.sum(), 0.99) + 1}")
    # Not all of that is motion: split each dim into frame-to-frame jitter and signal, using
    # raw video frames with valid neighbours: jitter var ~ var(x_t - (x_{t-1}+x_{t+1})/2) / 1.5
    RP, rok = F["frame_params"], F["frame_face"]
    trip = rok[1:-1] & rok[:-2] & rok[2:]
    raw = RP[:, 6:56]
    resid = (raw[1:-1] - (raw[:-2] + raw[2:]) / 2)[trip]
    jit = np.sqrt(resid.var(0) / 1.5)
    tot = raw[rok].std(0)
    sig = np.sqrt(np.clip(tot ** 2 - jit ** 2, 0, None))
    print(f"  per-frame jitter std: median {np.median(jit):.2f}; signal std (total minus jitter) > 0.2 in "
          f"{(sig > 0.2).sum()}, > 0.3 in {(sig > 0.3).sum()} dims; signal/jitter > 2 in {(sig > 2 * jit).sum()} dims")
    jj = RP[:, 58]
    jres = (jj[1:-1] - (jj[:-2] + jj[2:]) / 2)[trip]
    print(f"  jaw[0]: total std {jj[rok].std():.3f}, jitter std {np.sqrt(jres.var() / 1.5):.3f}")
    print("  top-10 std: " +", ".join(f"e{j}={sd[j]:.2f}" for j in order[:10]))
    print(f"  smallest std: {sd.min():.3f}; eyelid std {F['eyelid'][:n][ok].std(0).round(3)}, "
          f"jaw std {F['jaw'][:n][ok].std(0).round(3)}, head pose std (rad) {F['pose'][:n][ok].std(0).round(3)}")

    msd = lab[ok].std(0)
    print(f"\nMediaPipe on the same frames: {(msd > 0.02).sum()} of {len(names)} blendshapes with std > 0.02")

    X = np.c_[E, F["jaw"][:n][ok], F["eyelid"][:n][ok]]
    print("\nHow well FLAME params predict each MediaPipe shape with std > 0.01 (ridge, 2-fold held-out R^2):")
    rows = []
    for j, nm in enumerate(names):
        if nm == "_neutral" or msd[j] < 0.01:
            continue
        rows.append((ridge_r2(X, lab[ok, j]), nm, msd[j]))
    for r2, nm, s in sorted(rows, reverse=True):
        print(f"  {nm:20s} R2 {r2:+.2f}   (MediaPipe std {s:.3f})")

    if a.video:
        if not (os.path.exists(FLAME_NPZ) or os.path.exists(FLAME_PKL) or os.path.exists(FLAME_ZIP)):
            print(f"\n(no FLAME model at {FLAME_ZIP}: skipping mesh render)")
        else:
            path, err = contact_sheet(a.flame_npz, a.video, a.sheet)
            print(f"\ncontact sheet -> {path}")
            if err is not None:
                print(f"FLAME vs MediaPipe landmarks (105 pts) mean error: {err:.2f} px of {CROP_SIZE}")


if __name__ == "__main__":
    main()
