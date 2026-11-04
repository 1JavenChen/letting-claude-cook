"""Measured face labels from a calibrated 3-camera session, packaged as a training clip.

    .venv\\Scripts\\python tools/multicam_labels.py data/raw/session_<...> --name multicam_<date>

Needs calib.json (tools/calibrate.py), iphone.json + iphone_times.npz (tools/iphone_take.py)
and track_*.npz (tools/track_views.py).

1. Triangulate: at each iPhone frame (30 fps, the time reference) the webcams' landmarks are
   interpolated to the same true moment (their stamps minus each camera's measured delay),
   undistorted (webcam stamps first rebuilt from the steady frame rate: they arrive in bursts),
   and every landmark is triangulated from all cameras that saw the face. A
   camera that disagrees with the others (median re-projection > 6 px) is dropped.
2. Measure: the geometric retargeter (realtime/retarget.py) fits ARKit-style weights to the
   true 3D points - every shape from geometry except gaze; minus the resting face.
3. Package (data/processed/<name>.npz), on a 30 fps grid of true time:
     audio            session mic, 16 kHz
     labels_meas      measured labels (the new target), face_ok_meas where >= 2 cameras agreed
     labels_m3h       "3D hybrid": the usual hybrid recipe from better sensors (iPhone close-up
                      scores + triangulated geometry); reads talking as frowning for now
     labels_iph       the usual hybrid recipe on the iPhone close-up alone - current target
     labels_meas_lo   what the live app computes from the front webcam alone (hybrid labels):
                      the fusion model's camera input. Also stored as labels_hyb(_lo).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.retarget import LandmarkRetargeter, hybrid, resting_offset  # noqa: E402

FPS = 30
MAX_REPROJ = 6.0  # px: a camera further off than this from the others is dropped for that frame
RELAX = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel", "browInnerUp",
         "mouthUpperUpLeft", "mouthUpperUpRight", "eyeBlinkLeft", "eyeBlinkRight"]


def load_track(session, name):
    d = np.load(os.path.join(session, f"track_{name}.npz"))
    tr = {k: d[k] for k in d.files}
    if name != "iphone":
        tr["t"] = steady_times(tr["t"])
    return tr


def steady_times(t, window=301):
    """A webcam captures at a steady rate, but Windows hands the frames over in bursts (5 ms /
    60 ms apart), so the recorder's stamps jitter by +-25 ms. Rebuild them: frame index / rate,
    plus a running median of the offset (follows slow drift and the odd dropped frame)."""
    from scipy.ndimage import median_filter
    i = np.arange(len(t))
    step = np.polyfit(i, t, 1)[0]
    return median_filter(t - i * step, window, mode="nearest") + i * step


def resample_track(tr, times, max_gap=0.05):
    """Landmarks/scores linearly interpolated to `times` (only between two frames with a face,
    at most max_gap apart in total distance)."""
    t = tr["t"]
    hi = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    lo = hi - 1
    f = np.clip((times - t[lo]) / np.maximum(t[hi] - t[lo], 1e-6), 0, 1)
    ok = tr["ok"][lo] & tr["ok"][hi] & (t[hi] - t[lo] < 0.034 + max_gap) & (times >= t[0]) & (times <= t[-1])
    lmk = tr["lmk"][lo] * (1 - f)[:, None, None] + tr["lmk"][hi] * f[:, None, None]
    sc = tr["scores"][lo] * (1 - f)[:, None] + tr["scores"][hi] * f[:, None]
    lmk[~ok] = np.nan
    return lmk, sc, ok


def triangulate(views):
    """views: list of (P [3,4], normalized points [T,468,2], ok [T]). DLT per point using every
    view that has the face. Returns X [T,468,3] (nan where < 2 views)."""
    T = len(views[0][1])
    X = np.full((T, 468, 3), np.nan)
    masks = np.stack([v[2] for v in views], 1)  # [T,V]
    for combo in {tuple(m) for m in masks}:
        use = [i for i, m in enumerate(combo) if m]
        if len(use) < 2:
            continue
        for a in range(0, len(fr_all := np.flatnonzero((masks == combo).all(1))), 1000):
            fr = fr_all[a:a + 1000]
            rows = []
            for i in use:
                P, x = views[i][0], views[i][1][fr]  # [F,468,2]
                rows.append(x[..., 0:1] * P[2] - P[0])
                rows.append(x[..., 1:2] * P[2] - P[1])
            A = np.stack(rows, -2)  # [F,468,2V,4]
            _, _, Vt = np.linalg.svd(A)
            h = Vt[..., -1, :]
            X[fr] = h[..., :3] / h[..., 3:4]
    return X


def project(X, R, t, K, dist):
    """X [T,468,3] (iPhone frame) -> pixels [T,468,2] in a camera with pose (X_cam = R X + t)."""
    flat = X.reshape(-1, 3)
    good = np.isfinite(flat).all(1)
    out = np.full((len(flat), 2), np.nan)
    if good.any():
        p, _ = cv2.projectPoints(flat[good], cv2.Rodrigues(R)[0], t, K, dist)
        out[good] = p.reshape(-1, 2)
    return out.reshape(*X.shape[:2], 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--name", required=True, help="clip name in data/processed")
    p.add_argument("--z-weight", type=float, default=0.6, help="retargeter trust in depth vs x/y")
    args = p.parse_args()
    S = args.session

    with open(os.path.join(S, "calib.json")) as f:
        cal = json.load(f)["cameras"]
    with open(os.path.join(S, "iphone.json")) as f:
        info = json.load(f)
    with open(os.path.join(S, "session.json")) as f:
        sess = json.load(f)
    import torch
    names = list(torch.load(os.path.join(ROOT, "models", "fusion.pt"), map_location="cpu")["names"])

    ph = load_track(S, "iphone")
    mp_names = list(ph["names"])
    sc_cols = [mp_names.index(n) for n in names]
    ref_t = ph["t"]  # true time (the iPhone has no capture delay of its own relative to its audio)
    cams = {"iphone": {"R": np.eye(3), "t": np.zeros(3), "K": np.array(info["video"]["K"]),
                       "dist": np.array(info["video"]["dist"]), "lmk": ph["lmk"], "sc": ph["scores"],
                       "ok": ph["ok"]}}
    for name, wc in info["webcams"].items():
        tr = load_track(S, name)
        lmk, sc, ok = resample_track(tr, ref_t + wc["delay_s"])  # stamp = true time + delay
        Rw, tw = np.array(wc["R"]), np.array(wc["t_mm"])  # webcam frame -> iPhone frame
        cams[name] = {"R": Rw.T, "t": -Rw.T @ tw, "K": np.array(cal[name]["K"]),
                      "dist": np.array(cal[name]["dist"]), "lmk": lmk, "sc": sc, "ok": ok}
    order = list(cams)
    for c in cams.values():
        c["P"] = np.hstack([c["R"], c["t"][:, None]])
        pts = np.nan_to_num(c["lmk"][..., :2]).reshape(-1, 1, 2).astype(np.float64)
        c["norm"] = cv2.undistortPoints(pts, c["K"], c["dist"]).reshape(len(ref_t), 468, 2)
        c["use"] = c["ok"].copy()

    # triangulate; drop a camera that disagrees, re-solve
    for it in range(2):
        X = triangulate([(cams[n]["P"], cams[n]["norm"], cams[n]["use"]) for n in order])
        err = {}
        for n in order:
            c = cams[n]
            pr = project(X, c["R"], c["t"], c["K"], c["dist"])
            e = np.nanmedian(np.linalg.norm(pr - c["lmk"][..., :2], axis=-1), axis=1)
            err[n] = np.where(c["use"], e, np.nan)
        if it == 0:
            E = np.stack([err[n] for n in order], 1)
            nviews = np.stack([cams[n]["use"] for n in order], 1).sum(1)
            worst = np.nanargmax(np.where(np.isfinite(E), E, -1), 1)
            bad = (np.nanmax(np.where(np.isfinite(E), E, -1), 1) > MAX_REPROJ) & (nviews >= 3)
            for i, n in enumerate(order):
                cams[n]["use"] &= ~(bad & (worst == i))
    nviews = np.stack([cams[n]["use"] for n in order], 1).sum(1)
    E = np.stack([err[n] for n in order], 1)
    reproj = np.nanmax(np.where(np.isfinite(E), E, np.nan), 1)
    good = (nviews >= 2) & np.isfinite(X[:, 0, 0]) & (reproj < MAX_REPROJ)
    eye = np.linalg.norm(X[:, 33] - X[:, 263], axis=1)
    print(f"{len(ref_t)} moments: measured in 3D at {good.mean():.0%} "
          f"(3 cameras {np.mean(good & (nviews == 3)):.0%}, 2 cameras {np.mean(good & (nviews == 2)):.0%}); "
          f"re-projection median {np.nanmedian(reproj[good]):.2f} px; "
          f"eye corners {np.nanmedian(eye[good]):.1f} +- {np.nanstd(eye[good]):.1f} mm")
    for n in order:
        print(f"  {n}: face found {cams[n]['ok'].mean():.0%}, used {np.mean(good & cams[n]['use']):.0%}")

    # measured labels: geometry for every shape but gaze, minus the resting face
    sc = ph["scores"][:, sc_cols]
    rel_cols = [names.index(n) for n in RELAX]
    gi = np.flatnonzero(good)
    relaxed = gi[np.argsort(sc[gi][:, rel_cols].sum(1))[:max(60, len(gi) // 50)]]
    Xr = X * [1, -1, -1]  # retargeter frame: x right, y up, z toward camera
    rt = LandmarkRetargeter(names, z_weight=args.z_weight)
    rt.set_neutral(Xr[relaxed])
    geo = np.zeros((len(ref_t), len(names)), np.float32)
    for a in range(0, len(gi), 2000):
        idx = gi[a:a + 2000]
        geo[idx] = rt(Xr[idx], sc[idx])
    look = np.array([n.startswith("eyeLook") for n in names])
    off = np.where(look, 0, np.median(geo[relaxed], 0))
    meas = np.clip((geo - off) / np.maximum(1 - off, 0.2), 0, 1)
    meas[:, look] = sc[:, look]  # gaze: MediaPipe on the sharp close-up view
    meas[~good] = 0
    # hybrid from the better sensors: the usual per-shape recipe (realtime/retarget.hybrid), with
    # MediaPipe scores from the sharp close-up iPhone view and geometry from the triangulated points.
    # Fully geometric labels split jaw motion into other shapes (the generic ICT head's shapes
    # don't fit this face exactly); the iPhone's jawOpen tracks the measured chin drop best.
    rel_g = np.isin(gi, relaxed)
    m3h = np.zeros_like(meas)
    m3h[gi] = hybrid(sc[gi], geo[gi], names, resting_offset(sc[gi], geo[gi], names, rel_g))
    # "iPhone hybrid": the usual recipe computed entirely on the sharp close-up view (MediaPipe's own
    # smooth landmarks for geometry). The 3D fit reads talking as frowning (mouth-corner noise), so
    # this is the safer target until that's fixed.
    pi = np.flatnonzero(ph["ok"])
    pl = ph["lmk"] * [1, -1, -1]
    prel = pi[np.argsort(sc[pi][:, rel_cols].sum(1))[:max(60, len(pi) // 50)]]
    prt = LandmarkRetargeter(names)
    prt.set_neutral(pl[prel])
    pgeo = np.zeros_like(meas)
    for a in range(0, len(pi), 2000):
        pgeo[pi[a:a + 2000]] = prt(pl[pi[a:a + 2000]], sc[pi[a:a + 2000]])
    iph = np.zeros_like(meas)
    iph[pi] = hybrid(sc[pi], pgeo[pi], names, resting_offset(sc[pi], pgeo[pi], names, np.isin(pi, prel)))

    # camera input: hybrid labels from the front webcam alone, as the live app computes them
    fr = load_track(S, "front")
    fsc_cols = [list(fr["names"]).index(n) for n in names]
    fd = info["webcams"]["front"]["delay_s"]
    fl = fr["lmk"] * [1, -1, -1]
    fok = fr["ok"]
    fi = np.flatnonzero(fok)
    fsc = fr["scores"][:, fsc_cols]
    frel = fi[np.argsort(fsc[fi][:, rel_cols].sum(1))[:max(60, len(fi) // 50)]]
    frt = LandmarkRetargeter(names)
    frt.set_neutral(fl[frel])
    fgeo = np.zeros_like(fsc)
    for a in range(0, len(fi), 2000):
        idx = fi[a:a + 2000]
        fgeo[idx] = frt(fl[idx], fsc[idx])
    mouth_px = np.nanmedian(np.hypot(*(fr["lmk"][fi, 61, :2] - fr["lmk"][fi, 291, :2]).T)) * 1280 / 1920
    mouth_geo = bool(mouth_px >= 60)  # live runs the webcam at 1280 wide
    rel_mask = np.isin(np.arange(len(fi)), np.searchsorted(fi, frel))
    foff = resting_offset(fsc[fi], fgeo[fi], names, rel_mask, mouth_geo=mouth_geo)
    fhyb = np.zeros_like(fsc)
    fhyb[fi] = hybrid(fsc[fi], fgeo[fi], names, foff, mouth_geo=mouth_geo)

    # 30 fps grid of true time where mic, iPhone and front webcam all ran
    mic_t0 = sess["mic"]["first_sample_s"]
    audio, sr = sf.read(os.path.join(S, sess["mic"]["file"]), dtype="float32")
    audio = audio if audio.ndim == 1 else audio.mean(1)
    t_lo = max(ref_t[0], mic_t0, fr["t"][0] - fd)
    t_hi = min(ref_t[-1], mic_t0 + len(audio) / sr, fr["t"][-1] - fd)
    grid = t_lo + np.arange(int((t_hi - t_lo) * FPS)) / FPS

    def on_grid(t, vals, ok):
        j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        j = np.where(np.abs(t[j - 1] - grid) < np.abs(t[j] - grid), j - 1, j)
        hit = (np.abs(t[j] - grid) < 0.5 / FPS) & ok[j]
        return np.where(hit[:, None], vals[j], 0).astype(np.float32), hit

    labels_meas, ok_meas = on_grid(ref_t, meas, good)
    labels_m3h, _ = on_grid(ref_t, m3h, good)
    labels_iph, ok_iph = on_grid(ref_t, iph, ph["ok"])
    labels_cam, ok_cam = on_grid(fr["t"] - fd, fhyb, fok)
    a0 = int(round((t_lo - mic_t0) * sr))
    seg = audio[a0:a0 + int(round(len(grid) / FPS * sr))]
    audio16 = resample_poly(seg, 16000, sr).astype(np.float32)[:int(len(grid) / FPS * 16000)]

    out = os.path.join(ROOT, "data", "processed", args.name + ".npz")
    np.savez_compressed(out, audio=audio16, sr=16000, fps=FPS, names=np.array(names),
                        labels=labels_cam, face_ok=ok_cam, keep=ok_meas,
                        labels_meas=labels_meas, face_ok_meas=ok_meas,
                        labels_meas_lo=labels_cam, face_ok_meas_lo=ok_cam,
                        labels_m3h=labels_m3h, face_ok_m3h=ok_meas,
                        labels_m3h_lo=labels_cam, face_ok_m3h_lo=ok_cam,
                        labels_iph=labels_iph, face_ok_iph=ok_iph,
                        labels_iph_lo=labels_cam, face_ok_iph_lo=ok_cam,
                        labels_hyb=labels_cam, face_ok_hyb=ok_cam,
                        labels_hyb_lo=labels_cam, face_ok_hyb_lo=ok_cam,
                        grid_t=grid, front_mouth_px=mouth_px)
    np.savez_compressed(os.path.join(S, "landmarks3d_multi.npz"), times=ref_t, points=X.astype(np.float32),
                        good=good, nviews=nviews, reproj_px=reproj.astype(np.float32), measured=meas)
    print(f"{len(grid) / FPS / 60:.1f} min on the grid, measured labels on {ok_meas.mean():.0%}, "
          f"front camera input on {ok_cam.mean():.0%} (mouth ~{mouth_px:.0f} px wide at 1280)")
    print("shape            webcam-hybrid  measured (r)   3D-hybrid (r)")
    both = ok_meas & ok_cam
    for n in ("jawOpen", "mouthClose", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthFrownLeft",
              "mouthStretchLeft", "mouthRollLower", "mouthPressLeft", "eyeBlinkLeft", "browInnerUp", "browDownLeft"):
        i = names.index(n)
        r = np.corrcoef(labels_cam[both, i], labels_meas[both, i])[0, 1]
        r2 = np.corrcoef(labels_cam[both, i], labels_m3h[both, i])[0, 1]
        print(f"  {n:16s} {labels_cam[both, i].mean():.2f}           {labels_meas[both, i].mean():.2f} ({r:.2f})"
              f"    {labels_m3h[both, i].mean():.2f} ({r2:.2f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
