"""Retarget SMIRK FLAME labels to our 51 ARKit-style weights (run with .venv-smirk).

    .venv-smirk\\Scripts\\python tools/flame_to_arkit.py data/processed/<name>_flame.npz \\
        --match data/processed/<name>.npz

Geometric retargeting through MediaPipe-landmark correspondences on both heads
(models/flame_ict_corr.npz, built by tools/flame_ict_corr.py):

  1. FLAME: displacement of the ~437 corresponding surface points caused by expression
     only - points(shape, expr, jaw, eyelids) - points(shape, 0, 0, 0), no head pose.
  2. Map into the ICT head's frame with the similarity transform (rotation + scale)
     that best aligns the two neutral point sets.
  3. Solve, for all frames at once (projected-gradient / FISTA on the GPU),
         min_w  sum_t ||J w_t - d_t||^2 + lam ||w_t||^2 + mu ||w_t - w_t-1||^2,  0 <= w <= 1
     where J holds the ICT shape deltas at the points, mapped to our 51 names exactly
     like realtime/head3d.py (ict_mapping). Lip points are weighted up.
  4. Gaze (eyeLook*) has no FLAME source: copied from the clip's MediaPipe labels.

Adds `labels_flame` [n,51] (order = the clip's `names` minus `_neutral`), `face_ok_flame`,
`flame_fit_err` [n] (rms residual, cm) to the --match npz, shifted with `drift_src`
like tools/degrade_track.py if the clip was drift-corrected. Everything else is kept.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from smirk_flame import Flame  # noqa: E402
from src.FLAME.lbs import batch_rodrigues  # noqa: E402  (SMIRK, on sys.path via smirk_flame)
from realtime.head3d import ict_mapping  # noqa: E402  (no GL needed for this function)

CORR = os.path.join(ROOT, "models", "flame_ict_corr.npz")
ICT = os.path.join(ROOT, "models", "ict_head.npz")
LIPS = sorted({61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
               76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 184, 74, 73, 72, 11, 302, 303, 304, 408,
               62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292, 183, 42, 41, 38, 12, 268, 271, 272, 407})
EYES = sorted({33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
               263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398})
W_LIPS, W_EYES = 3.0, 2.0
# Inner-lip material points are unreliable in SMIRK fits: its FLAME closes the mouth with
# thick, rolled lips while the inner rim stays ~2 cm open, so the points on the contact line
# separate even when the lips visibly touch. Their weight is lowered and the visible opening
# is matched instead: GAP_COLS = (upper, lower) inner-lip landmark pairs whose vertical ICT
# separation must equal FLAME's visible lip gap in that column (see visible_gaps).
INNER = sorted({78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415})
W_INNER = 0.3
GAP_COLS = [(13, 14), (82, 87), (312, 317)]
W_GAP = 20.0
# Eyelids: FLAME's (and the user's) eye opening is ~6 mm, ICT's ~10 mm, so raw lid displacement
# under-states blinks. These (upper, lower) lid pairs get rows asking that the ICT eye opening
# change by the same FRACTION of its neutral opening as FLAME's does.
EYE_PAIRS = [(159, 145), (158, 153), (160, 144), (386, 374), (385, 380), (387, 373)]


def umeyama(src, dst):
    """Similarity (s, R, t) minimising ||s R src + t - dst||."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


class Retargeter:
    def __init__(self, names, device="cuda"):
        self.device = device
        self.names = list(names)
        c = np.load(CORR)
        ok = c["ok"]
        self.lmk = np.flatnonzero(ok)
        self.fv, self.fb = c["flame_vids"][ok], c["flame_bary"][ok]
        ict = np.load(ICT)
        iv, ib = c["ict_vids"][ok], c["ict_bary"][ok]
        self.ict_neutral = (ict["vertices"][iv] * ib[..., None]).sum(1)  # [K,3] cm
        M = ict_mapping(self.names, list(ict["names"]))                   # [51,53]
        deltas = (ict["deltas"][:, iv] * ib[None, ..., None]).sum(2)      # [53,K,3]
        self.J_all = np.einsum("si,ikc->kcs", M, deltas).reshape(-1, len(self.names))  # [3K,51]
        self.look = np.array([n.startswith("eyeLook") for n in self.names])
        w = np.ones(len(self.lmk))
        w[np.isin(self.lmk, LIPS)] = W_LIPS
        w[np.isin(self.lmk, EYES)] = W_EYES
        w[np.isin(self.lmk, INNER)] = W_INNER
        L = list(self.lmk)
        self.gap_pairs = [(L.index(u), L.index(l)) for u, l in GAP_COLS]
        J = self.J_all.reshape(len(L), 3, -1)
        self.eye_pairs = [(L.index(u), L.index(l)) for u, l in EYE_PAIRS]
        pairs = self.gap_pairs + self.eye_pairs
        gap_rows = np.stack([J[u, 1] - J[l, 1] for u, l in pairs])  # vertical separation
        self.J_all = np.concatenate([self.J_all, gap_rows])
        self.n_extra = len(pairs)
        self.pt_w = np.r_[np.repeat(w, 3), np.full(len(pairs), W_GAP)]
        n = self.ict_neutral
        self.ict_eye_open = np.array([n[u, 1] - n[l, 1] for u, l in self.eye_pairs])
        self.flame = Flame().to(device)
        self._lip_tris()

    def _lip_tris(self):
        """Triangles around the lips (FLAME 'lips' mask grown by two rings), for visible_gaps."""
        from smirk_flame import ASSETS, safe_pickle_load
        masks = safe_pickle_load(os.path.join(ASSETS, "FLAME_masks", "FLAME_masks.pkl"))
        f = self.flame.faces.cpu().numpy()
        sel = np.zeros(self.flame.v_template.shape[0], bool)
        sel[masks["lips"]] = True
        for _ in range(2):
            sel[f[sel[f].any(1)].ravel()] = True
        self.lip_tris = torch.as_tensor(f[sel[f].any(1)], device=self.device)

    def visible_gaps(self, v, p, bins=160):
        """Visible lip opening per column, as the camera sees it: the length of the uncovered
        stretch of the vertical image line through each GAP_COLS upper point, between the outer
        lip points 0 and 17. v [B,V,3] vertices, p [B,K,3] points, both already rotated by the
        head pose (SMIRK's camera is orthographic along z). Returns [B,C] (FLAME units).
        Measured in the camera view because that is what SMIRK fits: its closed mouths often
        have a few mm of opening when the head is turned to face front, hidden by head pitch."""
        L = list(self.lmk)
        y_top, y_bot = p[:, L.index(0), 1] + 0.003, p[:, L.index(17), 1] - 0.003
        t = v[:, self.lip_tris]                                   # [B,T,3,3]
        out = []
        frac = torch.linspace(0, 1, bins, device=v.device)
        yb = y_bot[:, None] + (y_top - y_bot)[:, None] * frac     # [B,bins]
        for u, _ in self.gap_pairs:
            cx = p[:, u, 0][:, None]                              # [B,1]
            ys, valid = [], []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                xa, xb = t[..., a, 0], t[..., b, 0]
                cross = (xa - cx) * (xb - cx) <= 0
                u = ((cx - xa) / torch.where((xb - xa).abs() < 1e-9, torch.full_like(xa, 1e-9), xb - xa)).clamp(0, 1)
                ys.append(t[..., a, 1] + u * (t[..., b, 1] - t[..., a, 1]))
                valid.append(cross)
            ys, valid = torch.stack(ys, -1), torch.stack(valid, -1)      # [B,T,3]
            lo = torch.where(valid, ys, torch.full_like(ys, 1e9)).min(-1).values
            hi = torch.where(valid, ys, torch.full_like(ys, -1e9)).max(-1).values
            covered = ((yb[:, None, :] >= lo[..., None]) & (yb[:, None, :] <= hi[..., None])).any(1)  # [B,bins]
            out.append((~covered).float().mean(1) * (y_top - y_bot))
        return torch.stack(out, 1)

    def points(self, v):
        idx = torch.as_tensor(self.fv, device=v.device)
        bary = torch.as_tensor(self.fb, device=v.device)
        return (v[:, idx] * bary[None, ..., None]).sum(2)  # [B,K,3]

    @torch.no_grad()
    def displacements(self, shape, expression, jaw, eyelid, pose, chunk=256):
        """FLAME expression-only displacement at the points + visible lip gaps, ICT units. [T,3K+C]"""
        dev = self.device
        sh = torch.as_tensor(shape, dtype=torch.float32, device=dev)[None]
        z3 = torch.zeros(1, 3, device=dev)
        v0, _ = self.flame(sh, torch.zeros(1, 50, device=dev), z3, z3, torch.zeros(1, 2, device=dev))
        p0 = self.points(v0)[0].cpu().numpy().astype(np.float64)
        s, R, _ = umeyama(p0, self.ict_neutral.astype(np.float64))
        self.align = (s, R)
        A = torch.as_tensor(s * R, dtype=torch.float32, device=dev)
        P0 = self.points(v0)
        eu = torch.as_tensor([u for u, _ in self.eye_pairs], device=dev)
        el = torch.as_tensor([l for _, l in self.eye_pairs], device=dev)
        f_open = P0[0, eu, 1] - P0[0, el, 1]                      # FLAME neutral eye openings
        eye_scale = torch.as_tensor(self.ict_eye_open, dtype=torch.float32, device=dev) / f_open
        out = []
        for i in range(0, len(expression), chunk):
            e = torch.as_tensor(expression[i:i + chunk], dtype=torch.float32, device=dev)
            b = len(e)
            v, _ = self.flame(sh.expand(b, -1), e, z3.expand(b, -1),
                              torch.as_tensor(jaw[i:i + chunk], dtype=torch.float32, device=dev),
                              torch.as_tensor(eyelid[i:i + chunk], dtype=torch.float32, device=dev))
            P = self.points(v)
            d = (P - P0) @ A.T
            Rp = batch_rodrigues(torch.as_tensor(pose[i:i + chunk], dtype=torch.float32, device=dev))
            gap = self.visible_gaps(v @ Rp.transpose(1, 2), P @ Rp.transpose(1, 2)) * s
            eye = ((P[:, eu, 1] - P[:, el, 1]) - f_open) * eye_scale  # same fraction of the opening
            out.append(torch.cat([d.reshape(b, -1), gap, eye], 1))
        return torch.cat(out)

    @torch.no_grad()
    def solve(self, d, valid, lam=0.02, mu=0.05, iters=600):
        """Batched box-constrained least squares with temporal smoothness (FISTA). d: [T,3K]."""
        dev = self.device
        free = ~self.look
        J = torch.as_tensor(self.J_all[:, free] * np.sqrt(self.pt_w)[:, None], dtype=torch.float32, device=dev)
        dw = d * torch.as_tensor(np.sqrt(self.pt_w), dtype=torch.float32, device=dev)
        H = J.T @ J
        scale = H.diagonal().mean()
        H = H + lam * scale * torch.eye(len(H), device=dev)
        g = dw @ J                                        # [T,S]
        valid_t = torch.as_tensor(valid, device=dev)
        link = (valid_t[1:] & valid_t[:-1]).float()[:, None]  # smooth only across consecutive good frames
        m = mu * scale
        L = torch.linalg.eigvalsh(H).max() + 4 * m
        w = torch.zeros_like(g)
        y, tk = w.clone(), 1.0
        for _ in range(iters):
            grad = y @ H - g
            diff = (y[1:] - y[:-1]) * link
            grad[1:] += m * diff
            grad[:-1] -= m * diff
            w_new = (y - grad / L).clamp(0, 1)
            t_new = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
            y = w_new + ((tk - 1) / t_new) * (w_new - w)
            w, tk = w_new, t_new
        res = (w @ J.T - dw) / torch.as_tensor(np.sqrt(self.pt_w), dtype=torch.float32, device=dev)
        rms = res[:, :-self.n_extra].pow(2).mean(1).sqrt()
        full = torch.zeros(len(w), len(self.names), device=dev)
        full[:, torch.as_tensor(free, device=dev)] = w
        return full.cpu().numpy(), rms.cpu().numpy()


def retarget_npz(flame_npz, names, device="cuda", **kw):
    F = np.load(flame_npz)
    ok = F["face_ok"]
    r = Retargeter(names, device)
    d = r.displacements(F["shape_mean"], F["expression"], F["jaw"], F["eyelid"], F["pose"])
    w, rms = r.solve(d, ok, **kw)
    w[~ok] = 0
    rms[~ok] = np.nan
    return w.astype(np.float32), ok, rms.astype(np.float32), r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flame_npz")
    ap.add_argument("--match", required=True, help="clip npz (prepare_data) to add labels_flame to")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    clip = dict(np.load(a.match))
    all_names = list(clip["names"])
    names = [x for x in all_names if x != "_neutral"]
    n = len(clip["face_ok"])
    F = np.load(a.flame_npz)
    if len(F["face_ok"]) != n or int(F["start_frame"]) != 0:
        sys.exit(f"{a.flame_npz} is not on the clip's grid ({len(F['face_ok'])} vs {n} frames): "
                 "run smirk_track.py with --match")
    t0 = time.time()
    w, ok, rms, r = retarget_npz(a.flame_npz, names, a.device, lam=a.lam, mu=a.mu)
    if "drift_src" in clip:  # same re-timing fix_drift.py applied to the MediaPipe labels
        src = clip["drift_src"]
        ok = ok[np.clip(np.round(src).astype(int), 0, n - 1)]
        rms = np.interp(src, np.arange(n), rms).astype(np.float32)
        w = np.stack([np.interp(src, np.arange(n), w[:, i]) for i in range(w.shape[1])], 1).astype(np.float32)
    lab = clip["labels"][:, [all_names.index(x) for x in names]]
    look = [i for i, x in enumerate(names) if x.startswith("eyeLook")]
    w[:, look] = np.where(ok[:, None], lab[:, look], 0)  # gaze: MediaPipe
    clip["labels_flame"], clip["face_ok_flame"], clip["flame_fit_err"] = w, ok, rms
    tmp = a.match + ".tmp.npz"
    np.savez_compressed(tmp, **clip)
    os.replace(tmp, a.match)
    s, _ = r.align
    print(f"{os.path.basename(a.match)}: {ok.sum()} frames retargeted in {time.time() - t0:.1f} s "
          f"(FLAME->ICT scale {s:.1f}, fit rms {np.nanmedian(rms):.3f} cm median)")


if __name__ == "__main__":
    main()
