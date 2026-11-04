"""Geometric retargeting: MediaPipe's 3D face landmarks -> ARKit-style blendshape weights.

MediaPipe's own blendshape scores don't mean what an ARKit rig (MetaHuman, the
ICT head) expects: on a relaxed face they read eyeSquint ~0.3 and mouthPucker
~0.2, smiles come without the cheek raise, and frowns/stretches never fire, so a
grimace looks like a smile. Its 478 landmarks are much more faithful. Here we
measure the face as geometry instead:

  1. undo head pose: align the frame's rigid points (forehead, nose bridge,
     temples) to the user's neutral face with a similarity transform,
  2. displacement d = aligned landmarks - neutral landmarks, mapped into the ICT
     head's frame (cm) by the similarity that aligns the two neutral faces,
  3. solve  min_w ||J w - d||^2 + lam ||w||^2,  0 <= w <= 1,  where J holds the ICT
     head's ARKit shape deltas at the same landmarks (models/flame_ict_corr.npz maps
     MediaPipe landmark ids onto the ICT surface). Lips and eyelids are weighted up;
     eye openings are matched as a fraction of the neutral opening, since eye sizes
     differ between people and the ICT head.
  4. gaze (eyeLook*) has no shape-geometry signal here: taken from MediaPipe's scores.

The ICT shapes follow Apple's ARKit definitions, so the weights mean the same
thing MetaHuman's ARKit mapping expects. Works per frame (live, ~0.1 ms) or for
whole clips at once (offline labels).
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR = os.path.join(ROOT, "models", "flame_ict_corr.npz")
ICT = os.path.join(ROOT, "models", "ict_head.npz")

# rigid-ish points for head-pose removal: forehead, nose bridge, temples, cheekbone tops
RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 127, 356, 234, 454, 21, 251, 54, 284, 103, 332, 67, 297]
LIPS = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415}
EYES = {33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398}
BROWS = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276}
W = {"lips": 3.0, "eyes": 2.0, "brows": 2.0}
EYE_PAIRS = [(159, 145), (158, 153), (160, 144), (386, 374), (385, 380), (387, 373)]
W_EYE_OPEN = 20.0


def umeyama(src, dst, w=None):
    """Similarity (s, R, t) minimising sum w ||s R src + t - dst||^2."""
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd((b * w[:, None]).T @ a)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (w @ (a ** 2).sum(1))
    return s, R, md - s * R @ ms


def to_metric(lmk, width, height):
    """MediaPipe normalized landmarks [...,478,3] -> right-handed 3D (x right, y up, z toward camera),
    in pixel units. (Image x right = the person's left = ICT +x.)"""
    p = np.asarray(lmk, np.float64)[..., :468, :].copy()
    p[..., 0] *= width
    p[..., 1] *= -height
    p[..., 2] *= -width
    return p


class LandmarkRetargeter:
    def __init__(self, names, lam=0.01, z_weight=0.3):
        """z_weight: trust in depth relative to x/y. A single webcam measures depth poorly, and
        lip protrusion (pucker) is mostly depth - over-trusting z makes puckers fit as presses."""
        from realtime.head3d import ict_mapping  # pure numpy, no GL
        self.names = list(names)
        c = np.load(CORR)
        self.ok = np.flatnonzero(c["ok"])
        ict = np.load(ICT)
        iv, ib = c["ict_vids"][self.ok], c["ict_bary"][self.ok]
        self.ict_neutral = (ict["vertices"][iv] * ib[..., None]).sum(1)          # [K,3] cm
        M = ict_mapping(self.names, list(ict["names"]))                         # [51,53]
        deltas = (ict["deltas"][:, iv] * ib[None, ..., None]).sum(2)            # [53,K,3]
        J = np.einsum("si,ikc->kcs", M, deltas)                                 # [K,3,51]
        L = list(self.ok)
        w = np.ones(len(L))
        for group, ids in (("lips", LIPS), ("eyes", EYES), ("brows", BROWS)):
            w[np.isin(self.ok, list(ids))] = W[group]
        self.eye_pairs = [(L.index(u), L.index(l)) for u, l in EYE_PAIRS if u in L and l in L]
        n = self.ict_neutral
        self.ict_eye_open = np.array([n[u, 1] - n[l, 1] for u, l in self.eye_pairs])
        eye_rows = np.stack([J[u, 1] - J[l, 1] for u, l in self.eye_pairs]) / self.ict_eye_open[:, None]
        self.look = np.array([x.startswith("eyeLook") for x in self.names])
        self.free = ~self.look
        rows = np.concatenate([J.reshape(-1, len(self.names)), eye_rows])[:, self.free]
        xyz = np.tile([1.0, 1.0, z_weight], len(w))
        self.sw = np.sqrt(np.r_[np.repeat(w, 3) * xyz, np.full(len(self.eye_pairs), W_EYE_OPEN)])
        self.J = rows * self.sw[:, None]
        H = self.J.T @ self.J
        self.H = H + lam * H.diagonal().mean() * np.eye(len(H))
        self.step = 1.0 / np.linalg.eigvalsh(self.H).max()
        self.rigid = np.array([L.index(i) for i in RIGID if i in L])
        self.neutral = None
        self.gain = np.ones(len(self.names))
        self.offset = np.zeros(len(self.names))

    # --- neutral face -------------------------------------------------------------
    def set_neutral(self, frames):
        """frames: [N,468,3] metric landmarks of a relaxed face (any head pose)."""
        P = np.asarray(frames, np.float64)[:, self.ok]
        ref = P[0]
        for _ in range(2):  # align all to a reference, take the median, repeat
            aligned = np.stack([self._align(p, ref) for p in P])
            ref = np.median(aligned, 0)
        self.neutral = ref
        s, R, _ = umeyama(ref, self.ict_neutral)
        self.to_ict = s * R
        self.eye_open = np.array([ref[u, 1] - ref[l, 1] for u, l in self.eye_pairs])

    def _align(self, p, ref):
        s, R, t = umeyama(p[self.rigid], ref[self.rigid])
        return (s * (R @ p.T)).T + t

    # --- solving ------------------------------------------------------------------
    def displacement(self, P):
        """[T,K,3] metric landmarks (already self.ok-indexed) -> weighted rows [T,R]."""
        A = np.stack([self._align(p, self.neutral) for p in P])
        d = (A - self.neutral) @ self.to_ict.T                                   # cm, ICT frame
        eye = np.stack([(A[:, u, 1] - A[:, l, 1]) for u, l in self.eye_pairs], 1)
        eye = (eye - self.eye_open) / self.eye_open                              # fraction of opening
        return np.concatenate([d.reshape(len(P), -1), eye], 1) * self.sw

    def solve(self, rows, w0=None, iters=150):
        """Box-constrained least squares (FISTA), many frames at once. rows [T,R] -> [T,S_free]."""
        g = rows @ self.J
        w = np.zeros_like(g) if w0 is None else w0.copy()
        y, tk = w.copy(), 1.0
        for _ in range(iters):
            w_new = np.clip(y - (y @ self.H - g) * self.step, 0, 1)
            t_new = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
            y = w_new + ((tk - 1) / t_new) * (w_new - w)
            w, tk = w_new, t_new
        return w

    def __call__(self, lmk_metric, mp_scores=None, w0=None, iters=150):
        """lmk_metric [T,468,3] (or [468,3]); mp_scores [T,S] MediaPipe scores for gaze.
        Returns ARKit-style weights [T,S] (or [S]) with calibration gains applied."""
        single = np.ndim(lmk_metric) == 2
        P = np.asarray(lmk_metric, np.float64)[None] if single else np.asarray(lmk_metric, np.float64)
        P = P[:, self.ok]
        out = np.zeros((len(P), len(self.names)), np.float32)
        out[:, self.free] = self.solve(self.displacement(P), w0, iters)
        if mp_scores is not None:
            out[:, self.look] = np.asarray(mp_scores, np.float32).reshape(len(P), -1)[:, self.look]
        out = np.clip((out - self.offset) * self.gain, 0, 1)
        return out[0] if single else out

    # --- calibration persistence ----------------------------------------------------
    def save(self, path):
        np.savez(path, neutral=self.neutral, to_ict=self.to_ict, eye_open=self.eye_open,
                 gain=self.gain, offset=self.offset, names=np.array(self.names))

    def load(self, path):
        d = np.load(path)
        assert list(d["names"]) == self.names
        self.neutral, self.to_ict, self.eye_open = d["neutral"], d["to_ict"], d["eye_open"]
        self.gain, self.offset = d["gain"], d["offset"]


# --- hybrid labels ----------------------------------------------------------------------
# Which source to trust per shape, from side-by-side renders and the RAVDESS emotion check
# (docs in README Phase 4). Geometry: eyes, smiles vs grimaces (MediaPipe scores disgust as a
# smile; geometry as frown + upper-lip raise), brow lowering (MediaPipe reads ~0.6 on a neutral
# face). MediaPipe scores: lip protrusion (pucker/funnel - mostly depth, which one camera
# measures poorly), jaw, inner/outer brow raise, side-to-side and cheek puff, gaze.
GEO_PREFIXES = ("eyeBlink", "eyeSquint", "eyeWide", "mouthSmile", "mouthStretch", "mouthFrown",
                "mouthLowerDown", "mouthUpperUp", "mouthClose", "mouthDimple", "cheekSquint",
                "noseSneer", "browDown", "mouthPress", "mouthRoll")


# Small faces (mouth under ~60 px wide) give noisy lip geometry - lopsided smiles, odd mouths -
# so there only the eyes and brows come from geometry (mouth_geo=False).
GEO_UPPER = ("eyeBlink", "eyeSquint", "eyeWide", "browDown", "cheekSquint")


def geo_columns(names, mouth_geo=True):
    return np.array([n.startswith(GEO_PREFIXES if mouth_geo else GEO_UPPER) for n in names])


def hybrid(mp, geo, names, offset=None, mouth_geo=True):
    """Combine MediaPipe scores and geometric weights ([T,S] or [S]) shape by shape, then
    subtract the person's resting values (`offset`, from their neutral face) and rescale so
    1 still means 1. Gaze is left as is."""
    g = geo_columns(names, mouth_geo)
    out = np.where(g, geo, mp).astype(np.float32)
    if offset is not None:
        look = np.array([n.startswith("eyeLook") for n in names])
        off = np.where(look, 0, offset)
        out = np.clip((out - off) / np.maximum(1 - off, 0.2), 0, 1)
    return out


def resting_offset(mp, geo, names, relaxed, mouth_geo=True):
    """Per-shape resting value: median over the person's relaxed frames (boolean mask/indices)."""
    return np.nanmedian(hybrid(mp[relaxed], geo[relaxed], names, mouth_geo=mouth_geo), axis=0)
