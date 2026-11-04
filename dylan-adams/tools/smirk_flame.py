"""FLAME + SMIRK helpers that avoid pytorch3d and chumpy (run with .venv-smirk).

SMIRK (third_party/smirk, MIT) predicts FLAME parameters from a 224x224 face crop.
Its own FLAME.py can't be imported on numpy 2 (it aliases np.float_ etc.) and its
renderer needs pytorch3d (no Windows wheels for new torch), so this module has:

  * convert_flame(): one-time conversion of FLAME2020/generic_model.pkl (a Python 2
    chumpy pickle) into a plain .npz, using a restricted unpickler with stub
    classes for chumpy/scipy objects - neither chumpy nor arbitrary pickle code runs.
  * Flame: a minimal FLAME layer (vertices + MediaPipe-embedded landmarks) built on
    SMIRK's lbs.py (pure torch).
  * load_encoder(): SMIRK's encoder with the released SMIRK_em1.pt weights.
  * project() / render_mesh(): SMIRK's orthographic camera and a tiny OpenCV
    painter's-algorithm renderer for eyeballing fits.

CLI:  python tools/smirk_flame.py convert   # FLAME2020.zip/pkl -> generic_model.npz
"""

import _compat_pickle
import argparse
import os
import pickle
import sys
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMIRK_DIR = os.path.join(ROOT, "third_party", "smirk")
ASSETS = os.path.join(SMIRK_DIR, "assets")
FLAME_DIR = os.path.join(ASSETS, "FLAME2020")
FLAME_PKL = os.path.join(FLAME_DIR, "generic_model.pkl")
FLAME_NPZ = os.path.join(FLAME_DIR, "generic_model.npz")
FLAME_ZIP = os.path.join(ROOT, "models", "flame", "FLAME2020.zip")
CHECKPOINT = os.path.join(SMIRK_DIR, "pretrained_models", "SMIRK_em1.pt")
CROP_SIZE = 224
CROP_SCALE = 1.4  # SMIRK's demo / test-time crop: landmark bbox * 1.4

sys.path.insert(0, SMIRK_DIR)
from src.FLAME.lbs import lbs, vertices2landmarks  # noqa: E402  (pure torch, no pytorch3d)


# --------------------------------------------------------------------------- #
# Safe loading of old pickles (FLAME generic_model.pkl, FLAME_masks.pkl)
# --------------------------------------------------------------------------- #
class _Stub:
    """Stands in for chumpy.Ch / scipy.sparse matrices: just keeps the pickled state."""
    _kind = "?"

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:  # (dict, slotstate) form
            state = {**(state[0] or {}), **(state[1] or {})}
        self.__dict__["_state"] = state


def _stub_class(kind):
    return type(f"Stub_{kind}", (_Stub,), {"_kind": kind})


class _SafeUnpickler(pickle.Unpickler):
    """Only numpy array reconstruction + stubs; anything else is refused."""
    ALLOWED = {
        ("copyreg", "_reconstructor"), ("builtins", "object"), ("_codecs", "encode"), ("builtins", "set"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) in _compat_pickle.NAME_MAPPING:  # Python 2 names
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if module.split(".")[0] == "chumpy":
            return _stub_class(f"chumpy.{name}")
        if module.startswith("scipy.sparse"):
            return _stub_class(f"sparse.{name}")
        if (module, name) in self.ALLOWED:
            if module.startswith("numpy.core"):
                module = module.replace("numpy.core", "numpy._core")
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def safe_pickle_load(path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f, encoding="latin1").load()


def _to_array(v, key="?"):
    if isinstance(v, _Stub):
        s = v._state
        if v._kind.startswith("chumpy"):
            if "x" in s:  # chumpy.Ch keeps its value in .x
                return np.asarray(s["x"])
            raise ValueError(f"{key}: chumpy object {v._kind} without .x (state keys {list(s)})")
        if v._kind.startswith("sparse"):
            from scipy import sparse
            shape = s.get("_shape") or s.get("shape")
            fmt = s.get("format", v._kind.split(".")[-1][:3])
            cls = sparse.csc_matrix if "csc" in fmt else sparse.csr_matrix
            return cls((s["data"], s["indices"], s["indptr"]), shape=shape).toarray()
        raise ValueError(f"{key}: unknown stub {v._kind}")
    return v if isinstance(v, (str, bytes)) else np.asarray(v)


def convert_flame(src=None, dst=FLAME_NPZ):
    """generic_model.pkl (chumpy pickle) -> plain npz. Returns dst."""
    src = src or FLAME_PKL
    raw = safe_pickle_load(src)
    out = {}
    for k, v in raw.items():
        try:
            out[k] = _to_array(v, k)
        except ValueError as e:
            print(f"  skipping {k}: {e}")
    for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f", "kintree_table"):
        if k not in out:
            raise KeyError(f"{src} has no {k} (keys: {sorted(raw)})")
    np.savez_compressed(dst, **{k: v for k, v in out.items() if not isinstance(v, (str, bytes))})
    return dst


def ensure_flame():
    """Make sure assets/FLAME2020/generic_model.npz exists (unzipping FLAME2020.zip if needed)."""
    if os.path.exists(FLAME_NPZ):
        return FLAME_NPZ
    if not os.path.exists(FLAME_PKL):
        if not os.path.exists(FLAME_ZIP):
            raise FileNotFoundError(
                f"FLAME model missing: put FLAME2020.zip (from flame.is.tue.mpg.de) at {FLAME_ZIP}")
        os.makedirs(FLAME_DIR, exist_ok=True)
        with zipfile.ZipFile(FLAME_ZIP) as z:
            for m in z.infolist():  # flatten, refuse path tricks
                name = os.path.basename(m.filename)
                if not name or m.is_dir():
                    continue
                with z.open(m) as fi, open(os.path.join(FLAME_DIR, name), "wb") as fo:
                    fo.write(fi.read())
        if not os.path.exists(FLAME_PKL):
            raise FileNotFoundError(f"{FLAME_ZIP} has no generic_model.pkl; got {os.listdir(FLAME_DIR)}")
    return convert_flame()


# --------------------------------------------------------------------------- #
# FLAME layer
# --------------------------------------------------------------------------- #
class Flame(nn.Module):
    """FLAME 2020 with SMIRK's conventions: 300 shape + 50 expression, eyelid blendshapes."""

    def __init__(self, n_shape=300, n_exp=50):
        super().__init__()
        m = np.load(ensure_flame())
        f32 = lambda a: torch.tensor(np.asarray(a, np.float32))  # noqa: E731
        self.register_buffer("faces", torch.tensor(m["f"].astype(np.int64)))
        self.register_buffer("v_template", f32(m["v_template"]))
        sd = m["shapedirs"]
        self.register_buffer("shapedirs", f32(np.concatenate([sd[:, :, :n_shape], sd[:, :, 300:300 + n_exp]], 2)))
        pd = m["posedirs"]
        self.register_buffer("posedirs", f32(pd.reshape(-1, pd.shape[-1]).T))
        self.register_buffer("J_regressor", f32(m["J_regressor"]))
        parents = torch.tensor(m["kintree_table"][0].astype(np.int64))
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", f32(m["weights"]))
        self.register_buffer("l_eyelid", f32(np.load(os.path.join(ASSETS, "l_eyelid.npy")))[None])
        self.register_buffer("r_eyelid", f32(np.load(os.path.join(ASSETS, "r_eyelid.npy")))[None])
        mp = np.load(os.path.join(ASSETS, "mediapipe_landmark_embedding", "mediapipe_landmark_embedding.npz"))
        self.register_buffer("mp_faces_idx", torch.tensor(mp["lmk_face_idx"].astype(np.int64)))
        self.register_buffer("mp_bary", f32(mp["lmk_b_coords"]))
        self.mp_indices = mp["landmark_indices"].astype(np.int64)  # which of MediaPipe's 478 points

    def forward(self, shape, expression, pose, jaw, eyelid=None, neck=None, eyes=None):
        b = expression.shape[0]
        if shape is None:
            shape = expression.new_zeros(b, self.shapedirs.shape[2] - expression.shape[1])
        neck = expression.new_zeros(b, 3) if neck is None else neck
        eyes = expression.new_zeros(b, 6) if eyes is None else eyes
        betas = torch.cat([shape, expression], 1)
        full_pose = torch.cat([pose, neck, jaw, eyes], 1)
        verts, _ = lbs(betas, full_pose, self.v_template.expand(b, -1, -1), self.shapedirs,
                       self.posedirs, self.J_regressor, self.parents, self.lbs_weights)
        if eyelid is not None:
            verts = verts + self.r_eyelid * eyelid[:, 1:2, None] + self.l_eyelid * eyelid[:, 0:1, None]
        lmk_mp = vertices2landmarks(verts, self.faces, self.mp_faces_idx.repeat(b, 1),
                                    self.mp_bary.repeat(b, 1, 1))
        return verts, lmk_mp


def face_mask_vertices():
    """Vertex indices of FLAME's 'face' region (SMIRK renders only these)."""
    return np.asarray(safe_pickle_load(os.path.join(ASSETS, "FLAME_masks", "FLAME_masks.pkl"))["face"])


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def load_encoder(device="cuda", checkpoint=CHECKPOINT):
    import src.smirk_encoder as se
    orig = se.create_backbone
    # timm would otherwise download ImageNet weights that the checkpoint overwrites anyway
    se.create_backbone = lambda name, pretrained=True: orig(name, pretrained=False)
    enc = se.SmirkEncoder()
    se.create_backbone = orig
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    enc.load_state_dict({k.replace("smirk_encoder.", ""): v for k, v in ck.items()
                         if k.startswith("smirk_encoder.")})
    return enc.to(device).eval()


# --------------------------------------------------------------------------- #
# Cropping / projection / rendering
# --------------------------------------------------------------------------- #
def crop_transform(landmarks_px, scale=CROP_SCALE, size=CROP_SIZE):
    """SMIRK's crop: square around the landmark bbox. Returns 2x3 affine (frame -> crop)."""
    x0, y0 = landmarks_px[:, 0].min(), landmarks_px[:, 1].min()
    x1, y1 = landmarks_px[:, 0].max(), landmarks_px[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(((x1 - x0) + (y1 - y0)) / 2 * scale)
    s = (size - 1) / max(side, 1)
    return np.array([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s]], np.float64)


def project(points, cam, size=CROP_SIZE):
    """SMIRK orthographic camera -> pixel coords in the crop. points [B,N,3], cam [B,3]=(s,tx,ty)."""
    s, t = cam[:, None, 0:1], cam[:, None, 1:3]
    xy = s * (points[..., :2] + t)
    xy = torch.stack([xy[..., 0], -xy[..., 1]], -1)
    return (xy + 1) * size / 2


def render_mesh(img, verts, px, faces, color=(200, 200, 200)):
    """Flat-shaded painter's-algorithm overlay. verts [N,3] (model space, for normals/depth),
    px [N,2] pixel positions in img. Returns a new image with the mesh drawn on black."""
    out = np.zeros_like(img)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = np.clip(n @ np.array([0.3, 0.3, 1.0]) / np.linalg.norm([0.3, 0.3, 1.0]), 0, 1)
    visible = n[:, 2] > 0
    order = np.argsort(tri[:, :, 2].mean(1))  # far (small z) first
    p = np.round(px * 16).astype(np.int32)  # sub-pixel precision (shift=4)
    for i in order:
        if visible[i]:
            c = tuple(int(v * (0.25 + 0.75 * shade[i])) for v in color)
            cv2.fillConvexPoly(out, p[faces[i]], c, lineType=cv2.LINE_AA, shift=4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert"])
    ap.add_argument("--src", help="generic_model.pkl (default: unzip models/flame/FLAME2020.zip)")
    a = ap.parse_args()
    if a.src:
        print("wrote", convert_flame(a.src))
    else:
        print("ready:", ensure_flame())
    m = np.load(FLAME_NPZ)
    print({k: m[k].shape for k in m.files})


if __name__ == "__main__":
    main()
