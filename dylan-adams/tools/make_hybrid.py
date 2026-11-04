r"""Add hybrid labels (realtime/retarget.hybrid) to processed clips that have labels_geo.

    .venv\Scripts\python tools/make_hybrid.py data/processed/*.npz

labels_hyb    = per-shape choice of MediaPipe score / geometric weight, minus the person's
                resting face (median of the clip's most relaxed kept frames)
labels_hyb_lo = the same from the low-resolution passes (fusion model's "weak camera" input),
                using the full-res resting face
face_ok_hyb(_lo) = both trackers found the face.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.retarget import hybrid, resting_offset  # noqa: E402

RELAX = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel", "browInnerUp",
         "mouthUpperUpLeft", "mouthUpperUpRight", "eyeBlinkLeft", "eyeBlinkRight"]


def main():
    for path in sys.argv[1:]:
        if path.endswith(("_flame.npz", "_lmk.npz", "_lmk320.npz", "_lmk480.npz")):
            continue
        d = dict(np.load(path))
        if "labels_geo" not in d:
            continue
        allnames = list(d["names"])
        names = [n for n in allnames if n != "_neutral"]
        cols = [allnames.index(n) for n in names]
        mp, geo = d["labels"][:, cols], d["labels_geo"]
        ok = d["face_ok"] & d["face_ok_geo"]
        rel_cols = [names.index(n) for n in RELAX]
        energy = np.where(ok & d["keep"], mp[:, rel_cols].sum(1), np.inf)
        relaxed = np.argsort(energy)[:max(30, int(0.05 * np.isfinite(energy).sum()))]
        mouth_px = np.nanmedian(d["mouth_px"][d["keep"]]) if "mouth_px" in d else 100.0
        mouth_geo = bool(mouth_px >= 60)  # small faces: lip geometry too noisy, eyes/brows only
        off = resting_offset(mp, geo, names, relaxed, mouth_geo)
        d["labels_hyb"] = hybrid(mp, np.nan_to_num(geo), names, off, mouth_geo)
        d["hyb_mouth_geo"] = mouth_geo
        d["face_ok_hyb"] = ok
        d["hyb_offset"] = off.astype(np.float32)
        if "labels_geo_lo" in d and "labels_lo" in d:
            d["labels_hyb_lo"] = hybrid(d["labels_lo"][:, cols], np.nan_to_num(d["labels_geo_lo"]), names, off,
                                        mouth_geo)
            d["face_ok_hyb_lo"] = d["face_ok_lo"] & d["face_ok_geo_lo"]
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **d)
        os.replace(tmp, path)
        print(f"{os.path.basename(path)}: hybrid labels ({'full' if mouth_geo else 'eyes/brows only'}, "
              f"mouth {mouth_px:.0f} px), resting offsets "
              f"squint {off[names.index('eyeSquintLeft')]:.2f} pucker {off[names.index('mouthPucker')]:.2f}"
              f"{'' if 'labels_hyb_lo' in d else ' (no low-res pass yet)'}")


if __name__ == "__main__":
    main()
