"""Compare retargeted FLAME labels (labels_flame) with the MediaPipe labels of a clip (.venv-smirk).

    .venv-smirk\\Scripts\\python tools/flame_labels_check.py data/processed/<name>.npz VIDEO \\
        --flame data/processed/<name>_flame.npz [--crop x0,y0,x1,y1] [--sheet out/flame_vs_mp.jpg]

Prints per shape (on kept frames where both trackers have a face): mean, std, correlation,
frame-to-frame jitter; then the check_labels.py speech-sync test on jawOpen from both.
Renders a contact sheet: video crop | ICT head from MediaPipe labels | ICT head from
labels_flame, for frames with smiles, wide-open mouths and closed-lip moments in speech.
"""

import argparse
import os
import sys

import av
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_labels import sync_curve  # noqa: E402
from prepare_data import FPS, audio_db  # noqa: E402


def jitter(x, m):
    t = m[1:-1] & m[:-2] & m[2:]
    return np.sqrt(((x[1:-1] - (x[:-2] + x[2:]) / 2)[t] ** 2).mean(0) / 1.5)


def sync_report(jaw, db, mask, label):
    lags, cs = sync_curve(jaw, db, mask)
    k = int(np.argmax(cs))
    contrast = (cs[k] - np.median(cs)) / (np.std(cs) + 1e-9)
    print(f"  {label:22s} peak corr {cs[k]:+.2f} at {lags[k]:+.2f}s (stands out {contrast:.1f} sd)")


def pick_frames(lab, names, ok, db, k_each=2, gap=FPS * 2):
    """Indices: smiles, wide-open mouths, closed lips during speech, plus a couple of others."""
    def top(score, k):
        out = []
        for i in np.argsort(-score):
            if not ok[i] or not np.isfinite(score[i]):
                continue
            if all(abs(i - j) > gap for j in out + chosen):
                out.append(int(i))
            if len(out) == k:
                break
        return out
    chosen = []
    smile = lab[:, names.index("mouthSmileLeft")] + lab[:, names.index("mouthSmileRight")]
    groups = []
    for label, score in [
        ("smile", smile),
        ("wide open", lab[:, names.index("jawOpen")]),
        ("closed lips in speech", np.where(db > np.percentile(db, 95) - 15,
                                           -lab[:, names.index("jawOpen")], -np.inf)),
        ("pucker", lab[:, names.index("mouthPucker")]),
    ]:
        idx = top(score, k_each)
        chosen += idx
        groups += [(label, i) for i in idx]
    return groups


def video_crop(video, t, crop, affine):
    with av.open(video) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, t - 2) / s.time_base), stream=s)
        best = None
        for fr in c.decode(s):
            ft = float(fr.pts * s.time_base)
            if ft >= t - 0.5 / 30:
                best = fr.to_ndarray(format="rgb24")
                break
    if crop:
        h, w = best.shape[:2]
        best = best[int(crop[1] * h): int(crop[3] * h), int(crop[0] * w): int(crop[2] * w)]
    if affine is not None and np.isfinite(affine).all():
        return cv2.warpAffine(best, affine.astype(np.float64), (224, 224))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("video")
    ap.add_argument("--flame", help="the _flame.npz (for the crop boxes of the sheet)")
    ap.add_argument("--crop")
    ap.add_argument("--sheet", default=os.path.join(ROOT, "out", "flame_vs_mp.jpg"))
    ap.add_argument("--size", type=int, default=300)
    a = ap.parse_args()
    crop = tuple(float(v) for v in a.crop.split(",")) if a.crop else None

    d = np.load(a.npz)
    all_names = list(d["names"])
    names = [x for x in all_names if x != "_neutral"]
    lab = d["labels"][:, [all_names.index(x) for x in names]]
    fl = d["labels_flame"]
    n = len(lab)
    ok = d["keep"] & d["face_ok"] & d["face_ok_flame"]
    print(f"{os.path.basename(a.npz)}: {ok.sum()} kept frames with both ({ok.sum() / FPS / 60:.1f} min)")
    jf, jm = jitter(fl, ok), jitter(lab, ok)
    print(f"{'shape':20s} {'meanF':>6s} {'meanMP':>6s} {'stdF':>6s} {'stdMP':>6s} {'corr':>6s} {'jitF':>6s} {'jitMP':>6s}")
    for i, nm in enumerate(names):
        x, y = fl[ok, i], lab[ok, i]
        c = np.corrcoef(x, y)[0, 1] if x.std() > 1e-6 and y.std() > 1e-6 else np.nan
        print(f"{nm:20s} {x.mean():6.3f} {y.mean():6.3f} {x.std():6.3f} {y.std():6.3f} {c:6.2f} {jf[i]:6.3f} {jm[i]:6.3f}")
    sf, sm = fl[ok].std(0), lab[ok].std(0)
    print(f"shapes with std > 0.02: FLAME {(sf > 0.02).sum()}, MediaPipe {(sm > 0.02).sum()} of {len(names)}")

    db = audio_db(d["audio"], n)
    mask = d["keep"] & d["face_ok_flame"] & (db > np.percentile(db, 95) - 20)
    if "voice_owner" in d:
        mask &= d["voice_owner"][:n] == (str(d["role"]) if "role" in d else "a")
    print("speech sync (check_labels.py):")
    sync_report(lab[:, names.index("jawOpen")], db, mask, "MediaPipe jawOpen")
    sync_report(fl[:, names.index("jawOpen")], db, mask, "FLAME jawOpen")
    opening = fl[:, names.index("jawOpen")] - fl[:, names.index("mouthClose")]
    sync_report(opening, db, mask, "FLAME jawOpen-mouthClose")

    # ---- contact sheet ----
    from realtime.head3d import Head3D
    head = Head3D(names, size=a.size)
    F = np.load(a.flame) if a.flame else None
    src = d["drift_src"] if "drift_src" in d else np.arange(n, dtype=float)
    rows = []
    for label, i in pick_frames(lab, names, ok, db):
        t = float(src[i]) / FPS  # video time this (re-timed) label came from
        aff = None
        if F is not None:
            j = int(np.argmin(np.abs(F["frame_times"] - t)))
            aff = F["crop_affine"][j]
        img = cv2.resize(video_crop(a.video, t, crop, aff), (a.size, a.size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        tiles = [img]
        for w, tag in ((lab[i], "MediaPipe"), (fl[i], "FLAME")):
            head.update(w)
            r = head.render()
            cv2.putText(r, tag, (8, a.size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tiles.append(r)
        jo = names.index("jawOpen")
        mc = names.index("mouthClose")
        cv2.putText(tiles[0], f"{label} t={t:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(tiles[0], f"jaw MP {lab[i, jo]:.2f} F {fl[i, jo]:.2f} close F {fl[i, mc]:.2f}",
                    (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        rows.append(np.hstack(tiles))
    os.makedirs(os.path.dirname(os.path.abspath(a.sheet)), exist_ok=True)
    cv2.imwrite(a.sheet, np.vstack(rows))
    print("contact sheet ->", a.sheet)


if __name__ == "__main__":
    main()
