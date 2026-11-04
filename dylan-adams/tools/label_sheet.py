"""Contact sheet: video frame | 3D head driven by label set A | by label set B.

    .venv\\Scripts\\python tools/label_sheet.py data/processed/<name>.npz data/raw/<video> \\
        --times 34,68.9,149.6 --a labels --b labels_geo --out out/sheet.jpg [--crop ...]

For judging labels by what they look like on a face, not by correlations.
"""

import argparse
import os
import sys

import av
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.head3d import Head3D  # noqa: E402

FPS = 30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("video")
    p.add_argument("--times", required=True, help="label-timeline seconds, comma-separated")
    p.add_argument("--a", default="labels")
    p.add_argument("--b", default="labels_geo")
    p.add_argument("--crop")
    p.add_argument("--out", default=os.path.join(ROOT, "out", "label_sheet.jpg"))
    args = p.parse_args()

    d = np.load(args.npz)
    names_all = list(d["names"])
    names = [s for s in names_all if s != "_neutral"]
    head = Head3D(names, size=300)
    src = d["drift_src"] if "drift_src" in d else np.arange(len(d["labels"]), dtype=np.float32)
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
    c = av.open(args.video)
    s = c.streams.video[0]
    rows = []
    for t in (float(x) for x in args.times.split(",")):
        i = int(t * FPS)
        tv = src[i] / FPS
        c.seek(int(max(0, tv - 1) / s.time_base), stream=s)
        for f in c.decode(s):
            if f.pts * s.time_base >= tv:
                break
        img = f.to_ndarray(format="bgr24")
        if crop:
            h, w = img.shape[:2]
            img = img[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
        h, w = img.shape[:2]
        side = min(h, w)
        img = cv2.resize(img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2], (300, 300))
        cv2.putText(img, f"{t:.1f}s", (5, 20), 0, 0.6, (0, 255, 255), 2)
        tiles = [img]
        for key in (args.a, args.b):
            vals = d[key][i]
            if vals.shape[0] == len(names_all):
                vals = vals[[names_all.index(n) for n in names]]
            head.update(dict(zip(names, vals)))
            r = head.render()
            cv2.putText(r, key, (5, 290), 0, 0.5, (255, 255, 255), 1)
            tiles.append(r)
        rows.append(np.hstack(tiles))
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
