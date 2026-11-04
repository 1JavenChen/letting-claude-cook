"""Make the ChArUco calibration board, sized for a screen at its native resolution.

    .venv\\Scripts\\python tools/make_board.py            # 13" iPad Air: 2732x2048 at 264 ppi
    .venv\\Scripts\\python tools/make_board.py --print    # US Letter sheet at 300 dpi instead

A ChArUco board is a checkerboard with a coded marker in every white square, so
each corner can be identified even when a camera sees only part of the board.
Shown full-screen on the iPad at native resolution, its squares have an exact,
known size (pixels / pixels-per-inch), which is what calibration relies on.

Writes data/calib/board.png and data/calib/board.json (the board's geometry,
read by tools/calibrate.py).
"""

import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "calib")
DICT = cv2.aruco.DICT_4X4_50  # 4x4 codes: bigger cells, readable through webcam blur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print", action="store_true", help="US Letter at 300 dpi instead of the iPad")
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--rows", type=int, default=5)
    args = p.parse_args()

    if args.print:
        width, height, ppi = 3300, 2550, 300  # 11 x 8.5 in, landscape
    else:
        width, height, ppi = 2732, 2048, 264  # 13" iPad Air, landscape
    square_px = min((width - 80) // args.cols, (height - 80) // args.rows)
    marker_px = int(square_px * 0.8)
    mm_per_px = 25.4 / ppi

    board = cv2.aruco.CharucoBoard((args.cols, args.rows), square_px * mm_per_px, marker_px * mm_per_px,
                                   cv2.aruco.getPredefinedDictionary(DICT))
    bw, bh = args.cols * square_px, args.rows * square_px
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    canvas = np.full((height, width), 255, np.uint8)  # board centered on a white screen
    y0, x0 = (height - bh) // 2, (width - bw) // 2
    canvas[y0:y0 + bh, x0:x0 + bw] = img

    os.makedirs(OUT, exist_ok=True)
    name = "board_print" if args.print else "board"
    cv2.imwrite(os.path.join(OUT, name + ".png"), canvas)
    meta = {"dictionary": "DICT_4X4_50", "cols": args.cols, "rows": args.rows,
            "square_mm": square_px * mm_per_px, "marker_mm": marker_px * mm_per_px,
            "image_px": [width, height], "ppi": ppi,
            "display": "US Letter print at 100% scale" if args.print else
            "13-inch iPad Air, full screen, native resolution (no zoom)"}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}.png: {args.cols}x{args.rows} squares of {meta['square_mm']:.2f} mm "
          f"({square_px} px at {ppi} ppi) -> {OUT}")


if __name__ == "__main__":
    main()
