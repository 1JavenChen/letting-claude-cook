"""Calibrate a multi-camera session from the ChArUco board shown at its start.

    .venv\\Scripts\\python tools/calibrate.py data/raw/session_<...> [--seconds 60]
    .venv\\Scripts\\python tools/calibrate.py --selftest     # check on synthetic cameras

For each camera: finds the board's corners (each one identified by its coded
marker), then solves
  - intrinsics: focal length, image centre and lens distortion of that camera,
  - extrinsics: where the side camera sits relative to the front camera
    (rotation + translation in mm), from moments when both saw the board.
Those let us triangulate any face point seen by both cameras into true 3D.

Writes <session>/calib.json; reports the reprojection error per camera
(well calibrated: under ~0.5 px).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_CORNERS = 8


def load_board(name="board"):
    with open(os.path.join(ROOT, "data", "calib", name + ".json")) as f:
        m = json.load(f)
    board = cv2.aruco.CharucoBoard((m["cols"], m["rows"]), m["square_mm"], m["marker_mm"],
                                   cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, m["dictionary"])))
    return board, m


def detect(board, gray):
    """(corner ids [N], image points [N,2]) of the board in one grayscale frame, or None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    return ids.ravel(), corners.reshape(-1, 2)


def intrinsics(board, detections, size):
    obj, img = [], []
    for ids, pts in detections:
        o, i = board.matchImagePoints(pts.reshape(-1, 1, 2), ids.reshape(-1, 1))
        if o is not None and len(o) >= MIN_CORNERS:
            obj.append(o.astype(np.float32))
            img.append(i.astype(np.float32))
    if len(obj) < 5:
        raise SystemExit(f"only {len(obj)} usable board views - show the board longer / closer")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return rms, K, dist, len(obj)


def pair_extrinsics(board, pairs, K1, d1, K2, d2, size):
    """Pose of camera 2 relative to camera 1 from simultaneous detections [(det1, det2)]."""
    obj, i1, i2 = [], [], []
    for (ids1, p1), (ids2, p2) in pairs:
        common, a, b = np.intersect1d(ids1, ids2, return_indices=True)
        if len(common) < MIN_CORNERS:
            continue
        o = board.getChessboardCorners()[common].astype(np.float32)
        obj.append(o)
        i1.append(p1[a].astype(np.float32))
        i2.append(p2[b].astype(np.float32))
    if len(obj) < 3:
        raise SystemExit(f"only {len(obj)} moments where both cameras saw the board - "
                         "hold it where both can see it for a few seconds")
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(obj, i1, i2, K1, d1, K2, d2, size,
                                              flags=cv2.CALIB_FIX_INTRINSIC)
    return rms, R, T, len(obj)


def read_stream(path, stream_index, seconds, step):
    """[(time_s, gray frame)] from one video stream, every `step`-th frame of the first `seconds`."""
    import av
    out = []
    with av.open(path) as c:
        s = c.streams.video[stream_index]
        first = None
        for k, f in enumerate(c.decode(s)):
            t = float(f.pts * s.time_base)  # shared session clock (used to pair cameras)
            first = t if first is None else first
            if t - first > seconds:  # `seconds` counts from this video's first frame
                break
            if k % step == 0:
                out.append((t, f.to_ndarray(format="gray")))
    return out


def calibrate_session(session, seconds, step):
    board, meta = load_board()
    cams = {}
    for name, idx in (("front", 0), ("side", 1)):
        # one file per camera (current recorder) or one file with a stream per camera (first version)
        own = os.path.join(session, name + ".mkv")
        frames = (read_stream(own, 0, seconds, step) if os.path.exists(own)
                  else read_stream(os.path.join(session, "cams.mkv"), idx, seconds, step))
        dets = [(t, detect(board, g)) for t, g in frames]
        cams[name] = {"size": frames[0][1].shape[::-1], "dets": [(t, d) for t, d in dets if d is not None]}
        print(f"{name}: board found in {len(cams[name]['dets'])} of {len(frames)} sampled frames")
    result = {"board": meta, "cameras": {}}
    for name, c in cams.items():
        rms, K, dist, n = intrinsics(board, [d for _, d in c["dets"]], c["size"])
        c.update(K=K, dist=dist)
        result["cameras"][name] = {"K": K.tolist(), "dist": dist.ravel().tolist(), "size": list(c["size"]),
                                   "R": np.eye(3).tolist(), "t_mm": [0, 0, 0], "rms_px": rms}
        print(f"{name}: intrinsics from {n} views, reprojection error {rms:.2f} px")
    # pair frames captured within half a frame of each other (shared wall clock)
    side = cams["side"]["dets"]
    st = np.array([t for t, _ in side])
    pairs = []
    for t, d in cams["front"]["dets"]:
        j = int(np.argmin(np.abs(st - t))) if len(st) else -1
        if j >= 0 and abs(st[j] - t) < 0.017:
            pairs.append((d, side[j][1]))
    rms, R, T, n = pair_extrinsics(board, pairs, cams["front"]["K"], cams["front"]["dist"],
                                   cams["side"]["K"], cams["side"]["dist"], cams["front"]["size"])
    result["cameras"]["side"].update(R=R.tolist(), t_mm=T.ravel().tolist(), stereo_rms_px=rms)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"side vs front: {np.linalg.norm(T):.0f} mm apart, {angle:.1f} deg rotated "
          f"({n} shared views, error {rms:.2f} px)")
    with open(os.path.join(session, "calib.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {os.path.join(session, 'calib.json')}")


def selftest():
    """Render the board into two virtual cameras with known poses; check we recover them."""
    board, meta = load_board()
    img = cv2.imread(os.path.join(ROOT, "data", "calib", "board.png"), cv2.IMREAD_GRAYSCALE)
    px_mm = 25.4 / meta["ppi"]
    bw, bh = meta["cols"] * meta["square_mm"], meta["rows"] * meta["square_mm"]
    x0 = (img.shape[1] * px_mm - bw) / 2  # board offset on the screen image, mm
    y0 = (img.shape[0] * px_mm - bh) / 2
    size = (1920, 1080)
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    R_side, _ = cv2.Rodrigues(np.array([0.0, np.radians(-35), 0.0]))
    t_side = np.array([[350.0], [0.0], [80.0]])  # side camera ~35 cm right, rotated 35 deg
    rng = np.random.default_rng(0)
    pairs, d1s, d2s = [], [], []
    for _ in range(40):
        rv = rng.normal(0, 0.35, 3)
        R_b, _ = cv2.Rodrigues(rv)
        t_b = np.array([[rng.uniform(-80, 80)], [rng.uniform(-60, 60)], [rng.uniform(450, 750)]])
        dets = []
        for R_c, t_c in ((np.eye(3), np.zeros((3, 1))), (R_side, t_side)):
            R, t = R_c @ R_b, R_c @ t_b + t_c          # board -> camera
            # screen pixel (u,v) -> board mm (u*px - x0, v*px - y0, 0) -> camera -> image
            A = np.array([[px_mm, 0, -x0], [0, px_mm, -y0], [0, 0, 1]])
            H = K @ np.column_stack([R[:, 0], R[:, 1], t.ravel()]) @ A
            frame = cv2.warpPerspective(img, H, size, borderValue=90)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            dets.append(detect(board, frame))
        if dets[0] is not None:
            d1s.append(dets[0])
        if dets[1] is not None:
            d2s.append(dets[1])
        if dets[0] is not None and dets[1] is not None:
            pairs.append(tuple(dets))
    rms1, K1, dist1, n1 = intrinsics(board, d1s, size)
    rms2, K2, dist2, n2 = intrinsics(board, d2s, size)
    rms, R, T, n = pair_extrinsics(board, pairs, K1, dist1, K2, dist2, size)
    ang_err = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ R_side.T)[0]))
    print(f"views: front {n1}, side {n2}, shared {n}")
    print(f"focal length: true 1400, found {K1[0, 0]:.1f} / {K2[0, 0]:.1f}")
    print(f"side camera position error {np.linalg.norm(T - t_side):.2f} mm, rotation error {ang_err:.3f} deg, "
          f"reprojection {rms1:.2f}/{rms2:.2f}/{rms:.2f} px")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", nargs="?")
    p.add_argument("--seconds", type=float, default=60, help="board part at the start of the session")
    p.add_argument("--step", type=int, default=5, help="use every Nth frame")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    if not args.session:
        sys.exit("give a session folder, or --selftest")
    calibrate_session(args.session, args.seconds, args.step)


if __name__ == "__main__":
    main()
