"""Bring a Live Link Face take (iPhone video + TrueDepth) into a multi-camera session.

    .venv\\Scripts\\python tools/iphone_take.py takes/<date>_<slate>_<take> data/raw/session_<...>

1. Unpack: the take's depth is Oodle-compressed, so Unreal's Capture Manager
   ingest turns it into EXR depth images + PNG video frames (opens the editor
   for a few minutes, skipped if already done).
2. Sync: finds where the session's mic recording sits in the iPhone's audio
   (cross-correlation, checked on several chunks), giving every iPhone video and
   depth frame a time on the session clock.
3. Calibrate: finds the ChArUco board in iPhone frames taken at the same moments
   as the front webcam's, and solves where the iPhone sits relative to it.

Writes <session>/iphone.json (paths, clock offset, intrinsics, pose) and
<session>/iphone_times.npz (video_t, depth_t, arkit_t on the session clock).
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import detect, load_board, pair_extrinsics, read_stream  # noqa: E402
from calibrate import intrinsics as calib_intrinsics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(os.environ["LOCALAPPDATA"], "CaptureManager", "Media")
SAVED = os.path.join(ROOT, "unreal", "VRFace", "Saved")


def media_dir(take):
    slate_take = os.path.basename(take.rstrip("/\\")).split("_", 1)[1]
    hits = glob.glob(os.path.join(MEDIA, "*", "*", slate_take))
    return hits[0] if hits and glob.glob(os.path.join(hits[0], "Depth", "*", "*.exr")) else None


def ingest(take):
    with open(os.path.join(SAVED, "vrface_ingest_request.txt"), "w") as f:
        f.write(os.path.abspath(take).replace("\\", "/") + "\n")
    print("unpacking in Unreal (the editor opens and closes by itself, ~7 min)...")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "unreal", "run_editor.py"),
                    os.path.join(ROOT, "tools", "unreal", "ingest_takes_gui.py")], capture_output=True)
    with open(os.path.join(SAVED, "vrface_ingest_result.txt")) as f:
        print(f.read())


def frame_log(take):
    """Times (s, iPhone audio clock) of each video, depth and ARKit blendshape frame (B rows, one per
    row of the take's *_cal.csv / *_raw.csv in Live Link (ARKit) mode)."""
    rows = [line.split(",") for line in open(os.path.join(take, "frame_log.csv")) if line.strip()]
    t0 = next(int(r[2]) / int(r[3]) for r in rows if r[0] == "A")
    times = {k: np.array([int(r[2]) / int(r[3]) - t0 for r in rows if r[0] == k]) for k in "VDB"}
    return times["V"], times["D"], times["B"]


def mono(path, rate=8000):
    x, r = sf.read(path)
    x = x if x.ndim == 1 else x.mean(1)
    return ss.resample_poly(x, rate, r)


def audio_offset(phone_wav, mic_wav, rate=8000):
    """Where the mic recording sits in the iPhone audio: iPhone audio time = off + (1 + drift) * mic
    time (the two clocks run at slightly different speeds, ~30 ppm), from chunks across the whole
    recording. Returns (off, drift, worst chunk's distance from the line in s, chunks used)."""
    a, b = mono(phone_wav, rate), mono(mic_wav, rate)
    env = lambda x: np.log(np.abs(ss.hilbert(ss.resample_poly(x, 1000, rate))) + 1e-4)  # noqa: E731
    ea, eb = env(a), env(b)
    c = ss.correlate(ea - ea.mean(), eb - eb.mean(), "full")
    coarse = (np.argmax(c) - (len(eb) - 1)) / 1000
    fine = []
    for s0 in np.linspace(1, len(b) / rate - 4, 16):  # refine on the raw waveform, chunk by chunk
        seg = b[int(s0 * rate):int((s0 + 3) * rate)]
        c0 = int((coarse + s0 - 0.5) * rate)
        if c0 < 0 or c0 + 4 * rate > len(a):
            continue
        win = a[c0:c0 + 4 * rate]
        cc = ss.correlate(win, seg, "valid")
        norm = np.sqrt(ss.correlate(win ** 2, np.ones(len(seg)), "valid") * np.sum(seg ** 2)) + 1e-9
        k = int(np.argmax(cc / norm))
        if (cc / norm)[k] > 0.15:
            fine.append((s0, (c0 + k) / rate - s0))
    if len(fine) < 3:
        sys.exit(f"audio sync unsure (coarse guess {coarse:.2f} s) - was the iPhone recording during the session?")
    m, o = np.array(fine).T
    for _ in range(2):  # line fit, dropping chunks that matched an echo or a repeated sound
        drift, off = np.polyfit(m, o, 1)
        res = np.abs(o - (off + drift * m))
        keep = res < max(0.01, 3 * np.median(res))
        m, o = m[keep], o[keep]
    drift, off = np.polyfit(m, o, 1)
    return float(off), float(drift), float(np.abs(o - (off + drift * m)).max()), len(m)


def intrinsics(media):
    """Apple's factory intrinsics (normalized in the file) as pixel K matrices, per camera."""
    with open(glob.glob(os.path.join(media, "Calibration", "*", "calibration.json"))[0]) as f:
        cals = json.load(f)["Calibrations"]
    out = {}
    for c in cals:
        w, h = c["Dimensions"]["Width"], c["Dimensions"]["Height"]
        (fx, fy), (cx, cy) = c["FocalLength"], c["PrincipalPoint"]
        out[c["CameraId"]] = {"K": [[fx * w, 0, cx * w], [0, fy * h, cy * h], [0, 0, 1]], "size": [w, h]}
    return out


def mov_audio(take, out_wav):
    """Video-only takes: the audio lives in the .mov."""
    import imageio_ffmpeg
    mov = glob.glob(os.path.join(take, "*.mov"))[0]
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", mov, "-vn", "-ac", "1",
                    "-ar", "48000", out_wav], check=True)
    return out_wav


def video_size(take):
    import av
    with av.open(glob.glob(os.path.join(take, "*.mov"))[0]) as c:
        s = c.streams.video[0]
        return (s.width, s.height)  # MetaHuman Animator mode 1080x1920, ARKit mode 1080x1440


def mirrored(take):
    """Live Link (ARKit) mode saves the front camera's video mirrored (selfie view); MHA mode doesn't."""
    return bool(glob.glob(os.path.join(take, "*_cal.csv")))


def detect_frames(board, take, video, wanted):
    """{frame index: board detection or None} for the wanted iPhone frames (PNGs, or the .mov)."""
    out = {}
    if video:
        for k in wanted:
            out[k] = detect(board, cv2.imread(video[k], cv2.IMREAD_GRAYSCALE))
        return out
    import av
    flip = mirrored(take)
    last = max(wanted)
    with av.open(glob.glob(os.path.join(take, "*.mov"))[0]) as c:
        for k, f in enumerate(c.decode(c.streams.video[0])):
            if k in wanted:
                g = f.to_ndarray(format="gray")
                out[k] = detect(board, np.ascontiguousarray(g[:, ::-1]) if flip else g)
            if k >= last:
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("take")
    p.add_argument("session")
    p.add_argument("--seconds", type=float, default=60, help="board part at the start of the session")
    args = p.parse_args()

    has_depth = os.path.exists(os.path.join(args.take, "depth_data.bin"))
    vt, dt, bt = frame_log(args.take)
    media, video, K = None, [], None
    if has_depth:
        media = media_dir(args.take)
        if media is None:
            ingest(args.take)
            media = media_dir(args.take) or sys.exit("ingest produced no depth images")
        video = sorted(glob.glob(os.path.join(media, "Video", "*", "*.png")))
        depth = sorted(glob.glob(os.path.join(media, "Depth", "*", "*.exr")))
        assert len(vt) == len(video) and len(dt) == len(depth), "frame log doesn't match the unpacked frames"
        wav = glob.glob(os.path.join(media, "Audio", "*", "*.wav"))[0]
        K = intrinsics(media)  # Apple's factory values; a board fit agreed to 0.2 px in focal length
    else:
        print("video-only take (no depth recorded): reading the .mov directly")
        wav = mov_audio(args.take, os.path.join(args.session, "iphone_audio.wav"))

    with open(os.path.join(args.session, "session.json")) as f:
        sess = json.load(f)
    off, drift, spread, n = audio_offset(wav, os.path.join(args.session, sess["mic"]["file"]))
    first = sess["mic"]["first_sample_s"]
    to_session = lambda t: first + (t - off) / (1 + drift)  # noqa: E731  iPhone audio clock -> session clock
    shift = to_session(0.0)
    print(f"sync: session mic starts {off:.3f} s into the iPhone audio, clocks differ by "
          f"{drift * 1e6:.0f} ppm ({n} chunks within {spread * 1000:.0f} ms of that); "
          f"iPhone t + {shift:.3f} s = session t at the start")
    vt, dt, bt = to_session(vt), to_session(dt), to_session(bt)

    # board: iPhone frames near each sampled webcam frame
    board, _ = load_board()
    with open(os.path.join(args.session, "calib.json")) as f:
        cal = json.load(f)["cameras"]
    lags = range(-24, 7)  # 60 fps frames: -400..+100 ms
    cams = {}
    for name in cal:
        frames = read_stream(os.path.join(args.session, name + ".mkv"), 0, args.seconds, 5)
        dets = [(t, d) for t, d in ((t, detect(board, g)) for t, g in frames) if d is not None]
        cams[name] = {"dets": dets, "near": [int(np.argmin(np.abs(vt - t))) for t, _ in dets]}
    wanted = {k + lag for c in cams.values() for k in c["near"] for lag in lags if 0 <= k + lag < len(vt)}
    print(f"looking for the board in {len(wanted)} iPhone frames...")
    phone = detect_frames(board, args.take, video, wanted)
    if K is None:  # no factory calibration in a video-only take: solve the lens from the board views
        views = [d for d in phone.values() if d is not None][::3]
        size = video_size(args.take)
        rms, Kv, dv, nv = calib_intrinsics(board, views, size)
        print(f"iphone lens from {nv} board views: focal {Kv[0, 0]:.0f} px, error {rms:.2f} px")
        K = {"Video": {"K": Kv.tolist(), "size": list(size)}}
        dp = dv.ravel()
    else:
        dp = np.zeros(5)
    Kp = np.array(K["Video"]["K"])
    out = {"take": os.path.abspath(args.take), "media": media, "shift_s": shift, "drift": drift,
           "sync_spread_s": spread, "clock_note": "session t = shift_s + iPhone audio t / (1 + drift)",
           "video": {**K["Video"], "dist": dp.tolist()}, "webcams": {}}
    if has_depth:
        out.update(depth=K["Depth"],
                   depth_note="EXR channel Y, cm, same viewpoint as video (depth px * 2 = video px)")
    else:
        out["mov"] = os.path.abspath(glob.glob(os.path.join(args.take, "*.mov"))[0])
        out["mirrored"] = mirrored(args.take)  # every reader flips the frames back (x -> w - 1 - x)

    for name, c in cams.items():
        Kc, dc = np.array(cal[name]["K"]), np.array(cal[name]["dist"])

        def pairs_at(lag):  # lag: iPhone frames shifted against the webcam's timestamps
            return [(d, phone[k + lag]) for (_, d), k in zip(c["dets"], c["near"]) if phone.get(k + lag) is not None]

        def fit(lag):
            return pair_extrinsics(board, pairs_at(lag), Kc, dc, Kp, dp, tuple(cal[name]["size"]))

        # Audio puts all on one clock, but a webcam stamps each frame some time after it was exposed.
        # The board was moving, so the lag where both views agree best is that delay.
        errs = []
        for lag in lags:
            try:
                errs.append(fit(lag)[0])
            except SystemExit:
                errs.append(np.inf)
        if not np.isfinite(min(errs)):
            print(f"{name}: too few board views shared with the iPhone to place it")
            continue
        best = lags[int(np.argmin(errs))]
        delay = -best * np.median(np.diff(vt))
        rms, R, T, n = fit(best)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f"{name}: frames stamped {delay * 1000:.0f} ms after they happened (board agreement "
              f"{errs[lags.index(0)]:.2f} px uncorrected -> {rms:.2f} px); iPhone is {np.linalg.norm(T):.0f} mm "
              f"away, {angle:.1f} deg rotated ({n} shared views)")
        # R, t_mm: a point X in this webcam's frame is R @ X + t in the iPhone's frame
        out["webcams"][name] = {"delay_s": delay, "R": R.tolist(), "t_mm": T.ravel().tolist(), "rms_px": rms}
        if name == "front":  # older scripts read these
            out.update(front_delay_s=delay, R=R.tolist(), t_mm=T.ravel().tolist(), stereo_rms_px=rms)
    with open(os.path.join(args.session, "iphone.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(args.session, "iphone_times.npz"), video_t=vt, depth_t=dt, arkit_t=bt)
    print(f"-> {os.path.join(args.session, 'iphone.json')}")


if __name__ == "__main__":
    main()
