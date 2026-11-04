"""Record a multi-camera training session: both webcams + the mic, on one clock.

    .venv\\Scripts\\python tools/record_multicam.py --minutes 15
    .venv\\Scripts\\python tools/record_multicam.py --list          # device names and indexes

Every camera is read by its own thread and every frame is stamped with the
same high-resolution clock (time.perf_counter) the instant it arrives; the
microphone's audio blocks are stamped the same way. (The first version used
one ffmpeg process for all devices; it delivered one camera's frames in
clumps of up to 14 with identical timestamps, which breaks matching frames
between cameras.) Video is encoded on the GPU (NVENC) at high quality.
Exposure can be locked (--exposure) so a bright screen in view doesn't make
the camera darken the whole scene, or slow its frame rate in dim light.

Output: data/raw/session_<date>_<time>/
  front.mkv, side.mkv   one file per camera; frame times in ms on the shared clock
  mic.wav               the microphone
  session.json          devices, settings, clock offsets, frame statistics

Recording protocol (talking part: docs/recording_prompts.md):
  1. Start the iPhone (Live Link Face, MetaHuman Animator mode), then this script.
  2. Clap once, clearly, in view of every camera (syncs the iPhone later).
  3. Board (data/calib/board.png full screen on the iPad, brightness ~50%):
     hold it STILL for about a second at each position, then move to the next:
     near, far, tilted, in the corners - and a few positions where every camera
     sees it. Moving while held up blurs it and it can't be detected.
  4. Put the iPad away; don't move any camera after this.
  5. Talk, make faces, turn your head slowly. Press Ctrl+C to stop early.
"""

import argparse
import datetime
import json
import os
import queue
import subprocess
import threading
import time
from fractions import Fraction

import av
import cv2
import imageio_ffmpeg
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def devices():
    """[(name, kind)] of DirectShow devices in the order OpenCV numbers cameras."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                       capture_output=True, text=True, errors="replace")
    out = []
    for line in r.stderr.splitlines():
        if line.count('"') >= 2 and "Alternative name" not in line:
            kind = "video" if "(video)" in line else "audio" if "(audio)" in line else "other"
            out.append((line.split('"')[1], kind))
    return out


def camera_index(name):
    """DirectShow index of a camera by name."""
    for _ in range(3):  # a camera released a moment ago can be missing from the list briefly
        cams = [n for n, k in devices() if k == "video"]
        if name in cams:
            break
        time.sleep(1.5)
    if name not in cams:
        raise SystemExit(f"Camera not connected: {name}\nConnected cameras: {', '.join(cams) or 'none'}"
                         "\n(Is the laptop on its dock? Use --front/--side to pick others.)")
    return cams.index(name)


def msmf_layout():
    """Media Foundation numbers cameras differently from DirectShow and can't report names, so
    identify them by ability: the C925e has a 2304x1536 mode, the laptop camera tops out at
    1920x1080. Returns {index: max width}.

    Opening the C925e through Media Foundation takes ~40 s, so the answer is cached in
    data/calib/msmf_layout.json and reused while the same cameras are connected."""
    cache = os.path.join(ROOT, "data", "calib", "msmf_layout.json")
    cams = [n for n, k in devices() if k == "video"]
    try:
        with open(cache) as f:
            saved = json.load(f)
        if saved["cameras"] == cams:
            return {int(k): v for k, v in saved["layout"].items()}
    except (OSError, ValueError, KeyError):
        pass
    print("Identifying cameras (first time only, ~1 minute)...")
    out = {}
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            break
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2304)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        out[i] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        time.sleep(0.5)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"cameras": cams, "layout": out}, f)
    return out


def msmf_index(device, layout):
    """Media Foundation index for a device name: the C925e is the camera with the large mode."""
    big = [i for i, w in layout.items() if w > 1920]
    small = [i for i, w in layout.items() if w <= 1920]
    if "C925e" in device and big:
        return big[0]
    if small:
        return small[0]
    raise SystemExit(f"Could not find {device} with Media Foundation (cameras found: {layout}); "
                     "try --backend dshow")


class Camera:
    """Reads one camera on its own thread; a second thread encodes to <name>.mkv."""

    def __init__(self, label, device, out_dir, size, fps, exposure, t0, backend="msmf", layout=None):
        self.label, self.t0 = label, t0
        w, h = (int(v) for v in size.split("x"))
        # Media Foundation delivers the C925e at a full 30 fps at 1080p; DirectShow through
        # OpenCV fell back to its uncompressed mode (5 fps) or ~19 fps.
        if backend == "msmf":
            self.cap = cv2.VideoCapture(msmf_index(device, layout), cv2.CAP_MSMF)
        else:
            self.cap = cv2.VideoCapture(camera_index(device), cv2.CAP_DSHOW,
                                        [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
                                         cv2.CAP_PROP_FRAME_WIDTH, w, cv2.CAP_PROP_FRAME_HEIGHT, h])
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(10):  # some cameras send black frames while starting up
            ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"{label}: could not read from {device}")
        cv2.imwrite(os.path.join(out_dir, label + "_preview.jpg"), frame)  # check which camera is which
        self.size = (frame.shape[1], frame.shape[0])
        self.out = av.open(os.path.join(out_dir, label + ".mkv"), "w")
        self.stream = self.out.add_stream("h264_nvenc", rate=fps, options={"preset": "p5", "rc": "vbr", "cq": "18"})
        self.stream.width, self.stream.height = self.size
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, 1000)
        self.stream.codec_context.time_base = Fraction(1, 1000)  # frame pts are in milliseconds
        self.q = queue.Queue(maxsize=120)
        self.times, self.dropped, self.running = [], 0, True
        self.latest = frame  # newest frame, for the preview window
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer = threading.Thread(target=self._write, daemon=True)

    def start(self):
        self.reader.start()
        self.writer.start()

    def _read(self):
        while self.running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                continue
            try:
                self.latest = frame
                self.q.put_nowait((t, frame))
            except queue.Full:
                self.dropped += 1  # encoder fell behind: drop rather than stall the camera

    def _write(self):
        last_ms = -1
        while self.running or not self.q.empty():
            try:
                t, frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            ms = max(int(round((t - self.t0) * 1000)), last_ms + 1)
            last_ms = ms
            vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
            vf.pts = ms
            vf.time_base = Fraction(1, 1000)
            for pkt in self.stream.encode(vf):
                self.out.mux(pkt)
            self.times.append(t - self.t0)

    def stop(self):
        self.running = False
        self.reader.join(timeout=2)
        self.writer.join(timeout=30)
        for pkt in self.stream.encode():
            self.out.mux(pkt)
        self.out.close()
        self.cap.release()

    def stats(self):
        ts = np.array(self.times)
        if len(ts) < 2:
            return {"frames": len(ts)}
        gaps = np.diff(ts)
        return {"frames": len(ts), "fps": round(len(ts) / (ts[-1] - ts[0]), 2),
                "longest_gap_ms": round(float(gaps.max()) * 1000), "dropped": self.dropped,
                "size": list(self.size)}


class Mic:
    def __init__(self, name, out_dir, t0):
        import sounddevice as sd
        import soundfile as sf
        idx = next((i for i, d in enumerate(sd.query_devices())
                    if d["max_input_channels"] > 0 and name.lower() in d["name"].lower()), None)
        if idx is None:
            raise SystemExit(f"Microphone not connected: {name}")
        info = sd.query_devices(idx)
        self.rate = int(info["default_samplerate"])
        self.file = sf.SoundFile(os.path.join(out_dir, "mic.wav"), "w", self.rate, 1, "PCM_24")
        self.t0, self.first, self.level = t0, None, 0.0
        self.stream = sd.InputStream(device=idx, channels=1, samplerate=self.rate,
                                     blocksize=self.rate // 100, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        if self.first is None:  # clock time of the first sample of the file
            self.first = time.perf_counter() - frames / self.rate - self.t0
        self.file.write(indata[:, 0].copy())
        self.level = float(np.sqrt(np.mean(indata[:, 0] ** 2)))  # for the preview's level meter

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.file.close()


class Monitor:
    """Live preview: every camera side by side with its frame rate, how many board corners it
    sees right now (green = enough to calibrate), a mic level meter, and time left. Q stops."""

    TITLE = "Recording - press Q to stop"

    def __init__(self, cams, mic, width=640):
        self.cams, self.mic, self.width = cams, mic, width
        self.corners = {c.label: 0 for c in cams}
        self.last_detect = 0.0
        self.counts = {c.label: (0, time.perf_counter()) for c in cams}
        self.fps = {c.label: 0.0 for c in cams}
        try:
            from calibrate import detect, load_board
            self.board, self.detect = load_board()[0], detect
        except Exception:  # noqa: BLE001  (no board file: the preview still works)
            self.board = None
        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)

    def show(self, seconds_left):
        now = time.perf_counter()
        if self.board is not None and now - self.last_detect > 0.5:  # board check twice a second
            self.last_detect = now
            for c in self.cams:
                if c.latest is not None:
                    r = self.detect(self.board, cv2.cvtColor(c.latest, cv2.COLOR_BGR2GRAY))
                    self.corners[c.label] = 0 if r is None else len(r[0])
        tiles = []
        for c in self.cams:
            n0, t0 = self.counts[c.label]
            if now - t0 > 1:
                self.fps[c.label] = (len(c.times) - n0) / (now - t0)
                self.counts[c.label] = (len(c.times), now)
            img = c.latest if c.latest is not None else np.zeros((360, 640, 3), np.uint8)
            h = int(img.shape[0] * self.width / img.shape[1])
            tile = cv2.resize(img, (self.width, h))
            fps = self.fps[c.label]
            cv2.putText(tile, f"{c.label}  {fps:.0f} fps", (10, 28), 0, 0.8,
                        (60, 220, 60) if fps >= 27 else (40, 160, 255), 2)
            if self.board is not None:
                n = self.corners[c.label]
                cv2.putText(tile, f"board: {n} corners" if n else "board: not seen", (10, h - 14), 0, 0.7,
                            (60, 220, 60) if n >= 8 else (180, 180, 180), 2)
            tiles.append(tile)
        canvas = np.hstack(tiles)
        bar = np.zeros((44, canvas.shape[1], 3), np.uint8)
        if self.mic:
            lvl = min(1.0, self.mic.level * 8)
            cv2.rectangle(bar, (10, 12), (10 + int(300 * lvl), 32),
                          (60, 200, 60) if lvl < 0.9 else (40, 40, 230), -1)
            cv2.rectangle(bar, (10, 12), (310, 32), (120, 120, 120), 1)
            cv2.putText(bar, "mic", (320, 30), 0, 0.6, (200, 200, 200), 1)
        m, sec = divmod(max(0, int(seconds_left)), 60)
        cv2.putText(bar, f"REC  {m}:{sec:02d} left   (Q = stop)", (canvas.shape[1] - 330, 30), 0, 0.6,
                    (80, 80, 255), 2)
        cv2.imshow(self.TITLE, np.vstack([canvas, bar]))
        return (cv2.waitKey(30) & 0xFF) != ord("q")

    def close(self):
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--front", default="Logitech Webcam C925e")
    p.add_argument("--side", default="ASUS FHD webcam")
    p.add_argument("--mic", default="Focusrite", help="microphone name (substring)")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=float,
                   help="lock exposure (DirectShow log2 seconds, e.g. -6 = 1/64 s); default: automatic")
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--name", help="session folder name (default: session_<date>_<time>)")
    p.add_argument("--only", choices=["front", "side"], help="record just one camera (testing)")
    p.add_argument("--backend", choices=["msmf", "dshow"], default="msmf",
                   help="Windows camera system (msmf = Media Foundation: full 30 fps on the C925e)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-preview", action="store_true", help="don't show the live preview window")
    args = p.parse_args()
    if args.list:
        for i, (n, k) in enumerate(devices()):
            print(f"  {k:6s} {n}")
        return

    start = datetime.datetime.now()
    name = args.name or start.strftime("session_%Y%m%d_%H%M")
    out_dir = os.path.join(ROOT, "data", "raw", name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    labels = [args.only] if args.only else ["front", "side"]
    layout = msmf_layout() if args.backend == "msmf" else None
    print("Opening cameras (the C925e takes ~40 s to start)...")
    cams = [Camera(lb, getattr(args, lb), out_dir, args.size, args.fps, args.exposure, t0, args.backend, layout)
            for lb in labels]
    mic = Mic(args.mic, out_dir, t0) if args.mic else None
    for c in cams:
        c.start()
    if mic:
        mic.start()
    print(f"Recording {args.minutes:g} min to {out_dir}\nClap once, then show the board, holding it still "
          f"at each position. Ctrl+C stops early.")
    try:
        end = time.perf_counter() + args.minutes * 60
        monitor = None if args.no_preview else Monitor(cams, mic)
        last_print = 0.0
        while time.perf_counter() < end:
            if monitor:
                if not monitor.show(end - time.perf_counter()):
                    break  # q pressed in the preview window
            else:
                time.sleep(0.2)
            if time.perf_counter() - last_print > 1:
                last_print = time.perf_counter()
                print("  " + " | ".join(f"{c.label}: {len(c.times)} frames" for c in cams), end="\r")
        if monitor:
            monitor.close()
    except KeyboardInterrupt:
        pass
    print()
    for c in cams:
        c.stop()
    if mic:
        mic.stop()
    meta = {"start": start.isoformat(), "clock": "seconds since recording start (time.perf_counter)",
            "cameras": {c.label: {"device": getattr(args, c.label), "file": c.label + ".mkv", **c.stats()}
                        for c in cams},
            "mic": {"device": args.mic, "file": "mic.wav", "rate": mic.rate,
                    "first_sample_s": mic.first} if mic else None,
            "exposure": args.exposure, "backend": args.backend,
            "iphone": "recorded separately in Live Link Face (MetaHuman Animator mode); synced by the clap"}
    with open(os.path.join(out_dir, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for c in cams:
        s = c.stats()
        print(f"  {c.label}: {s.get('frames')} frames, {s.get('fps')} fps, longest gap "
              f"{s.get('longest_gap_ms')} ms, dropped {s.get('dropped')}")


if __name__ == "__main__":
    main()
