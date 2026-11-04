"""Real-time face from webcam + microphone, fused, at 30 fps.

    python live.py                              # default camera + mic, preview window
    python live.py --camera 1 --mic Focusrite   # pick devices (see --list)
    python live.py --unreal                     # also stream to Unreal (Live Link, UDP 11111)
    python live.py --file data/raw/x.mp4        # test on a recorded video instead
    python live.py --list                       # list cameras and microphones
    python live.py --view 2d                    # cartoon face instead of the 3D head

Preview keys:  1 fused   2 camera only   3 audio only
               n  calibrate: relax your face, look at the camera, press N (held ~2 s)
               m  pretend the mouth is hidden (e.g. by a headset / hand)
               q  quit

Every tick (1/30 s) it takes the last 3 s of audio and camera tracking, runs
wav2vec2 + the fusion model on the GPU, and outputs the frame from
`lookahead` frames ago (the model peeks that far into the future, ~133 ms).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS, save_animation  # noqa: E402
from model.encoders import ENCODERS  # noqa: E402
from model.fusion import FusionHead  # noqa: E402
from realtime.filters import OneEuro  # noqa: E402
from realtime.idle import Blinker, Hold  # noqa: E402
from realtime.livelink import LiveLinkSender  # noqa: E402
from realtime.preview import bars, draw_face  # noqa: E402
from realtime.sources import AudioRing, CameraSource, FaceTracker, FileSource, MicSource  # noqa: E402

MODES = {ord("1"): "fused", ord("2"): "camera", ord("3"): "audio"}
SHOW = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthSmileLeft", "mouthPressLeft",
        "mouthStretchLeft", "browInnerUp", "eyeBlinkLeft", "eyeBlinkRight"]


def list_devices():
    import sounddevice as sd
    print("Microphones:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["hostapi"] == 0:
            print(f"  {i}: {d['name']}")
    print("Cameras (OpenCV index):")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"  {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()


class Fuser:
    def __init__(self, ckpt_path, device="cuda", window_s=3.0):
        ck = torch.load(ckpt_path, map_location=device)
        self.names = ck["names"]
        self.target = ck.get("target", "labels")  # which camera values the model was trained on
        self.head = FusionHead(**ck["config"]).to(device).eval()
        self.head.load_state_dict(ck["state_dict"])
        self.encoder = ENCODERS[ck["encoder"]](device=device, half=True)
        self.device = device
        self.T = int(window_s * FPS)
        self.window_s = window_s
        self.lookahead = self.head.lookahead_frames
        self.mouth_cols = self.head.mouth_cols.cpu().numpy()
        for _ in range(3):  # warm up the GPU so the first live frames aren't slow
            self.encoder.encode(np.zeros(int(window_s * 16000), np.float32), self.T)

    @torch.no_grad()
    def step(self, ring, tracker, now, mode="fused", mouth_hidden=False):
        """Returns (weights[S], head_pose, output_time)."""
        times = now - (self.T - 1 - np.arange(self.T)) / FPS
        audio = ring.window(self.window_s, now)
        feats = self.encoder.encode(audio, self.T)
        video, seen, head = tracker.on_grid(times)
        out_i = self.T - 1 - self.lookahead
        self.last_seen = bool(seen[out_i])
        if mode == "camera":
            last = np.flatnonzero(seen[:out_i + 1])
            w = video[last[-1]] if len(last) else self.head.prior.cpu().numpy()
            return w, head, times[out_i]
        mouth = np.full(self.T, 0.0 if mouth_hidden else 1.0, np.float32)
        if mode == "audio":
            video[:] = 0
            seen[:] = 0
        self.last_seen = bool(seen[out_i])
        if mouth_hidden:
            video[:, self.mouth_cols] = 0
        t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None].to(self.device)  # noqa: E731
        pred = self.head(t(feats), t(video), t(seen), t(mouth * seen))[0, out_i].cpu().numpy()
        return pred, head, times[out_i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(ROOT, "models", "fusion.pt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--camera-delay", type=float, default=0.18,
                   help="seconds between a moment happening and the webcam delivering its frame "
                        "(C925e measured at 183 ms against the iPhone, Oct 28); audio is matched to it")
    p.add_argument("--mic", help="microphone name substring or index (default: system default)")
    p.add_argument("--file", help="use a recorded video instead of camera + mic")
    p.add_argument("--start", type=float, default=0.0, help="seconds into --file")
    p.add_argument("--crop", help="crop for --file, x0,y0,x1,y1 fractions")
    p.add_argument("--seconds", type=float, help="stop after this long")
    p.add_argument("--mode", default="fused", choices=["fused", "camera", "audio"])
    p.add_argument("--unreal", action="store_true", help="stream to Unreal Live Link (UDP)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--head", type=float, default=0.0,
                   help="scale for head rotation sent to Unreal (0 = off; try 1 for radians)")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="anti-jitter filter strength: 1 = default, 2 = smoother, 0.5 = lighter, 0 = off")
    p.add_argument("--view", default="3d", choices=["3d", "2d"], help="preview: 3D head or 2D cartoon")
    p.add_argument("--view-head", type=float, default=1.0, help="head rotation scale in the 3D preview")
    p.add_argument("--labels", choices=["hybrid", "mediapipe"],
                   help="camera face values: hybrid geometric labels or raw MediaPipe scores "
                   "(default: whatever the model was trained on)")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--record", help="save the output curves as an animation .json")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        return list_devices()

    if sys.platform == "win32":  # 1 ms sleep precision instead of Windows' default ~15 ms
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    fuser = Fuser(args.model)
    ring = AudioRing()
    labels = args.labels or ("hybrid" if fuser.target == "labels_hyb" else "mediapipe")
    tracker = FaceTracker(fuser.names, labels=labels)
    print(f"camera labels: {labels} (model trained on {fuser.target})")
    if args.file:
        crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
        src = FileSource(args.file, tracker, ring, args.start, crop)
        src.start()
        mic = None
    else:
        import sounddevice as sd
        dev = args.mic
        if dev is not None and not dev.isdigit():
            dev = next(i for i, d in enumerate(sd.query_devices())
                       if d["max_input_channels"] > 0 and dev.lower() in d["name"].lower())
        mic = MicSource(ring, int(dev) if dev is not None else None)
        mic.start()
        src = CameraSource(tracker, args.camera, capture_delay=args.camera_delay)
        src.start()
        print(f"Camera {args.camera} at {src.size[0]}x{src.size[1]}, mic: {mic.name}")
    sender = LiveLinkSender(fuser.names, host=args.host) if args.unreal else None
    print(f"Running ({args.mode}); output lags real time by {fuser.lookahead / FPS * 1000:.0f} ms "
          f"+ processing. {'Press q in the window to quit.' if not args.no_window else ''}")

    head3d = None
    if args.view == "3d" and not args.no_window:
        from realtime.head3d import Head3D
        head3d = Head3D(fuser.names, size=390, head_scale=args.view_head)
    mode, hidden = args.mode, False
    blinker = Blinker(fuser.names, FPS)
    hold = Hold(fuser.names, fuser.mouth_cols, FPS)
    smoother = OneEuro(fuser.names, FPS, scale=1 / args.smooth) if args.smooth > 0 else None
    frames, stats, skipped = [], [], 0
    t_start = time.perf_counter()
    next_tick = t_start + 0.5  # let a little audio/video arrive first
    while True:
        now = time.perf_counter()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now - 0.001))
            continue
        next_tick += 1 / FPS
        if next_tick < time.perf_counter() - 1 / FPS:  # over a frame behind: skip, don't pile up
            next_tick = time.perf_counter() + 1 / FPS
            skipped += 1
        tick = time.perf_counter()
        w, head, t_out = fuser.step(ring, tracker, tick, mode, hidden)
        w = hold(w, fuser.last_seen)
        w = blinker(w, fuser.last_seen)
        if smoother:
            w = smoother(w)
        done = time.perf_counter()
        stats.append((done - tick, done - t_out))
        frames.append(w)
        if sender:
            sender.send(w, [args.head * h for h in head])

        if not args.no_window:
            canvas = np.full((480, 900, 3), 30, np.uint8)
            with tracker.lock:
                latest = tracker.latest_frame
            if latest is not None:
                cam = cv2.cvtColor(latest[0], cv2.COLOR_RGB2BGR)
                h = 300
                cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))[:, :400]
                canvas[10:10 + cam.shape[0], 10:10 + cam.shape[1]] = cam
            wd = dict(zip(fuser.names, w))
            if head3d:
                head3d.update(w, head)
                canvas[0:390, 480:870] = head3d.render(bg=(30, 30, 30))
            else:
                draw_face(canvas, wd, 600, 190, 120, roll=-head[2])
            bars(canvas, wd, SHOW, 10, 320)
            c, l = np.mean([s[0] for s in stats[-30:]]), np.mean([s[1] for s in stats[-30:]])
            if tracker.rt is not None:
                cv2.putText(canvas, tracker.status, (470, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (120, 230, 120) if tracker.locked else (80, 200, 255), 1)
            cv2.putText(canvas, f"mode: {mode}{' (mouth hidden)' if hidden else ''}", (470, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, f"compute {c * 1000:.0f} ms | latency {l * 1000:.0f} ms | "
                        f"tracking {tracker.track_ms:.0f} ms", (470, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(canvas, "1 fused  2 camera  3 audio  n calibrate  m hide mouth  q quit", (470, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow("VR face - live", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in MODES:
                mode = MODES[key]
            if key == ord("m"):
                hidden = not hidden
            if key == ord("n"):
                print("calibrated" if tracker.calibrate() else "calibration needs a visible face")

        if args.seconds and time.perf_counter() - t_start > args.seconds:
            break
        if args.file and not src.running:
            break

    src.stop()
    src.join(timeout=2)
    if mic:
        mic.stop()
    cv2.destroyAllWindows()
    st = np.array(stats[30:]) if len(stats) > 60 else np.array(stats)
    print(f"{len(frames)} frames in {time.perf_counter() - t_start:.1f}s | compute per frame: "
          f"median {np.median(st[:, 0]) * 1000:.0f} ms, p95 {np.percentile(st[:, 0], 95) * 1000:.0f} ms | "
          f"latency (sound/camera to output): median {np.median(st[:, 1]) * 1000:.0f} ms | "
          f"{skipped} slow ticks skipped")
    if args.record:
        save_animation(args.record, frames, fuser.names)
        print(f"Saved {args.record}")


if __name__ == "__main__":
    main()
