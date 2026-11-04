"""Live inputs: microphone audio, webcam frames, and face tracking.

Everything is timestamped with time.perf_counter() so audio and video can be
lined up on one 30 fps grid. A video file can stand in for camera + mic
(FileSource), played back at real-time speed, to test the live pipeline on
recorded footage.
"""

import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
SR = 16000


class AudioRing:
    """The last few seconds of 16 kHz mono audio, with the time of the newest sample."""

    def __init__(self, seconds=4.0):
        self.buf = np.zeros(int(seconds * SR), np.float32)
        self.t_end = time.perf_counter()
        self.lock = threading.Lock()

    def push(self, samples, t_end):
        n = len(samples)
        with self.lock:
            if n >= len(self.buf):
                self.buf[:] = samples[-len(self.buf):]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = samples
            self.t_end = t_end

    def window(self, seconds, t_end):
        """Audio for (t_end - seconds, t_end]; zeros for anything not yet heard."""
        n = int(seconds * SR)
        with self.lock:
            lag = int(round((self.t_end - t_end) * SR))  # samples newer than t_end
            stop = len(self.buf) - max(lag, 0)
            out = self.buf[max(0, stop - n):stop].copy()
        pad_after = max(-lag, 0)  # t_end is in the future of what we have
        out = np.concatenate([out, np.zeros(pad_after, np.float32)])[-n:] if pad_after else out
        return np.pad(out, (n - len(out), 0))


class MicSource:
    def __init__(self, ring, device=None):
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        self.rate = int(info["default_samplerate"])
        self.ring = ring
        self.name = info["name"]
        self.stream = sd.InputStream(device=device, channels=1, samplerate=self.rate,
                                     blocksize=self.rate // 100, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        t = time.perf_counter()
        x = indata[:, 0].astype(np.float32)
        if self.rate != SR:
            g = np.gcd(SR, self.rate)
            x = resample_poly(x, SR // g, self.rate // g).astype(np.float32)
        self.ring.push(x, t)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()


class FaceTracker:
    """Runs MediaPipe on frames handed to it; keeps a short history of results.

    Each result: (capture_time, blendshapes[S] or None, head (yaw, pitch, roll) radians).

    With labels="hybrid" (default) the blendshapes are the hybrid labels the models are
    trained on (realtime/retarget.py): geometric weights from the 3D landmarks for eyes,
    smiles/grimaces and brow lowering, MediaPipe scores for lip protrusion, jaw and the rest,
    minus the person's resting face. The resting face is estimated automatically from the
    most relaxed recent frames until calibrate() locks one in (saved to CALIB).
    labels="mediapipe" gives MediaPipe's raw scores.
    """

    CALIB = os.path.join(ROOT, "models", "calib_live.npz")

    def __init__(self, names, history_s=5.0, labels="hybrid"):
        self.names = names
        self.labels = labels
        self.rt = None
        if labels == "hybrid":
            from realtime.retarget import LandmarkRetargeter
            self.rt = LandmarkRetargeter(names)
            self.offset = None
            self.locked = False
            self.buf = deque(maxlen=600)  # last ~20 s of (landmarks, scores) for the resting face
            self.prev_w = None
            self.calib_lock = threading.Lock()
            self.status = "estimating your resting face - relax, or press N"
            if os.path.exists(self.CALIB):
                try:
                    self.rt.load(self.CALIB)
                    c = np.load(self.CALIB)
                    self.offset = c["hyb_offset"]
                    self.offset_upper = c["hyb_offset_upper"] if "hyb_offset_upper" in c.files else self.offset
                    self.locked = True
                    self.status = "calibrated (press N to redo)"
                except Exception:  # noqa: BLE001  (e.g. calibration from an older version)
                    pass
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.results = deque(maxlen=int(history_s * 60))
        self.latest_frame = None
        self.lock = threading.Lock()
        self.last_ms = -1
        self.track_ms = 0.0

    FROZEN_S = 0.3  # a real camera's image is never exactly the same (sensor noise)

    def _frozen(self, rgb, t_capture):
        """True once the camera has delivered the identical picture for FROZEN_S: a stuck driver or
        bumped cable can keep resending the last frame with fresh timestamps."""
        small = cv2.resize(rgb, (32, 18), interpolation=cv2.INTER_AREA).astype(np.int16)
        prev = getattr(self, "_prev_small", None)
        self._prev_small = small
        if prev is None or np.abs(small - prev).mean() > 0.2:
            self._same_since = t_capture
            return False
        return t_capture - self._same_since > self.FROZEN_S

    def process(self, rgb, t_capture):
        t0 = time.perf_counter()
        if self._frozen(rgb, t_capture):
            with self.lock:
                self.results.append((t_capture, None, (0.0, 0.0, 0.0)))  # counts as "face not seen"
                self.latest_frame = (rgb, None)
            return
        ms = max(int(t_capture * 1000), self.last_ms + 1)
        self.last_ms = ms
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms)
        blend, head = None, (0.0, 0.0, 0.0)
        if res.face_blendshapes:
            d = {c.category_name: c.score for c in res.face_blendshapes[0]}
            blend = np.array([d.get(n, 0.0) for n in self.names], np.float32)
            if self.rt is not None:
                h, w = rgb.shape[:2]
                lmk = np.array([[q.x * w, -q.y * h, -q.z * w] for q in res.face_landmarks[0][:468]])
                with self.calib_lock:
                    blend = self._hybrid(lmk, blend)
            R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
            head = (float(np.arctan2(R[0, 2], R[2, 2])), float(np.arcsin(-np.clip(R[1, 2], -1, 1))),
                    float(np.arctan2(R[1, 0], R[1, 1])))
        with self.lock:
            self.results.append((t_capture, blend, head))
            self.latest_frame = (rgb, res)
        self.track_ms = 0.9 * self.track_ms + 0.1 * (time.perf_counter() - t0) * 1000

    def _hybrid(self, lmk, scores):
        from realtime.retarget import hybrid
        self.buf.append((lmk, scores))
        if not self.locked and len(self.buf) >= 60 and (self.offset is None or len(self.buf) % 150 == 0):
            self._set_rest(self._relaxed())
        if self.offset is None:
            return scores
        geo = self.rt(lmk, scores, w0=self.prev_w, iters=60)
        self.prev_w = geo[None, self.rt.free]
        mouth_geo = float(np.hypot(*(lmk[61, :2] - lmk[291, :2]))) >= 60  # small face: trust lips less
        offset = self.offset if mouth_geo else getattr(self, "offset_upper", self.offset)
        return hybrid(scores, geo, self.names, offset, mouth_geo=mouth_geo)

    def _relaxed(self, n=45):
        """Indices (into buf) of the most relaxed recent frames, by MediaPipe's expression scores."""
        cols = [self.names.index(x) for x in ("jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
                                              "mouthFunnel", "browInnerUp", "eyeBlinkLeft", "eyeBlinkRight")]
        energy = np.array([s[cols].sum() for _, s in self.buf])
        return np.argsort(energy)[:n]

    def _set_rest(self, idx):
        from realtime.retarget import resting_offset
        L = np.stack([self.buf[i][0] for i in idx])
        S = np.stack([self.buf[i][1] for i in idx])
        self.rt.set_neutral(L)
        geo = self.rt(L, S)
        self.offset = resting_offset(S, geo, self.names, np.arange(len(idx)))
        self.offset_upper = resting_offset(S, geo, self.names, np.arange(len(idx)), mouth_geo=False)
        self.prev_w = None

    def calibrate(self, seconds=2.0):
        """Use the last `seconds` of frames as the resting face (hold still, relaxed) and save it."""
        if self.rt is None or len(self.buf) < 10:
            return False
        with self.calib_lock:
            return self._calibrate(seconds)

    def _calibrate(self, seconds):
        n = min(len(self.buf), int(seconds * 30))
        self._set_rest(np.arange(len(self.buf) - n, len(self.buf)))
        self.locked = True
        self.rt.save(self.CALIB)
        d = dict(np.load(self.CALIB))
        d["hyb_offset"] = self.offset
        d["hyb_offset_upper"] = self.offset_upper
        np.savez(self.CALIB, **d)
        self.status = "calibrated (press N to redo)"
        return True

    def on_grid(self, times):
        """Tracker results resampled onto `times` -> (video[T,S], seen[T], head at the end).

        Interpolates between the two results around each time (the camera and
        the 30 Hz output clock aren't in step; holding the last result instead
        repeats frames and makes the face stutter). Times after the newest
        result hold it; gaps over 0.2 s count as "face not seen"."""
        with self.lock:
            res = [r for r in self.results if r[1] is not None]
            any_res = bool(self.results)
        S = len(self.names)
        video = np.zeros((len(times), S), np.float32)
        seen = np.zeros(len(times), np.float32)
        if not res:
            return video, seen, (0.0, 0.0, 0.0)
        rt = np.array([r[0] for r in res])
        vals = np.stack([r[1] for r in res])
        heads = np.array([r[2] for r in res])
        hi = np.clip(np.searchsorted(rt, times), 1, max(1, len(rt) - 1))
        lo = hi - 1
        if len(rt) == 1:
            lo = hi = np.zeros(len(times), int)
        span = np.maximum(rt[hi] - rt[lo], 1e-6)
        f = np.clip((times - rt[lo]) / span, 0, 1)[:, None]
        video[:] = vals[lo] * (1 - f) + vals[hi] * f
        nearest = np.minimum(np.abs(times - rt[lo]), np.abs(times - rt[hi]))
        gap = np.where((times > rt[lo]) & (times < rt[hi]), span, nearest)
        seen[:] = (gap < 0.2) & (times > rt[0] - 0.05) & (times < rt[-1] + 0.2)
        video[seen == 0] = 0
        h = np.clip(np.searchsorted(rt, times[-1]) - 1, 0, len(rt) - 1)
        return video, seen, tuple(heads[h]) if any_res else (0.0, 0.0, 0.0)


class CameraSource(threading.Thread):
    def __init__(self, tracker, index=0, width=1280, height=720, fps=30, capture_delay=0.05):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.capture_delay = capture_delay  # sensor-to-app latency estimate
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.running = True

    def run(self):
        while self.running:
            ok, bgr = self.cap.read()
            t = time.perf_counter() - self.capture_delay
            if ok:
                self.tracker.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)

    def stop(self):
        self.running = False
        self.cap.release()


class FileSource(threading.Thread):
    """Plays a video file as if it were a live camera + mic (for testing)."""

    def __init__(self, path, tracker, ring, start=0.0, crop=None, max_width=1280):
        super().__init__(daemon=True)
        import av
        from tools.prepare_data import extract_audio
        self.audio = extract_audio(path, start)
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.start_s = start
        self.crop = crop
        self.max_width = max_width
        self.tracker = tracker
        self.ring = ring
        self.running = True
        self.t0 = None

    def run(self):
        tb = self.stream.time_base
        if self.start_s:
            self.container.seek(int(max(0, self.start_s - 2) / tb), stream=self.stream)
        self.t0 = time.perf_counter()
        fed = 0
        for frame in self.container.decode(self.stream):
            if not self.running:
                break
            t_vid = float(frame.pts * tb) - self.start_s
            if t_vid < 0:
                continue
            # release the audio heard up to now
            now = time.perf_counter() - self.t0
            upto = min(len(self.audio), int(now * SR))
            if upto > fed:
                self.ring.push(self.audio[fed:upto], self.t0 + upto / SR)
                fed = upto
            wait = t_vid - (time.perf_counter() - self.t0)
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.1:
                continue  # running behind: drop the frame like a live camera would
            rgb = frame.to_ndarray(format="rgb24")
            if self.crop:
                h, w = rgb.shape[:2]
                c = self.crop
                rgb = rgb[int(c[1] * h):int(c[3] * h), int(c[0] * w):int(c[2] * w)]
            h, w = rgb.shape[:2]
            if w > self.max_width:
                rgb = cv2.resize(rgb, (self.max_width, int(h * self.max_width / w)))
            self.tracker.process(rgb, self.t0 + t_vid)
        self.running = False

    def stop(self):
        self.running = False
