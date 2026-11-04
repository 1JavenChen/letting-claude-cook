"""Turn talking-head videos into (audio, blendshape label) training pairs.

For every video in data/raw/:
  1. extract the audio track (16 kHz mono),
  2. run MediaPipe's FaceLandmarker on every frame to get 52 ARKit blendshapes
     for the largest face in view (assumed to be you),
  3. resample the labels onto a fixed 30 fps grid,
  4. build a "keep" mask that drops frames where no face was found, or where
     there's speech but your mouth is still (probably someone else talking),
  5. save data/processed/<name>.npz plus a <name>.png diagnostic plot.

Usage:
    python tools/prepare_data.py              # process every new video
    python tools/prepare_data.py --force      # reprocess everything
"""

import argparse
import os
import subprocess
import sys

import av
import cv2
import imageio_ffmpeg
import matplotlib
import mediapipe as mp
import numpy as np
import soundfile as sf
from mediapipe.tasks.python import BaseOptions, vision
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS, MOUTH_SHAPES  # noqa: E402

SR = 16000
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
MAX_WIDTH = 640  # default downscale for speed; raise it (--max-width) for zoomed-out 1080p shots


def extract_audio(video_path, start=0.0, duration=None):
    span = ["-ss", str(start)] + (["-t", str(duration)] if duration else [])
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", *span, "-i", video_path,
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _face_area(landmarks):
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def track_face(video_path, crop=None, start=0.0, duration=None, max_width=MAX_WIDTH, landmarks=False):
    """Return (times_s, blendshapes[N, 52] with NaN where no face, names, mouth_px[N]
    [, landmarks[N, 468, 3] float32 if landmarks=True: right-handed pixel units, NaN = no face]).

    crop: optional (x0, y0, x1, y1) as fractions of the frame, e.g. the webcam
    box of a stream layout. Applied before downscaling, so small faces survive.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=3,
        output_face_blendshapes=True,
    )
    # PyAV gives each frame's exact presentation timestamp from the file.
    # (OpenCV's position estimate drifted by seconds over a 3-hour video.)
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = stream.time_base
    end = start + duration if duration else None
    span = duration or (float(stream.duration * tb) - start if stream.duration else None)
    total = int(span * float(stream.average_rate)) if span and stream.average_rate else None
    if start:
        container.seek(int(max(0, start - 2) / tb), stream=stream)  # lands on an earlier keyframe
    times, rows, names, mouths, lmks = [], [], None, [], []
    last_ms = -1
    progress = tqdm(total=total, desc="  face tracking", unit="frame", leave=False)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for frame in container.decode(stream):
            t = float(frame.pts * tb)
            if t < start:
                continue
            if end is not None and t >= end:
                break
            ms = max(int(round((t - start) * 1000)), last_ms + 1)  # MediaPipe needs increasing ms
            last_ms = ms
            rgb = frame.to_ndarray(format="rgb24")
            if crop:
                h, w = rgb.shape[:2]
                rgb = rgb[int(crop[1] * h): int(crop[3] * h), int(crop[0] * w): int(crop[2] * w)]
            h0, w0 = h, w = rgb.shape[:2]  # mouth size is measured at this (source) resolution
            if w > max_width:
                rgb = cv2.resize(rgb, (max_width, int(h * max_width / w)))
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms)
            progress.update()
            times.append(ms / 1000.0)
            if result.face_blendshapes:
                i = max(range(len(result.face_landmarks)),
                        key=lambda k: _face_area(result.face_landmarks[k]))
                cats = result.face_blendshapes[i]
                pts = result.face_landmarks[i]
                mouths.append(float(np.hypot((pts[61].x - pts[291].x) * w0, (pts[61].y - pts[291].y) * h0)))
                if landmarks:  # pixel-space 3D (see realtime/retarget.to_metric), of the source crop
                    lmks.append(np.array([[q.x * w0, -q.y * h0, -q.z * w0] for q in pts[:468]], np.float32))
                if names is None:
                    names = [c.category_name for c in cats]
                rows.append([c.score for c in cats])
            else:
                rows.append(None)
                mouths.append(np.nan)
                if landmarks:
                    lmks.append(np.full((468, 3), np.nan, np.float32))
    progress.close()
    container.close()
    if names is None:
        return (np.array(times), None, None, None) + ((None,) if landmarks else ())
    blend = np.array([r if r is not None else [np.nan] * len(names) for r in rows], np.float32)
    out = (np.array(times), blend, names, np.array(mouths, np.float32))
    return out + (np.array(lmks, np.float32),) if landmarks else out


def resample_labels(times, blend, n_out):
    """Interpolate per-video-frame labels onto the fixed FPS grid."""
    grid = np.arange(n_out) / FPS
    valid = ~np.isnan(blend[:, 0])
    out = np.zeros((n_out, blend.shape[1]), np.float32)
    if valid.sum() < 2:
        return out, np.zeros(n_out, bool)
    for j in range(blend.shape[1]):
        out[:, j] = np.interp(grid, times[valid], blend[valid, j])
    # A grid point is only trustworthy if a face was found close to it.
    vt = times[valid]
    idx = np.clip(np.searchsorted(vt, grid), 1, len(vt) - 1)
    nearest = np.minimum(np.abs(vt[idx] - grid), np.abs(vt[idx - 1] - grid))
    return out, nearest < 0.1


def audio_db(audio, n_out):
    hop = SR // FPS
    pad = np.pad(audio, (0, max(0, n_out * hop - len(audio))))[: n_out * hop]
    rms = np.sqrt(np.mean(pad.reshape(n_out, hop) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-6)


def rolling_mean(x, width):
    return np.convolve(x, np.ones(width) / width, mode="same")


def build_mask(db, mouth, face_ok):
    """Keep frames with a face, except speech-with-a-still-mouth stretches."""
    # "Speech" = within 20 dB of the loudest parts of the video. Relative to the
    # top end because we can't assume any share of the video is silent. Loud
    # music also counts, which is fine: music + still mouth is dropped too.
    speech_floor = np.percentile(db, 95) - 20
    speech = rolling_mean((db > speech_floor).astype(float), FPS // 2) > 0.5

    motion = np.r_[0, np.abs(np.diff(mouth, axis=0)).sum(axis=1)]
    motion = rolling_mean(motion, FPS)  # average over ~1 second
    ref = np.percentile(motion[face_ok], 75) if face_ok.any() else 0
    still = motion < 0.25 * ref

    someone_else = speech & still
    keep = face_ok & ~someone_else
    return keep, speech, motion, ref, someone_else


def plot_diagnostics(path, db, jaw, keep, someone_else, face_ok):
    t = np.arange(len(db)) / FPS
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(t, db, lw=0.6, color="0.3")
    axes[0].set_ylabel("audio dB")
    axes[1].plot(t, jaw, lw=0.6, color="tab:red", label="jawOpen (MediaPipe)")
    axes[1].set_ylabel("jawOpen")
    axes[1].set_xlabel("seconds")
    for ax in axes:
        ax.fill_between(t, 0, 1, where=~face_ok, transform=ax.get_xaxis_transform(),
                        color="tab:gray", alpha=0.3, lw=0, label="no face")
        ax.fill_between(t, 0, 1, where=someone_else, transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.3, lw=0, label="speech, mouth still (dropped)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{os.path.basename(path)}  -  kept {keep.mean():.0%} of frames")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def process(video_path, out_dir, crop=None, tag=None, start=0.0, duration=None,
            max_width=MAX_WIDTH):
    name = os.path.splitext(os.path.basename(video_path))[0] + (f"_{tag}" if tag else "")
    audio = extract_audio(video_path, start, duration)
    times, blend, names, mouth_px = track_face(video_path, crop, start, duration, max_width)
    if blend is None:
        print("  no face found in any frame - skipped")
        return None

    n = min(int(len(audio) / SR * FPS), int(times[-1] * FPS))
    labels, face_ok = resample_labels(times, blend, n)
    mouth_px = resample_labels(times, mouth_px[:, None], n)[0][:, 0]
    audio = audio[: n * (SR // FPS)]
    db = audio_db(audio, n)
    mouth = labels[:, [names.index(s) for s in MOUTH_SHAPES]]
    keep, _, _, _, someone_else = build_mask(db, mouth, face_ok)

    np.savez_compressed(os.path.join(out_dir, name + ".npz"), audio=audio, sr=SR, fps=FPS,
                        labels=labels, names=np.array(names), keep=keep, face_ok=face_ok,
                        mouth_px=mouth_px)
    plot_diagnostics(os.path.join(out_dir, name + ".png"), db, labels[:, names.index("jawOpen")],
                     keep, someone_else, face_ok)
    return n / FPS, keep.mean(), face_ok.mean(), someone_else.mean()


def save_preview(video_path, crop, out_path):
    """Grab a frame ~60 s in, apply the crop, save it so you can check framing."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 60_000)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok, frame = cap.read()
    cap.release()
    if crop:
        h, w = frame.shape[:2]
        frame = frame[int(crop[1] * h): int(crop[3] * h), int(crop[0] * w): int(crop[2] * w)]
    cv2.imwrite(out_path, frame)
    print(f"Preview: {out_path} ({frame.shape[1]}x{frame.shape[0]} px)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    p.add_argument("--out", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--crop", help="webcam region as fractions x0,y0,x1,y1 "
                   "(e.g. 0.7,0.65,1,1 for a bottom-right facecam)")
    p.add_argument("--tag", help="suffix for output names, so one video can be processed "
                   "once per person, e.g. --tag me / --tag friend with different crops")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the video to begin")
    p.add_argument("--duration", type=float, help="seconds to process (default: to the end)")
    p.add_argument("--max-width", type=int, default=MAX_WIDTH,
                   help="downscale frames to this width before tracking (bigger = finer mouths, slower)")
    p.add_argument("--only", help="process just this one video file name from the raw folder")
    p.add_argument("--preview", action="store_true",
                   help="save a cropped frame (~1 min in) of each video as <name>_crop.jpg and stop")
    args = p.parse_args()
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
    os.makedirs(args.out, exist_ok=True)

    videos = sorted(f for f in os.listdir(args.raw) if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
    if args.only:
        videos = [f for f in videos if f == args.only]
    if not videos:
        print(f"No videos found in {args.raw}")
        return
    total_s = kept_s = 0.0
    for f in videos:
        stem = os.path.splitext(f)[0] + (f"_{args.tag}" if args.tag else "")
        if not args.force and os.path.exists(os.path.join(args.out, stem + ".npz")):
            print(f"{f}: already processed (use --force to redo)")
            continue
        if args.preview:
            save_preview(os.path.join(args.raw, f), crop, os.path.join(args.out, stem + "_crop.jpg"))
            continue
        print(f"{f}:")
        stats = process(os.path.join(args.raw, f), args.out, crop, args.tag,
                        args.start, args.duration, args.max_width)
        if stats:
            dur, kept, face, other = stats
            total_s += dur
            kept_s += dur * kept
            print(f"  {dur / 60:.1f} min | face found {face:.0%} | "
                  f"dropped as other speaker {other:.0%} | kept {kept:.0%}")
    if total_s:
        print(f"\nNew usable training audio: {kept_s / 60:.1f} of {total_s / 60:.1f} min. "
              f"Check the .png plots in {args.out} to sanity-check the filtering.")


if __name__ == "__main__":
    main()
