"""Record training footage: webcam video + microphone audio, in sync, to data/raw/.

    python tools/record_webcam.py --minutes 20
    python tools/record_webcam.py --list            # device names

One ffmpeg process captures both devices, so audio and video share a clock
(no drift - unlike edited YouTube videos). Video is stored at 1080p30 H.264;
sit so your face fills roughly the middle third of the frame. Press q in this
window to stop early.

While recording, talk naturally: read docs/recording_prompts.md, tell stories,
react - smiles, frowns, raised brows, silences with your mouth closed. Variety
matters more than length.
"""

import argparse
import datetime
import os
import subprocess

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-device", default="Logitech Webcam C925e")
    p.add_argument("--audio-device", default="Analogue 1 + 2 (2- Focusrite USB Audio)")
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--name", help="output name (default: webcam_<date>_<time>)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        subprocess.run([FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        return

    name = args.name or "webcam_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(ROOT, "data", "raw", name + ".mp4")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-stats",
           "-f", "dshow", "-rtbufsize", "512M", "-vcodec", "mjpeg", "-video_size", args.size,
           "-framerate", "30", "-audio_buffer_size", "20",
           "-i", f"video={args.video_device}:audio={args.audio_device}",
           "-t", str(args.minutes * 60),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ac", "1", out]
    print(f"Recording {args.minutes:g} min to {out} (press q to stop)")
    subprocess.run(cmd, check=True)
    print(f"Saved {out}. Next: python tools/prepare_data.py --only {name}.mp4 --max-width 1920")


if __name__ == "__main__":
    main()
