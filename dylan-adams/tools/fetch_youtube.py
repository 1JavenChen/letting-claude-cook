"""Download just the end of your own YouTube videos into data/raw/.

    python tools/fetch_youtube.py URL [URL ...] [--last 15]

Uses yt-dlp's time-range download, so only the last N minutes are fetched
(e.g. the fullscreen-webcam section at the end of a stream). --last 0 fetches
the whole video.
"""

import argparse
import os
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urls", nargs="+")
    p.add_argument("--last", type=float, default=15, help="minutes from the end to download (0 = whole video)")
    p.add_argument("--max-height", type=int, default=1080)
    args = p.parse_args()

    name = "%(title)s [%(id)s]" + (" last%(section_start)d" if args.last > 0 else "")
    out = os.path.join(ROOT, "data", "raw", name + ".%(ext)s")
    mins = int(args.last)
    secs = int(round((args.last - mins) * 60))
    for url in args.urls:
        cmd = [os.path.join(os.path.dirname(sys.executable), "yt-dlp"),
               "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
               "-f", f"bv*[height<={args.max_height}]+ba/b[height<={args.max_height}]",
               "--merge-output-format", "mp4",
               "--restrict-filenames", "-o", out, url]
        if args.last > 0:
            cmd[-3:-3] = ["--download-sections", f"*-{mins}:{secs:02d}-inf"]
        print(f"Fetching {f'last {args.last:g} min of ' if args.last > 0 else ''}{url}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
