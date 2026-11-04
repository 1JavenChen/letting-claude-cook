"""Run the Phase 3 data pipeline over a public dataset, several videos at once.

    python tools/process_public.py hdtf --jobs 8 --max-width 960
    python tools/process_public.py ravdess --jobs 8

For every video in data/public/<dataset>/videos/ (HDTF: raw/) this runs, each
step skipped when already done:
    prepare_data.py   -> data/processed/<video name>.npz   (names start with pub_<dataset>_)
    [fix_drift.py --mode constant]  with --fix-drift: one A/V offset per clip, moved to the natural -0.05 s lead
    sync_segments.py  -> keep only stretches where the mouth follows the speech
    degrade_track.py  -> labels_lo (320 px "weak webcam") for fusion training
Features (tools/extract_features.py) are a separate GPU step afterwards.
Progress goes to out/logs/datasets.log.
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_datasets import PUBLIC, log  # noqa: E402

PY = sys.executable
TOOLS = os.path.join(ROOT, "tools")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")


def run(cmd):
    r = subprocess.run([PY, *cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {r.stderr.strip().splitlines()[-1:]}")
    return lines[-1] if lines else ""


def one(video, out_dir, args):
    stem = os.path.splitext(os.path.basename(video))[0]
    npz = os.path.join(out_dir, stem + ".npz")
    t0 = time.time()
    msgs = []
    if not os.path.exists(npz):
        msgs.append(run([os.path.join(TOOLS, "prepare_data.py"), "--raw", os.path.dirname(video),
                         "--out", out_dir, "--only", os.path.basename(video), "--max-width", str(args.max_width)]))
        if not os.path.exists(npz):
            return stem, "no face found - skipped"
    keys = set(np.load(npz).files)
    if args.fix_drift and "drift_src" not in keys:
        msgs.append(run([os.path.join(TOOLS, "fix_drift.py"), npz, "--mode", "constant", "--min-corr", "0.1"]))
        keys = set(np.load(npz).files)
        keys.discard("keep_raw")
    if "keep_raw" not in keys:
        msgs.append(run([os.path.join(TOOLS, "sync_segments.py"), npz, "--min-mouth", str(args.min_mouth)]))
    if "labels_lo" not in keys:
        msgs.append(run([os.path.join(TOOLS, "degrade_track.py"), npz, video, "--width", str(args.lo_width)]))
    return stem, f"{time.time() - t0:.0f} s | " + " | ".join(m.split(": ", 1)[-1] for m in msgs if m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--lo-width", type=int, default=320)
    p.add_argument("--min-mouth", type=float, default=60, help="sync_segments.py --min-mouth (px)")
    p.add_argument("--fix-drift", action="store_true")
    p.add_argument("--only", help="comma-separated substrings of video names")
    p.add_argument("--out", default=os.path.join(ROOT, "data", "processed"))
    args = p.parse_args()

    vdir = os.path.join(PUBLIC, args.dataset, "videos")
    if not os.path.isdir(vdir):
        vdir = os.path.join(PUBLIC, args.dataset, "raw")
    videos = sorted(os.path.join(vdir, f) for f in os.listdir(vdir)
                    if f.endswith(VIDEO_EXTS) and f.startswith("pub_") and ".part" not in f
                    and time.time() - os.path.getmtime(os.path.join(vdir, f)) > 120)  # still downloading?
    if args.only:
        videos = [v for v in videos if any(s in os.path.basename(v) for s in args.only.split(","))]
    log(f"process {args.dataset}: {len(videos)} videos, {args.jobs} at once, max-width {args.max_width}")
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(one, v, args.out, args): v for v in videos}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            try:
                stem, msg = fu.result()
            except Exception as e:  # noqa: BLE001
                stem, msg = os.path.basename(futs[fu]), f"ERROR {e}"
            log(f"  [{k}/{len(videos)}] {stem}: {msg}")


if __name__ == "__main__":
    main()
