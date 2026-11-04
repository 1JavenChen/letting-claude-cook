"""Run tools/retarget_labels.py over the public clips, several at once.

    .venv\\Scripts\\python tools/retarget_public.py ravdess cremad hdtf --jobs 5

Skips clips that already have labels_geo_lo or have nothing kept. Videos are the
ones tools/process_public.py processed (data/public/<dataset>/videos, HDTF: raw).
Log: out/logs/retarget_public.log.
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "out", "logs", "retarget_public.log")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def todo(dataset):
    vdir = os.path.join(ROOT, "data", "public", dataset, "raw" if dataset == "hdtf" else "videos")
    out = []
    for f in sorted(os.listdir(vdir)):
        stem, ext = os.path.splitext(f)
        npz = os.path.join(ROOT, "data", "processed", stem + ".npz")
        if ext.lower() not in (".mp4", ".mkv", ".webm") or not os.path.exists(npz):
            continue
        with np.load(npz) as d:
            if "labels_geo_lo" in d.files or not d["keep"].any():
                continue
        out.append((npz, os.path.join(vdir, f)))
    return out


def one(npz, video):
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "retarget_labels.py"), npz, video,
                        "--max-width", "640", "--lo-width", "320"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    if r.returncode != 0:
        return f"ERROR {r.stderr.strip().splitlines()[-1:]}"
    first = next((l for l in r.stdout.splitlines() if "neutral from" in l), "")
    return f"{time.time() - t:.0f} s | {first.split(': ', 1)[-1]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--jobs", type=int, default=5)
    args = p.parse_args()
    work = [w for ds in args.datasets for w in todo(ds)]
    log(f"retarget {args.datasets}: {len(work)} clips, {args.jobs} at once")
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(one, *w): w for w in work}
        for k, f in enumerate(cf.as_completed(futs), 1):
            log(f"  [{k}/{len(work)}] {os.path.basename(futs[f][0])}: {f.result()}")
    log("done")


if __name__ == "__main__":
    main()
