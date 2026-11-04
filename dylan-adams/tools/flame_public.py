"""SMIRK + FLAME->ARKit retargeting over the public clips, several at once (run with .venv-smirk).

    .venv-smirk\\Scripts\\python tools/flame_public.py --jobs 4 [--only pub_ravdess] [--all]

For every data/processed/pub_*.npz (by default only those with kept frames) whose video
is found under data/public/<dataset>/{videos,raw}/: smirk_track.py --match (-> <name>_flame.npz)
then flame_to_arkit.py (adds labels_flame / face_ok_flame to the clip). Resumable: clips
that already have labels_flame are skipped. Log: out/logs/flame_public.log.
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
PUBLIC = os.path.join(ROOT, "data", "public")
LOG = os.path.join(ROOT, "out", "logs", "flame_public.log")
PY = sys.executable


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_video(stem):
    ds = stem.split("_")[1]
    for sub in ("videos", "raw"):
        for ext in (".mp4", ".mkv", ".webm", ".mov"):
            p = os.path.join(PUBLIC, ds, sub, stem + ext)
            if os.path.exists(p):
                return p
    return None


def run(cmd):
    r = subprocess.run([PY, *cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    if r.returncode != 0:
        tail = [l for l in r.stderr.splitlines() if l.strip()][-1:]
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {tail}")
    return [l for l in r.stdout.splitlines() if l.strip()]


def one(stem, video):
    t0 = time.time()
    npz = os.path.join(PROC, stem + ".npz")
    fnpz = os.path.join(PROC, stem + "_flame.npz")
    msg = ""
    if not os.path.exists(fnpz):
        out = run([os.path.join(ROOT, "tools", "smirk_track.py"), video, "--match", npz, "--out", fnpz])
        msg = out[-2] if len(out) >= 2 else ""
    out = run([os.path.join(ROOT, "tools", "flame_to_arkit.py"), fnpz, "--match", npz])
    return f"{stem}: {time.time() - t0:.0f} s | {msg.split(' -> ')[0]} | {out[-1].split(': ', 1)[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", help="comma-separated substrings")
    ap.add_argument("--all", action="store_true", help="also clips with no kept frames")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    todo, skipped = [], 0
    for f in sorted(os.listdir(PROC)):
        if not (f.startswith("pub_") and f.endswith(".npz")) or f.endswith("_flame.npz"):
            continue
        stem = f[:-4]
        if a.only and not any(s in stem for s in a.only.split(",")):
            continue
        with np.load(os.path.join(PROC, f)) as d:
            if "labels_flame" in d.files or (not a.all and not d["keep"].any()):
                skipped += 1
                continue
        video = find_video(stem)
        if video is None:
            log(f"{stem}: video not found - skipped")
            continue
        todo.append((stem, video))
    log(f"start: {len(todo)} clips to do, {skipped} skipped (done or nothing kept), {a.jobs} jobs")
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(one, s, v): s for s, v in todo}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            try:
                log(f"[{k}/{len(todo)}] {fu.result()}")
            except Exception as e:  # noqa: BLE001
                log(f"[{k}/{len(todo)}] {futs[fu]}: ERROR {e}")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
