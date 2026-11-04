"""Download public research talking-face datasets for pretraining.

    python tools/fetch_datasets.py hdtf    [--jobs 4] [--limit N]
    python tools/fetch_datasets.py mead    [--actors 24] [--levels 3]
    python tools/fetch_datasets.py ravdess [--actors 24]
    python tools/fetch_datasets.py cremad  [--actors 40]
    python tools/fetch_datasets.py concat  mead|ravdess|cremad [--jobs 12]

Everything lands in data/public/<dataset>/raw/ (downloads only; nothing in
there is ever executed - archives are read member by member and only media
files with expected names are written, under names we build ourselves).
Short-clip datasets (MEAD, RAVDESS, CREMA-D: 2-5 s per clip) are joined per
speaker (and emotion, for MEAD) by `concat` into
data/public/<dataset>/videos/pub_<dataset>_<group>.mp4 plus a .json listing
where each original clip starts, because training samples 4 s windows. HDTF
clips are long already; its raw/ folder is used directly.

Every step is resumable: finished files are skipped, dead YouTube links are
remembered in raw/failed.txt (use --retry-failed to try them again).
Licenses and citations: docs/datasets.md. Research / non-commercial use only.
"""

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "data", "public")
LOG = os.path.join(ROOT, "out", "logs", "datasets.log")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
YTDLP = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
UA = {"User-Agent": "vr-lipsync-fetch/1.0 (class research project)"}

HDTF_BASE = "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/"
MEAD_FOLDER = "1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j"  # Part0 (Google Drive), from the MEAD project page
RAVDESS_RECORD = "https://zenodo.org/api/records/1188976"
CREMAD_MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/"


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def raw_dir(ds):
    d = os.path.join(PUBLIC, ds, "raw")
    os.makedirs(d, exist_ok=True)
    return d


def fetch_text(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8-sig", errors="replace")


def download(url, dest, retries=4):
    """Resumable download to dest (via dest.part + HTTP Range)."""
    if os.path.exists(dest):
        return dest
    part = dest + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, dest)
            return dest
        except Exception as e:  # noqa: BLE001 - network errors of all kinds: retry
            log(f"  download error ({attempt + 1}/{retries}) {os.path.basename(dest)}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def safe(s):
    return re.sub(r"[^A-Za-z0-9_-]", "", s)


# ---------------------------------------------------------------- HDTF

def hseconds(t):
    s = 0
    for v in t.split(":"):
        s = s * 60 + int(v)
    return s


def hdtf_jobs():
    jobs = []
    for subset in ("RD", "WDA", "WRA"):
        urls = {}
        for line in fetch_text(HDTF_BASE + f"{subset}_video_url.txt").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                urls[parts[0].replace(".mp4", "")] = parts[1].split("&")[0]
        for line in fetch_text(HDTF_BASE + f"{subset}_annotion_time.txt").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].replace(".mp4", "")
            ranges = []
            for r in parts[1:]:
                m = re.fullmatch(r"([\d:]+)-([\d:]+)", r)
                if m and hseconds(m[2]) > hseconds(m[1]):
                    ranges.append((hseconds(m[1]), hseconds(m[2])))
            if name in urls and ranges:
                jobs.append((subset, safe(name), urls[name], ranges))
    return jobs


def hdtf_one(job, out, max_height):
    subset, name, url, ranges = job
    stem = f"pub_hdtf_{subset}_{name}"
    want = [os.path.join(out, f"{stem}_s{a}.mp4") for a, _ in ranges]
    if all(os.path.exists(w) for w in want):
        return stem, "have", 0
    cmd = [YTDLP, "--ffmpeg-location", FFMPEG, "--no-playlist", "--no-progress", "-q",
           "-f", f"bv*[height>=720][height<={max_height}]+ba/b[height>=720][height<={max_height}]",
           "--merge-output-format", "mp4", "--restrict-filenames",
           "-o", os.path.join(out, f"{stem}_s%(section_start)d.%(ext)s")]
    for (a, b), w in zip(ranges, want):
        if not os.path.exists(w):
            cmd += ["--download-sections", f"*{a}-{b}"]
    cmd.append(url)
    try:  # a stalled stream (e.g. after the laptop slept) must not block the pool forever
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=1800)
        err = (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
    except subprocess.TimeoutExpired:
        err = "timed out after 30 min"
    got = sum(os.path.exists(w) for w in want)
    if got == 0:
        return stem, f"FAILED: {err}", 0
    return stem, f"ok {got}/{len(want)} sections", sum(b - a for (a, b), w in zip(ranges, want) if os.path.exists(w))


def cmd_hdtf(args):
    out = raw_dir("hdtf")
    failed_path = os.path.join(out, "failed.txt")
    failed = set()
    if os.path.exists(failed_path) and not args.retry_failed:
        failed = {l.split("\t")[0] for l in open(failed_path, encoding="utf-8")}
    jobs = [j for j in hdtf_jobs() if f"pub_hdtf_{j[0]}_{j[1]}" not in failed]
    if args.limit:
        jobs = jobs[: args.limit]
    log(f"HDTF: {len(jobs)} source videos to check ({len(failed)} known-dead skipped)")
    secs = 0
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(hdtf_one, j, out, args.max_height) for j in jobs]
        for fu in cf.as_completed(futs):
            stem, status, s = fu.result()
            secs += s
            log(f"  {stem}: {status}")
            if status.startswith("FAILED"):
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{stem}\t{status}\n")
    # yt-dlp leaves .part / intermediate files when a section dies half-way
    for f in os.listdir(out):
        if f.endswith((".part", ".ytdl")) or re.search(r"\.f\d+\.\w+$", f):
            os.remove(os.path.join(out, f))
    log(f"HDTF: done, {secs / 60:.0f} min fetched this run")


# ---------------------------------------------------------------- MEAD

def drive_folder(folder_id):
    """[(title, id, is_folder)] of a public Google Drive folder (no account needed)."""
    html = fetch_text(f"https://drive.google.com/embeddedfolderview?id={folder_id}")
    out = []
    for m in re.finditer(r'<a href="([^"]+)"[^>]*>.*?flip-entry-title">([^<]+)<', html, re.S):
        href, title = m.group(1), m.group(2).strip()
        f = re.search(r"/folders/([\w-]+)", href)
        d = re.search(r"/file/d/([\w-]+)", href)
        if f:
            out.append((title, f.group(1), True))
        elif d:
            out.append((title, d.group(1), False))
    # entries whose href is an icon still carry their id in the flip-entry id attribute
    for m in re.finditer(r'id="entry-([\w-]+)".*?flip-entry-title">([^<]+)<', html, re.S):
        if not any(t == m.group(2).strip() for t, _, _ in out):
            out.append((m.group(2).strip(), m.group(1), True))
    return out


MEAD_RE = re.compile(r"^(?:\./)?video/front/([a-z]+)/level_(\d)/(\d{3})\.mp4$")


def mead_actor(actor, folder_id, out, levels):
    adir = os.path.join(out, actor)
    done = os.path.join(adir, ".done")
    if os.path.exists(done):
        return actor, "have"
    files = {t: i for t, i, _ in drive_folder(folder_id)}
    tars = [t for t in sorted(files) if re.fullmatch(r"video(_\d+)?\.tar", t)]
    if not tars:
        return actor, f"SKIPPED: no video tar in folder ({sorted(files)})"
    n = 0
    seen_front = False
    t0 = time.time()
    for name in tars:
        url = f"https://drive.usercontent.google.com/download?id={files[name]}&export=download&confirm=t"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            if "text/html" in (r.headers.get("Content-Type") or ""):
                page = r.read(4096).decode("utf-8", "replace")
                title = re.search(r"<title>([^<]*)", page)
                # Drive's per-file download quota: respect it, try again later (resets within ~24 h)
                return actor, f"UNAVAILABLE ({title[1] if title else 'html page'}) - retry later"
            with tarfile.open(fileobj=r, mode="r|") as tar:
                for m in tar:
                    hit = MEAD_RE.match(m.name)
                    if "/front/" in m.name or m.name.rstrip("/").endswith("video/front"):
                        seen_front = True
                    elif seen_front and m.name.count("/") >= 2:
                        break  # views are stored one after another: front is finished
                    if not (hit and m.isfile()):
                        continue
                    emo, lvl, idx = hit.groups()
                    if emo != "neutral" and int(lvl) not in levels:
                        continue
                    dest = os.path.join(adir, f"{emo}_L{lvl}_{idx}.mp4")
                    if os.path.exists(dest):
                        continue
                    os.makedirs(adir, exist_ok=True)
                    src = tar.extractfile(m)
                    with open(dest + ".part", "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
                    os.replace(dest + ".part", dest)
                    n += 1
        if seen_front:
            break
    if not seen_front:
        return actor, "FAILED: no front view found in archive"
    open(done, "w").close()
    return actor, f"ok {n} front clips in {time.time() - t0:.0f} s"


def cmd_mead(args):
    out = raw_dir("mead")
    actors = sorted((t, i) for t, i, is_dir in drive_folder(MEAD_FOLDER) if is_dir and re.fullmatch(r"[MW]\d{3}", t))
    men = [a for a in actors if a[0][0] == "M"]
    women = [a for a in actors if a[0][0] == "W"]
    pick = []
    for k in range(max(len(men), len(women))):  # alternate so a subset stays balanced
        pick += ([men[k]] if k < len(men) else []) + ([women[k]] if k < len(women) else [])
    pick = pick[: args.actors]
    log(f"MEAD: {len(actors)} actors listed, taking {len(pick)}: {' '.join(a for a, _ in pick)}; "
        f"levels {args.levels} + neutral, front view only")
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(mead_actor, a, i, out, set(args.levels)) for a, i in pick]
        for fu in cf.as_completed(futs):
            try:
                a, status = fu.result()
            except Exception as e:  # noqa: BLE001
                a, status = "?", f"FAILED: {e}"
            log(f"  MEAD {a}: {status}")


# ---------------------------------------------------------------- RAVDESS

def ravdess_actor(a, files, out):
    key = f"Video_Speech_Actor_{a:02d}.zip"
    adir = os.path.join(out, f"A{a:02d}")
    if os.path.exists(os.path.join(adir, ".done")):
        return f"A{a:02d}: have"
    zpath = os.path.join(out, key)
    t0 = time.time()
    download(files[key], zpath)
    os.makedirs(adir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            base = info.filename.replace("\\", "/").split("/")[-1]
            # modality 01 = audio-visual (02 is video-only), vocal channel 01 = speech
            if re.fullmatch(r"01-01-\d\d-\d\d-\d\d-\d\d-\d\d\.mp4", base):
                with z.open(info) as src, open(os.path.join(adir, base), "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
                n += 1
    os.remove(zpath)
    open(os.path.join(adir, ".done"), "w").close()
    return f"A{a:02d}: {n} audio-visual speech clips ({time.time() - t0:.0f} s)"


def cmd_ravdess(args):
    out = raw_dir("ravdess")
    rec = json.loads(fetch_text(RAVDESS_RECORD))
    files = {f["key"]: f["links"]["self"] for f in rec["files"]}
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(ravdess_actor, a, files, out) for a in range(1, args.actors + 1)]
        for fu in cf.as_completed(futs):
            try:
                log(f"  RAVDESS {fu.result()}")
            except Exception as e:  # noqa: BLE001
                log(f"  RAVDESS FAILED: {e}")


# ---------------------------------------------------------------- CREMA-D

def cmd_cremad(args):
    out = raw_dir("cremad")
    # SentenceFilenames.csv lists every clip name (e.g. 1001_DFA_ANG_XX)
    lst = fetch_text(CREMAD_MEDIA.replace("media.githubusercontent.com/media", "raw.githubusercontent.com")
                     + "SentenceFilenames.csv")
    names = sorted({m for m in re.findall(r"\b(\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2})\b", lst)})
    actors = sorted({n[:4] for n in names})
    step = max(1, len(actors) // args.actors)
    pick = actors[::step][: args.actors]  # spread over the id range
    todo = [n for n in names if n[:4] in pick]
    log(f"CREMA-D: {len(names)} clips from {len(actors)} actors listed, taking {len(pick)} actors ({len(todo)} clips)")

    def one(n):
        dest = os.path.join(out, n[:4], n + ".flv")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            download(CREMAD_MEDIA + f"VideoFlash/{n}.flv", dest, retries=3)
            return 1
        except RuntimeError:
            return 0
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        got = sum(pool.map(one, todo))
    log(f"CREMA-D: have {got}/{len(todo)} clips")


# ---------------------------------------------------------------- concat

def count_frames(path):
    import av
    with av.open(path) as c:
        return sum(1 for p in c.demux(c.streams.video[0]) if p.size)


def normalize(src, dst, vf):
    """One clip -> N frames at 30 fps (H.264) + exactly N/30 s of 48 kHz PCM (.mkv),
    audio padded or cut to the video length. Exact lengths and PCM matter: AAC
    priming (~21 ms per segment) or audio overshoot from -shortest add up to
    seconds of A/V drift once 60+ clips are joined."""
    if os.path.exists(dst):
        return dst
    vtmp = dst + ".v.mkv"
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-an",
                        "-vf", (vf + "," if vf else "") + "fps=30,format=yuv420p",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "30", vtmp],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        return None
    n = count_frames(vtmp)
    samples = n * 1600  # 48000 / 30
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", vtmp, "-i", src, "-map", "0:v:0", "-map", "1:a:0",
                        "-af", f"aresample=48000,apad=whole_len={samples},atrim=end_sample={samples}",
                        "-ac", "1", "-c:v", "copy", "-c:a", "pcm_s16le", dst + ".tmp.mkv"],
                       capture_output=True, text=True, errors="replace")
    os.remove(vtmp)
    if r.returncode != 0 or n < 15:
        return None
    os.replace(dst + ".tmp.mkv", dst)
    return dst


GROUPERS = {
    # dataset: (raw subfolder -> list of (group name, [clip paths])), video filter
    "mead": "crop=1080:1080:420:0,scale=720:720",
    "ravdess": "",
    "cremad": "",
}


def groups_for(ds, raw):
    out = []
    for actor in sorted(os.listdir(raw)):
        adir = os.path.join(raw, actor)
        if not os.path.isdir(adir):
            continue
        clips = sorted(f for f in os.listdir(adir) if f.endswith((".mp4", ".flv")))
        if ds == "mead":
            by = {}
            for c in clips:
                by.setdefault(c.split("_")[0], []).append(c)
            out += [(f"{actor}_{emo}", [os.path.join(adir, c) for c in cs]) for emo, cs in sorted(by.items())]
        else:
            out.append((actor, [os.path.join(adir, c) for c in clips]))
    return out


def cmd_concat(args):
    ds = args.dataset
    raw = raw_dir(ds)
    vdir = os.path.join(PUBLIC, ds, "videos")
    tmp = os.path.join(PUBLIC, ds, "norm")
    os.makedirs(vdir, exist_ok=True)
    groups = groups_for(ds, raw)
    log(f"concat {ds}: {len(groups)} groups")
    vf = args.vf if args.vf is not None else GROUPERS[ds]
    for name, clips in groups:
        final = os.path.join(vdir, f"pub_{ds}_{name}.mp4")
        if os.path.exists(final) or not clips:
            continue
        gtmp = os.path.join(tmp, name)
        os.makedirs(gtmp, exist_ok=True)
        dsts = [os.path.join(gtmp, os.path.splitext(os.path.basename(c))[0] + ".mkv") for c in clips]
        with cf.ThreadPoolExecutor(args.jobs) as pool:
            res = list(pool.map(lambda p: normalize(p[0], p[1], vf), zip(clips, dsts)))
        good = [(c, d) for c, d in zip(clips, res) if d]
        listing = os.path.join(gtmp, "list.txt")
        with open(listing, "w", encoding="utf-8") as f:
            for _, d in good:
                f.write(f"file '{d.replace(os.sep, '/')}'\n")
        segs, t = [], 0.0
        for c, d in good:
            dur = count_frames(d) / 30
            segs.append({"clip": os.path.splitext(os.path.basename(c))[0], "start": round(t, 3), "dur": round(dur, 3)})
            t += dur
        r = subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                            final + ".tmp.mp4"],
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            log(f"  {name}: concat FAILED {r.stderr[-300:]}")
            continue
        os.replace(final + ".tmp.mp4", final)
        with open(os.path.splitext(final)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "group": name, "segments": segs}, f, indent=0)
        shutil.rmtree(gtmp, ignore_errors=True)
        log(f"  {name}: {len(good)}/{len(clips)} clips, {t / 60:.1f} min -> {os.path.basename(final)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hdtf")
    h.add_argument("--jobs", type=int, default=4)
    h.add_argument("--limit", type=int, default=0, help="only the first N source videos")
    h.add_argument("--max-height", type=int, default=1080)
    h.add_argument("--retry-failed", action="store_true")
    m = sub.add_parser("mead")
    m.add_argument("--actors", type=int, default=24)
    m.add_argument("--levels", type=int, nargs="+", default=[3], help="emotion intensities (neutral is always taken)")
    m.add_argument("--jobs", type=int, default=2)
    r = sub.add_parser("ravdess")
    r.add_argument("--actors", type=int, default=24)
    r.add_argument("--jobs", type=int, default=6)
    c = sub.add_parser("cremad")
    c.add_argument("--actors", type=int, default=40)
    c.add_argument("--jobs", type=int, default=16)
    k = sub.add_parser("concat")
    k.add_argument("dataset", choices=sorted(GROUPERS))
    k.add_argument("--jobs", type=int, default=12)
    k.add_argument("--vf", help="ffmpeg video filter applied to every clip (default per dataset)")
    args = p.parse_args()
    {"hdtf": cmd_hdtf, "mead": cmd_mead, "ravdess": cmd_ravdess, "cremad": cmd_cremad,
     "concat": cmd_concat}[args.cmd](args)


if __name__ == "__main__":
    main()
