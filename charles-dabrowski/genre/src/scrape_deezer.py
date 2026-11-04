"""Collect 30 s Deezer previews for 7 broad genres.

Strategy per genre:
  seeds  = artists appearing in the genre's chart, editorial playlists and radios
  expand = breadth-first over /artist/{id}/related
  keep   = artists whose albums are mostly tagged with the target genre_id
  label  = the album's Deezer genre_id (must equal the target genre)
Caps per artist / album keep the set diverse; tracks found under two genres are dropped.
"""
import argparse, csv, json, random, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.deezer.com"
GENRES = {"classical": 98, "pop": 132, "jazz": 129, "hiphop": 116, "rock": 152, "electronic": 106, "country": 84}

ROOT = Path(__file__).resolve().parents[1] / "data"
random.seed(0)

# ----------------------------------------------------------------------------- rate-limited GET
class Limiter:
    """~45 requests / 5 s shared across threads."""
    def __init__(self, n=45, per=5.0):
        self.n, self.per, self.ts, self.lock = n, per, deque(), threading.Lock()
    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                while self.ts and now - self.ts[0] > self.per:
                    self.ts.popleft()
                if len(self.ts) < self.n:
                    self.ts.append(now); return
                sleep = self.per - (now - self.ts[0]) + 0.01
            time.sleep(sleep)

LIM = Limiter()
SESSION = requests.Session()

def get(path, **params):
    url = path if path.startswith("http") else f"{API}{path}"
    for attempt in range(8):
        LIM.wait()
        try:
            r = SESSION.get(url, params=params, timeout=20)
            d = r.json()
        except Exception as e:
            time.sleep(2 * (attempt + 1)); continue
        if isinstance(d, dict) and "error" in d:
            code = d["error"].get("code")
            if code == 4:                      # quota
                time.sleep(3 * (attempt + 1)); continue
            return None                        # not found etc.
        return d
    return None

def get_all(path, limit=100, max_items=400, **params):
    out, d = [], get(path, limit=limit, **params)
    while d and "data" in d:
        out += d["data"]
        if len(out) >= max_items or not d.get("next"):
            break
        d = get(d["next"])
    return out

# ----------------------------------------------------------------------------- per-genre crawl
def seed_artists(gid):
    ids = {}
    for t in get_all(f"/chart/{gid}/tracks", max_items=100):
        ids[t["artist"]["id"]] = t["artist"]["name"]
    ed = get(f"/editorial/{gid}/charts") or {}
    for t in ed.get("tracks", {}).get("data", []):
        ids[t["artist"]["id"]] = t["artist"]["name"]
    for a in ed.get("artists", {}).get("data", []):
        ids[a["id"]] = a["name"]
    for p in ed.get("playlists", {}).get("data", []):
        for t in get_all(f"/playlist/{p['id']}/tracks", max_items=200):
            ids[t["artist"]["id"]] = t["artist"]["name"]
    for r in (get(f"/genre/{gid}/radios") or {}).get("data", []):
        for t in get_all(f"/radio/{r['id']}/tracks", max_items=100):
            ids[t["artist"]["id"]] = t["artist"]["name"]
    return ids

def crawl_genre(name, gid, target, max_artists, per_artist, per_album, log):
    seeds = seed_artists(gid)
    log(f"[{name}] {len(seeds)} seed artists")
    queue = deque(seeds.items()); seen = set(seeds); random.shuffle(queue)
    tracks, n_artists, n_pure = {}, 0, 0
    while queue and len(tracks) < target and n_artists < max_artists:
        aid, aname = queue.popleft(); n_artists += 1
        albums = get_all(f"/artist/{aid}/albums", max_items=100)
        if not albums:
            continue
        hits = [a for a in albums if a.get("genre_id") == gid]
        purity = len(hits) / len(albums)
        if purity < 0.5 or len(hits) == 0:
            continue
        n_pure += 1
        # expand only through on-genre artists
        for rel in (get(f"/artist/{aid}/related", limit=20) or {}).get("data", []):
            if rel["id"] not in seen:
                seen.add(rel["id"]); queue.append((rel["id"], rel["name"]))
        # prefer full albums, dedupe by title (deluxe / remaster copies)
        hits.sort(key=lambda a: (a.get("record_type") != "album", a.get("title", "")))
        used_titles, got = set(), 0
        for alb in hits:
            if got >= per_artist:
                break
            key = alb["title"].lower().split("(")[0].strip()
            if key in used_titles:
                continue
            used_titles.add(key)
            ts = [t for t in get_all(f"/album/{alb['id']}/tracks", max_items=50)
                  if t.get("preview") and t.get("duration", 0) >= 60 and t["artist"]["id"] == aid]
            random.shuffle(ts)
            for t in ts[:per_album]:
                if got >= per_artist:
                    break
                tkey = (aid, t["title_short"].lower())
                if t["id"] in tracks or tkey in {(v["artist_id"], v["title"].lower()) for v in tracks.values()}:
                    continue
                tracks[t["id"]] = dict(track_id=t["id"], genre=name, genre_id=gid, artist_id=aid, artist=aname,
                                       album_id=alb["id"], album=alb["title"], title=t["title_short"],
                                       duration=t["duration"], release=alb.get("release_date", ""), preview=t["preview"])
                got += 1
        if n_artists % 20 == 0:
            log(f"[{name}] artists visited {n_artists} (on-genre {n_pure}), tracks {len(tracks)}, queue {len(queue)}")
    log(f"[{name}] done: {len(tracks)} tracks from {n_pure} artists ({n_artists} visited)")
    return list(tracks.values())

# ----------------------------------------------------------------------------- download
def download(row, out_dir):
    f = out_dir / row["genre"] / f"{row['track_id']}.mp3"
    if f.exists() and f.stat().st_size > 50_000:
        return True
    f.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            r = SESSION.get(row["preview"], timeout=30)
            if r.ok and len(r.content) > 50_000:
                f.write_bytes(r.content); return True
        except Exception:
            time.sleep(1)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1600)
    ap.add_argument("--max-artists", type=int, default=600)
    ap.add_argument("--per-artist", type=int, default=12)
    ap.add_argument("--per-album", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    def log(msg):
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    with ThreadPoolExecutor(len(GENRES)) as ex:
        futs = {ex.submit(crawl_genre, n, g, args.target, args.max_artists, args.per_artist, args.per_album, log): n
                for n, g in GENRES.items()}
        rows = [r for f in as_completed(futs) for r in f.result()]

    # drop tracks (by id or artist) that landed in more than one genre
    by_artist = {}
    for r in rows:
        by_artist.setdefault(r["artist_id"], set()).add(r["genre"])
    rows = [r for r in rows if len(by_artist[r["artist_id"]]) == 1]
    ids = {}
    for r in rows:
        ids.setdefault(r["track_id"], set()).add(r["genre"])
    rows = [r for r in rows if len(ids[r["track_id"]]) == 1]
    log(f"total after cross-genre filter: {len(rows)}")

    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    log("downloading previews...")
    ok = 0
    with ThreadPoolExecutor(16) as ex:
        for i, good in enumerate(ex.map(lambda r: download(r, out / "audio"), rows)):
            ok += good
            if i % 500 == 0:
                log(f"downloaded {i}/{len(rows)}")
    log(f"download done: {ok}/{len(rows)} ok")

if __name__ == "__main__":
    main()
