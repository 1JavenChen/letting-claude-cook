"""Grow the training set: new artists only (none from the current data or the fresh test set), same labelling rules.
Downloads each preview immediately (signed URLs expire) with back-off on CDN 429s. New rows are appended to data/tracks.csv."""
import csv, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).parent))
import scrape_deezer as sd
from common import ROOT, GENRES
GID = dict(sd.GENRES, score=173); random.seed(7)
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1650
DATA = ROOT / "data"; lock = threading.Lock()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

known = list(csv.DictReader(open(DATA / "tracks.csv")))
fresh = list(csv.DictReader(open(DATA / "fresh" / "tracks.csv"))) if (DATA / "fresh" / "tracks.csv").exists() else []
blocked = {str(r["artist_id"]) for r in known} | {str(r["artist_id"]) for r in fresh}
known_tracks = {str(r["track_id"]) for r in known} | {str(r["track_id"]) for r in fresh}
by_genre = {g: sorted({r["artist_id"] for r in known if r["genre"] == g}) for g in GENRES}

def download(row):
    f = DATA / "audio" / row["genre"] / f"{row['track_id']}.mp3"; f.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            r = sd.SESSION.get(row["preview"], timeout=30)
            if r.status_code == 200 and len(r.content) > 50_000:
                f.write_bytes(r.content); return True
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1)); continue
            return False
        except Exception:
            time.sleep(2)
    return False

def crawl(g):
    gid = GID[g]; rows, seen = [], set(blocked)
    queue = by_genre[g][:]; random.shuffle(queue); queue = [(a, True) for a in queue]   # (artist_id, is_seed)
    visited = 0
    while queue and len(rows) < TARGET:
        aid, is_seed = queue.pop(0)
        if not is_seed:
            albums = sd.get_all(f"/artist/{aid}/albums", max_items=100); hits = [a for a in albums if a.get("genre_id") == gid]
            if not albums or len(hits) / len(albums) < 0.5: continue
            visited += 1
            hits.sort(key=lambda a: (a.get("record_type") != "album", a.get("title", ""))); used, got = set(), 0
            for alb in hits:
                if got >= 12: break
                key = alb["title"].lower().split("(")[0].strip()
                if key in used: continue
                used.add(key)
                ts = [t for t in sd.get_all(f"/album/{alb['id']}/tracks", max_items=50) if t.get("preview") and t.get("duration", 0) >= 60 and str(t["artist"]["id"]) == str(aid) and str(t["id"]) not in known_tracks]
                random.shuffle(ts)
                for t in ts[:4]:
                    if got >= 12: break
                    row = dict(track_id=t["id"], genre=g, genre_id=gid, artist_id=aid, artist=t["artist"]["name"], album_id=alb["id"], album=alb["title"],
                               title=t["title_short"], duration=t["duration"], release=alb.get("release_date", ""), preview=t["preview"])
                    if download(row):
                        rows.append(row); got += 1; time.sleep(0.15)
            if visited % 25 == 0: log(f"[{g}] {len(rows)} clips from {visited} new artists, queue {len(queue)}")
        for rel in (sd.get(f"/artist/{aid}/related", limit=25) or {}).get("data", []):
            if str(rel["id"]) not in seen:
                seen.add(str(rel["id"])); queue.append((rel["id"], False))
    log(f"[{g}] done: {len(rows)} new clips from {visited} new artists")
    return rows

with ThreadPoolExecutor(len(GENRES)) as ex:
    all_rows = [r for rs in ex.map(crawl, GENRES) for r in rs]
with open(DATA / "tracks.csv", "a", newline="") as f:
    csv.DictWriter(f, fieldnames=list(known[0].keys())).writerows(all_rows)
log(f"appended {len(all_rows)} rows to tracks.csv")
log("EXTEND DONE")
