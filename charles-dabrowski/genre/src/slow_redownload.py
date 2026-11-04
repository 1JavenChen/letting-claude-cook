"""Gentle retry for previews the CDN throttled (HTTP 429): 1 request at a time, pauses on 429."""
import csv, sys, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scrape_deezer import get, ROOT
time.sleep(240)   # cool-down first
rows = list(csv.DictReader(open(ROOT / "tracks.csv")))
miss = [r for r in rows if not (ROOT / "audio" / r["genre"] / f"{r['track_id']}.mp3").exists()]
miss.sort(key=lambda r: r["genre"] != "classical")      # classical first (it is the short class)
print(f"{len(miss)} missing", flush=True); ok = 0
for i, r in enumerate(miss):
    t = get(f"/track/{r['track_id']}")
    if not t or not t.get("preview"):
        continue
    for attempt in range(4):
        x = requests.get(t["preview"], timeout=30)
        if x.status_code == 200 and len(x.content) > 50_000:
            f = ROOT / "audio" / r["genre"] / f"{r['track_id']}.mp3"; f.write_bytes(x.content); ok += 1; break
        if x.status_code == 429:
            time.sleep(30 * (attempt + 1))
        else:
            break
    time.sleep(0.5)
    if i % 100 == 0:
        print(time.strftime("%H:%M:%S"), f"{i}/{len(miss)} ok {ok}", flush=True)
print(time.strftime("%H:%M:%S"), f"slow redownload done: {ok}/{len(miss)}", flush=True)
