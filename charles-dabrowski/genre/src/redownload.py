"""Re-download previews whose signed URL expired: fetch a fresh URL per track and save it immediately."""
import csv, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scrape_deezer import get, download, ROOT

rows = list(csv.DictReader(open(ROOT / "tracks.csv")))
miss = [r for r in rows if not (ROOT / "audio" / r["genre"] / f"{r['track_id']}.mp3").exists()]
print(f"{len(miss)} missing", flush=True)

def fetch(r):
    t = get(f"/track/{r['track_id']}")
    if not t or not t.get("preview"):
        return False
    r["preview"] = t["preview"]
    return download(r, ROOT / "audio")

ok = 0
with ThreadPoolExecutor(8) as ex:
    for i, good in enumerate(ex.map(fetch, miss)):
        ok += good
        if i % 500 == 0:
            print(time.strftime("%H:%M:%S"), f"redownloaded {i}/{len(miss)} (ok {ok})", flush=True)
print(time.strftime("%H:%M:%S"), f"redownload done: {ok}/{len(miss)} ok", flush=True)
