"""mp3 previews -> cached log-mel tensors with an artist-disjoint train/val/test split."""
import csv, hashlib, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, G2I, load_audio, log_mel

DATA = ROOT / "data"

def assign_splits(rows):
    """Artist-disjoint split, stratified per genre: artists sorted by hash, 80/10/10 by track count."""
    for g in {r["genre"] for r in rows}:
        per_artist = {}
        for r in rows:
            if r["genre"] == g:
                per_artist.setdefault(r["artist_id"], []).append(r)
        order = sorted(per_artist, key=lambda a: hashlib.md5(str(a).encode()).hexdigest())
        total, seen = sum(len(v) for v in per_artist.values()), 0
        for a in order:
            frac = seen / total
            split = "train" if frac < 0.8 else "val" if frac < 0.9 else "test"
            for r in per_artist[a]:
                r["split"] = split
            seen += len(per_artist[a])

def featurize(row):
    f = DATA / "audio" / row["genre"] / f"{row['track_id']}.mp3"
    try:
        return row["track_id"], log_mel(load_audio(f)).half()
    except Exception as e:
        return row["track_id"], None

def main():
    rows = [r for r in csv.DictReader(open(DATA / "tracks.csv"))
            if (DATA / "audio" / r["genre"] / f"{r['track_id']}.mp3").exists()]
    prev = DATA / "labels.csv"
    if prev.exists():                                   # keep existing splits frozen; new artists go to train only
        old = {r["artist_id"]: r["split"] for r in csv.DictReader(open(prev))}
        for r in rows:
            r["split"] = old.get(r["artist_id"], "train")
        print(f"kept splits for {sum(r['artist_id'] in old for r in rows)} clips, {sum(r['artist_id'] not in old for r in rows)} new clips -> train", flush=True)
    else:
        assign_splits(rows)
    print(f"{len(rows)} tracks with audio", flush=True)
    with ProcessPoolExecutor(8) as ex:
        feats = dict(ex.map(featurize, rows, chunksize=16))
    rows = [r for r in rows if feats[r["track_id"]] is not None]
    print(f"{len(rows)} decoded", flush=True)
    with open(DATA / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for split in ["train", "val", "test"]:
        rs = [r for r in rows if r["split"] == split]
        x = torch.stack([feats[r["track_id"]] for r in rs])
        y = torch.tensor([G2I[r["genre"]] for r in rs])
        torch.save({"x": x, "y": y, "ids": [r["track_id"] for r in rs]}, DATA / f"features_{split}.pt")
        counts = {g: int((y == i).sum()) for g, i in G2I.items()}
        print(split, tuple(x.shape), counts, flush=True)

if __name__ == "__main__":
    main()
