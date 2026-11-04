"""Fresh evaluation set: N new 30 s Deezer previews from artists NOT in the dataset, then CNN vs CLAP zero-shot."""
import csv, json, random, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import scrape_deezer as sd
from common import ROOT, GENRES, load_audio, log_mel, load_model, device
GID = dict(sd.GENRES, score=173); random.seed(1)
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 63
OUT = ROOT / "data" / "fresh"; OUT.mkdir(exist_ok=True); (OUT / "audio").mkdir(exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

known = list(csv.DictReader(open(ROOT / "data" / "tracks.csv")))
known_artists = {r["artist_id"] for r in known}; known_tracks = {r["track_id"] for r in known}
by_genre = {g: sorted({r["artist_id"] for r in known if r["genre"] == g}) for g in GENRES}

def collect(g):
    gid = GID[g]; rows, seen = [], set(known_artists)
    seeds = random.sample(by_genre[g], min(60, len(by_genre[g])))
    for aid in seeds:
        for rel in (sd.get(f"/artist/{aid}/related", limit=25) or {}).get("data", []):
            if len(rows) >= N_PER: return rows
            if str(rel["id"]) in seen: continue
            seen.add(str(rel["id"]))
            albums = sd.get_all(f"/artist/{rel['id']}/albums", max_items=60)
            hits = [a for a in albums if a.get("genre_id") == gid]
            if not albums or len(hits) / len(albums) < 0.5: continue
            random.shuffle(hits); got = 0
            for alb in hits[:3]:
                ts = [t for t in sd.get_all(f"/album/{alb['id']}/tracks", max_items=40) if t.get("preview") and t.get("duration", 0) >= 60 and str(t["id"]) not in known_tracks]
                if not ts: continue
                t = random.choice(ts)
                row = dict(track_id=t["id"], genre=g, artist_id=rel["id"], artist=rel["name"], album=alb["title"], title=t["title_short"], preview=t["preview"])
                if sd.download(row, OUT / "audio"):
                    rows.append(row); got += 1
                if got >= 2 or len(rows) >= N_PER: break
    return rows

if not (OUT / "tracks.csv").exists():
    rows = []
    for g in GENRES:
        r = collect(g); rows += r; log(f"[{g}] {len(r)} fresh clips")
    with open(OUT / "tracks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
rows = [r for r in csv.DictReader(open(OUT / "tracks.csv")) if (OUT / "audio" / r["genre"] / f"{r['track_id']}.mp3").exists()]
log(f"{len(rows)} fresh clips from {len({r['artist_id'] for r in rows})} new artists")

# ---------------------------------------------------------------- evaluate both
import clap_zero
dev = device(); net = load_model("cnn_v3_8cls", dev); K = len(GENRES)
cm_cnn, cm_clap, agree, both_right = np.zeros((K, K), int), np.zeros((K, K), int), 0, 0
preds = []
for i, r in enumerate(rows):
    a = load_audio(OUT / "audio" / r["genre"] / f"{r['track_id']}.mp3"); y = GENRES.index(r["genre"])
    with torch.no_grad(): pc = net(log_mel(a)[None, None].to(dev)).softmax(1)[0].cpu().numpy()
    pz = clap_zero.probs(a)
    cm_cnn[y, pc.argmax()] += 1; cm_clap[y, pz.argmax()] += 1
    agree += pc.argmax() == pz.argmax(); both_right += (pc.argmax() == y) and (pz.argmax() == y)
    preds.append({**r, "cnn": GENRES[pc.argmax()], "cnn_p": float(pc.max()), "clap": GENRES[pz.argmax()], "clap_p": float(pz.max())})
    if i % 100 == 0: log(f"{i}/{len(rows)} cnn {np.trace(cm_cnn)/max(1,cm_cnn.sum()):.3f} clap {np.trace(cm_clap)/max(1,cm_clap.sum()):.3f}")

def metrics(cm):
    rec = {g: cm[i, i] / max(1, cm[i].sum()) for i, g in enumerate(GENRES)}; prec = {g: cm[i, i] / max(1, cm[:, i].sum()) for i, g in enumerate(GENRES)}
    f1 = {g: 2 * rec[g] * prec[g] / max(1e-9, rec[g] + prec[g]) for g in GENRES}
    return {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(np.mean(list(f1.values()))), "f1": f1, "recall": rec, "confusion": cm.tolist()}
res = {"n": len(rows), "n_artists": len({r["artist_id"] for r in rows}), "per_genre_n": {g: int(cm_cnn[i].sum()) for i, g in enumerate(GENRES)},
       "cnn_v3_8cls": metrics(cm_cnn), "clap_larger_clap_general": metrics(cm_clap), "agreement": agree / len(rows), "both_right": both_right / len(rows)}
rd = ROOT / "results" / "fresh_test"; rd.mkdir(exist_ok=True)
json.dump(res, open(rd / "metrics.json", "w"), indent=1)
with open(rd / "predictions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(preds[0].keys())); w.writeheader(); w.writerows(preds)
print(json.dumps({k: v for k, v in res.items() if k not in ("cnn_v3_8cls", "clap_larger_clap_general")}))
for name in ("cnn_v3_8cls", "clap_larger_clap_general"):
    m = res[name]; print(f"{name:26s} acc {m['accuracy']:.3f} macro F1 {m['macro_f1']:.3f} | " + " ".join(f"{g[:5]} {m['f1'][g]:.2f}" for g in GENRES))
print("FRESH TEST DONE")
