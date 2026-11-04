"""CLAP audio embeddings (512-d, mean of three 10 s chunks) for every clip in labels.csv -> data/features_clap.pt"""
import csv, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load_audio, G2I
import clap_zero
import torchaudio, numpy as np
from common import SR
OUT = ROOT / "data" / "features_clap.pt"
rows = list(csv.DictReader(open(ROOT / "data" / "labels.csv")))
done = torch.load(OUT) if OUT.exists() else {"ids": [], "x": [], "y": [], "split": []}
have = set(done["ids"]); todo = [r for r in rows if r["track_id"] not in have]
print(f"{len(todo)} clips to embed ({len(have)} cached)", flush=True)
st = clap_zero._load(); X = list(done["x"]) if torch.is_tensor(done["x"]) is False else list(done["x"].unbind(0))
t0 = time.time()
for i, r in enumerate(todo):
    a = load_audio(ROOT / "data" / "audio" / r["genre"] / f"{r['track_id']}.mp3")
    a48 = torchaudio.functional.resample(torch.from_numpy(a), SR, 48_000).numpy(); n = 480_000
    chunks = [a48[j:j + n] for j in range(0, len(a48) - n + 1, n)]
    inp = st["proc"](audio=chunks, sampling_rate=48_000, return_tensors="pt").to(st["dev"])
    with torch.no_grad():
        e = clap_zero._emb(st["model"].get_audio_features(**inp)).float().cpu().mean(0)
    X.append(e); done["ids"].append(r["track_id"]); done["y"].append(G2I[r["genre"]]); done["split"].append(r["split"])
    if i % 500 == 0 or i == len(todo) - 1:
        torch.save({"ids": done["ids"], "x": torch.stack(X), "y": torch.tensor(done["y"]), "split": done["split"]}, OUT)
        print(time.strftime("%H:%M:%S"), f"{i + 1}/{len(todo)} ({(time.time() - t0) / (i + 1):.2f} s/clip)", flush=True)
print("CLAP EMBED DONE", flush=True)
