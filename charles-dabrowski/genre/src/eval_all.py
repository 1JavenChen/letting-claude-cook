"""Score models on the frozen test split and the fresh 504-clip set. Usage: python src/eval_all.py cnn_v3_8cls cnn_v4_w48 clap_head clap_zero"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, load_audio, log_mel, load_model, device
dev = device(); K = len(GENRES)

def clap_embed(a):
    import clap_zero, torchaudio
    from common import SR
    st = clap_zero._load(); a48 = torchaudio.functional.resample(torch.from_numpy(a), SR, 48_000).numpy(); n = 480_000
    chunks = [a48[j:j + n] for j in range(0, len(a48) - n + 1, n)]
    with torch.no_grad(): return clap_zero._emb(st["model"].get_audio_features(**st["proc"](audio=chunks, sampling_rate=48_000, return_tensors="pt").to(st["dev"]))).float().cpu().mean(0)

def make_predictor(name):
    if name == "clap_zero":
        import clap_zero; return lambda a: clap_zero.probs(a)
    if name == "clap_head":
        ck = torch.load(ROOT / "models" / "clap_head" / "best.pt"); import torch.nn as nn
        h = ck["hidden"]; net = nn.Sequential(nn.Dropout(0), nn.Linear(512, h), nn.GELU(), nn.Dropout(0), nn.Linear(h, K)) if h else nn.Sequential(nn.Dropout(0), nn.Linear(512, K))
        net.load_state_dict(ck["state"]); net.eval()
        def f(a):
            with torch.no_grad(): return net(((clap_embed(a) - ck["mu"]) / ck["sd"])[None]).softmax(1)[0].numpy()
        return f
    if name.startswith("ensemble:"):                     # ensemble:cnn_v4_w48+clap_head -> mean of probabilities
        parts = [make_predictor(n) for n in name.split(":")[1].split("+")]
        return lambda a: np.mean([np.pad(f(a), (0, K - len(f(a)))) for f in parts], 0)
    net = load_model(name, dev)
    def f(a):
        with torch.no_grad(): return net(log_mel(a)[None, None].to(dev)).softmax(1)[0].cpu().numpy()
    return f

sets = {"test": [(ROOT / "data/audio" / r["genre"] / f"{r['track_id']}.mp3", r["genre"]) for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["split"] == "test"],
        "fresh": [(ROOT / "data/fresh/audio" / r["genre"] / f"{r['track_id']}.mp3", r["genre"]) for r in csv.DictReader(open(ROOT / "data/fresh/tracks.csv"))]}
sets = {k: [(f, g) for f, g in v if f.exists()] for k, v in sets.items()}
audio = {k: [(load_audio(f), g) for f, g in v] for k, v in sets.items()}
out = ROOT / "results" / "comparison.json"; allres = json.load(open(out)) if out.exists() else {}
for name in sys.argv[1:]:
    pred = make_predictor(name); allres[name] = {}
    for k, clips in audio.items():
        cm = np.zeros((K, K), int)
        for a, g in clips: cm[GENRES.index(g), int(np.asarray(pred(a)).argmax())] += 1
        rec = {g: cm[i, i] / max(1, cm[i].sum()) for i, g in enumerate(GENRES)}; prec = {g: cm[i, i] / max(1, cm[:, i].sum()) for i, g in enumerate(GENRES)}
        f1 = {g: 2 * rec[g] * prec[g] / max(1e-9, rec[g] + prec[g]) for g in GENRES}
        allres[name][k] = {"n": int(cm.sum()), "accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(np.mean(list(f1.values()))), "f1": f1, "confusion": cm.tolist()}
        print(f"{name:14s} {k:5s} n={cm.sum():4d} acc {allres[name][k]['accuracy']:.3f} macroF1 {allres[name][k]['macro_f1']:.3f} | " + " ".join(f"{g[:5]} {f1[g]:.2f}" for g in GENRES), flush=True)
    json.dump(allres, open(out, "w"), indent=1)
print("EVAL ALL DONE")
