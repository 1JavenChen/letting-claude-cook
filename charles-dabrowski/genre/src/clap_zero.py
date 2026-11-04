"""Zero-shot genre with LAION CLAP (music-trained checkpoint): no training, just text prompts vs audio embeddings."""
import os, sys
from pathlib import Path
import numpy as np, torch, torchaudio
sys.path.insert(0, str(Path(__file__).parent))
from common import GENRES, SR

CKPT = os.environ.get("CLAP_CKPT", "laion/larger_clap_general")   # larger_clap_music gives degenerate embeddings with transformers 5
PROMPTS = {"classical": ["classical music", "a classical orchestra or piano piece"],
           "pop": ["pop music", "a pop song"],
           "jazz": ["jazz music", "a jazz song"],
           "hiphop": ["hip hop music", "a rap song"],
           "rock": ["rock music", "a rock song with electric guitars"],
           "electronic": ["electronic dance music", "an electronic track with synthesizers and a drum machine"],
           "country": ["country music", "a country song"],
           "score": ["a film score", "orchestral soundtrack music from a movie"]}
TEMPLATES = ["This is {}.", "The sound of {}.", "{}"]
_state = {}
import threading
_LOAD_LOCK = threading.Lock()

def _emb(out):
    """transformers >= 5 returns an output object; older versions return the tensor."""
    return out if torch.is_tensor(out) else out.pooler_output

def _load():
  with _LOAD_LOCK:
    if not _state:
        from transformers import ClapModel, ClapProcessor
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = ClapModel.from_pretrained(CKPT).to(dev).eval(); proc = ClapProcessor.from_pretrained(CKPT)
        texts, owner = [], []
        for g in GENRES:
            for p in PROMPTS[g]:
                for t in TEMPLATES:
                    texts.append(t.format(p)); owner.append(g)
        with torch.no_grad():
            te = _emb(model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True).to(dev)))
        te = torch.nn.functional.normalize(te, dim=-1)
        T = torch.stack([te[[i for i, o in enumerate(owner) if o == g]].mean(0) for g in GENRES]); T = torch.nn.functional.normalize(T, dim=-1)
        _state.update(model=model, proc=proc, dev=dev, T=T)
    return _state

def probs(audio_22k: np.ndarray, n_classes=len(GENRES)) -> np.ndarray:
    """audio at SR (22 050 Hz), any length -> P(genre). Embeds 10 s chunks (CLAP's window) and averages."""
    st = _load(); a = torchaudio.functional.resample(torch.from_numpy(audio_22k), SR, 48_000).numpy()
    n = 48_000 * 10; chunks = [a[i:i + n] for i in range(0, max(1, len(a) - n + 1), n)] or [a]
    inp = st["proc"](audio=chunks, sampling_rate=48_000, return_tensors="pt").to(st["dev"])
    with torch.no_grad():
        ae = torch.nn.functional.normalize(_emb(st["model"].get_audio_features(**inp)), dim=-1).mean(0, keepdim=True)
        ae = torch.nn.functional.normalize(ae, dim=-1)
        logits = st["model"].logit_scale_a.exp() * ae @ st["T"][:n_classes].T
    return logits.softmax(-1)[0].float().cpu().numpy()

if __name__ == "__main__":
    # evaluate on the test split and write results/clap_zero_shot/metrics_test.json + confusion png
    import csv, json, time
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import ROOT, load_audio
    rows = [r for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["split"] == "test"]
    out = ROOT / "results" / ("clap_" + CKPT.split("/")[-1].replace("-", "_")); out.mkdir(exist_ok=True)
    K = len(GENRES); cm = np.zeros((K, K), int); t0 = time.time()
    for i, r in enumerate(rows):
        p = probs(load_audio(ROOT / "data/audio" / r["genre"] / f"{r['track_id']}.mp3"))
        cm[GENRES.index(r["genre"]), p.argmax()] += 1
        if i % 100 == 0:
            print(time.strftime("%H:%M:%S"), f"{i}/{len(rows)} running acc {np.trace(cm)/max(1,cm.sum()):.3f}", flush=True)
    acc = np.trace(cm) / cm.sum(); rec = {g: cm[i, i] / cm[i].sum() for i, g in enumerate(GENRES)}; prec = {g: cm[i, i] / max(1, cm[:, i].sum()) for i, g in enumerate(GENRES)}
    f1 = {g: 2 * rec[g] * prec[g] / max(1e-9, rec[g] + prec[g]) for g in GENRES}
    json.dump({"model": CKPT, "split": "test", "n": int(cm.sum()), "accuracy": acc, "recall": rec, "precision": prec, "f1": f1, "macro_f1": float(np.mean(list(f1.values()))), "confusion": cm.tolist()}, open(out / "metrics_test.json", "w"), indent=1)
    print(f"CLAP zero-shot test acc {acc:.3f} macro F1 {np.mean(list(f1.values())):.3f}"); [print(f"  {g:11s} f1 {f1[g]:.2f}") for g in GENRES]
    cmn = cm / cm.sum(1, keepdims=True); fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(GENRES, rotation=45, ha="right"); ax.set_yticklabels(GENRES)
    for i in range(K):
        for j in range(K): ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"CLAP zero-shot — test acc {acc:.1%}"); fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "confusion_test.png", dpi=150)
    print("CLAP EVAL DONE")
