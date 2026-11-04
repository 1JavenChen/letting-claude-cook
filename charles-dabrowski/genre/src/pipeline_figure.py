"""One clip through the pipeline: waveform -> log-mel -> CNN probabilities."""
import csv, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, SR, load_audio, log_mel, load_model, device
name = sys.argv[1] if len(sys.argv) > 1 else "cnn_v3_8cls"; want = sys.argv[2] if len(sys.argv) > 2 else "country"
LABELS = [g.replace("hiphop", "hip-hop").replace("score", "film score") for g in GENRES]
rows = [r for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["split"] == "test" and r["genre"] == want]
dev = device(); model = load_model(name, dev)
for r in rows[:40]:                                   # first confidently-correct test clip of that genre
    f = ROOT / "data/audio" / r["genre"] / f"{r['track_id']}.mp3"; a = load_audio(f); m = log_mel(a)
    with torch.no_grad(): p = model(m[None, None].to(dev)).softmax(1)[0].cpu().numpy()
    if GENRES[p.argmax()] == want and p.max() > 0.8: break
fig, ax = plt.subplots(1, 3, figsize=(15, 2.45), gridspec_kw={"width_ratios": [2.2, 3.2, 1.9], "wspace": 0.5})
t = np.arange(len(a)) / SR; ax[0].plot(t[::20], a[::20], lw=.4, color="#4B2C7F"); ax[0].set_xlim(0, 30); ax[0].set_ylim(-1, 1)
ax[0].set_title(f"1  waveform: {r['artist']} – {r['title'][:28]}", fontsize=11, loc="left"); ax[0].set_xlabel("time (s)"); ax[0].set_yticks([])
ax[0].text(0.5, -0.45, "661 500 samples at 22 kHz", transform=ax[0].transAxes, ha="center", fontsize=9, color="#6B687A")
ax[1].imshow(m, origin="lower", aspect="auto", cmap="magma", extent=[0, 30, 0, 128]); ax[1].set_title("2  log-mel spectrogram, what the CNN sees", fontsize=11, loc="left")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("mel band", labelpad=1)
ax[1].text(0.5, -0.45, "128 frequency bands × 646 frames; training uses random 10 s crops", transform=ax[1].transAxes, ha="center", fontsize=9, color="#6B687A")
order = np.argsort(p)[::-1]; cols = ["#E8702A" if i == order[0] else "#C9C3D8" for i in order]
ax[2].barh([LABELS[i] for i in order][::-1], p[order][::-1], color=cols[::-1]); ax[2].set_xlim(0, 1); ax[2].set_title("3  CNN output: P(genre)", fontsize=11, loc="left")
for i, v in zip(order[::-1], p[order][::-1]): ax[2].text(v + 0.02, LABELS[i], f"{v:.0%}", va="center", fontsize=9)
ax[2].set_xticks([]); ax[2].spines[["top", "right", "bottom"]].set_visible(False)
ax[2].text(0.5, -0.45, "5 conv blocks → global pooling → 2 dense layers, 2.5 M parameters", transform=ax[2].transAxes, ha="center", fontsize=9, color="#6B687A")
for a_ in ax[:2]: a_.spines[["top", "right"]].set_visible(False)
fig.canvas.draw()
for k in (0, 1): fig.text(ax[k].get_position().x1 + 0.022, 0.5, "→", fontsize=26, ha="center", va="center", color="#4B2C7F")
fig.savefig(ROOT / "results/dataset/pipeline_example.png", dpi=170, bbox_inches="tight"); print("wrote pipeline_example.png", r["artist"], r["title"], p.max())
