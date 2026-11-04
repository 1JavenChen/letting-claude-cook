"""How do genres differ on a spectrogram?  (a) mean log-mel spectrum per genre, (b) most typical clip per genre
(closest to the genre's mean spectrum), (c) PCA of the CNN's 512-d embeddings on the test set."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, SR, N_MELS, load_model, device, GenreDataset
import torchaudio
name = sys.argv[1] if len(sys.argv) > 1 else "cnn_v3_8cls"
out = ROOT / "results" / "dataset"; out.mkdir(exist_ok=True)
LABELS = [g.replace("hiphop", "hip-hop").replace("score", "film score") for g in GENRES]
COLORS = ["#4B2C7F", "#E8702A", "#2A9D8F", "#E63946", "#1D3557", "#F4A261", "#6A994E", "#9B5DE5"]

d = torch.load(ROOT / "data" / "features_train.pt"); x, y = d["x"], d["y"]
mel_hz = torchaudio.functional.melscale_fbanks(1025, 20, SR // 2, N_MELS, SR).argmax(0) * (SR / 2) / 1024   # centre freq per mel bin
# (a) mean spectrum per genre (dB relative to each genre's own max, so shapes are comparable)
fig, ax = plt.subplots(figsize=(7, 3.6))
spec = {}
for i, g in enumerate(GENRES):
    m = x[y == i].float().mean((0, 2)).numpy(); spec[g] = m
    ax.plot(mel_hz, 10 * (m - m.max()) / np.log(10), color=COLORS[i], lw=2, label=LABELS[i])
ax.set_xscale("log"); ax.set_xlim(40, 11000); ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k"])
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("mean energy (dB, rel. to peak)"); ax.set_title("Average spectrum per genre (all training clips)")
ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=9, frameon=False); fig.tight_layout(); fig.savefig(out / "mean_spectrum.png", dpi=170)

# (b) most typical clip per genre, shared colour scale
fig, axes = plt.subplots(1, len(GENRES), figsize=(17, 2.9), sharey=True)
vmin, vmax = np.percentile(x[:2000].float().numpy(), [2, 99.5])
for i, (ax, g) in enumerate(zip(axes, GENRES)):
    idx = (y == i).nonzero().flatten()
    per_clip = x[idx].float().mean(2)                        # (n, 128) time-averaged spectra
    j = idx[(per_clip - torch.from_numpy(spec[g])).pow(2).sum(1).argmin()]
    ax.imshow(x[j].float(), origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, extent=[0, 30, 0, N_MELS])
    ax.set_title(LABELS[i], color=COLORS[i], fontweight="bold"); ax.set_xlabel("time (s)"); ax.set_xticks([0, 10, 20, 30])
axes[0].set_ylabel("mel band (low → high pitch)"); fig.tight_layout(); fig.savefig(out / "typical_spectrograms.png", dpi=170)

# (c) PCA of CNN embeddings on the test split
dev = device(); model = load_model(name, dev); ds = GenreDataset("test")
feats, ys = [], []
with torch.no_grad():
    for i in range(0, len(ds), 32):
        xb = ds.x[i:i + 32].float().unsqueeze(1).to(dev)
        h = model.features(model.norm(xb)); feats.append(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1).cpu()); ys.append(ds.y[i:i + 32])
F, Y = torch.cat(feats), torch.cat(ys); F = (F - F.mean(0)) / (F.std(0) + 1e-6)
U, S, V = torch.pca_lowrank(F, q=2, center=False); P = (F @ V[:, :2]).numpy()
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, g in enumerate(GENRES):
    ax.scatter(P[Y == i, 0], P[Y == i, 1], s=9, alpha=.65, color=COLORS[i], label=LABELS[i], edgecolors="none")
ax.set_title("Test clips in the trained network's embedding space (PCA)"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(ncol=4, fontsize=8.5, frameon=False, markerscale=2.2, loc="upper center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(out / "embedding_pca.png", dpi=170)
print("explained var", (S ** 2 / (F ** 2).sum()).tolist()); print("wrote", out)
