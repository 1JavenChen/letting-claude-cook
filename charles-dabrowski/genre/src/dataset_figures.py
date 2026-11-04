"""Dataset overview figures: class counts per split + one example log-mel per genre."""
import csv, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, G2I, SR, HOP

rows = list(csv.DictReader(open(ROOT / "data" / "labels.csv")))
out = ROOT / "results" / "dataset"; out.mkdir(parents=True, exist_ok=True)
c = Counter((r["genre"], r["split"]) for r in rows)
fig, ax = plt.subplots(figsize=(7, 3.5)); bottom = np.zeros(len(GENRES))
for split in ["train", "val", "test"]:
    v = np.array([c[(g, split)] for g in GENRES]); ax.bar(GENRES, v, bottom=bottom, label=split); bottom += v
for i, t in enumerate(bottom):
    ax.text(i, t + 10, int(t), ha="center", fontsize=8)
ax.set_ylabel("30 s clips"); ax.legend(); ax.set_title(f"{len(rows)} Deezer previews, {len({r['artist_id'] for r in rows})} artists (artist-disjoint split)")
plt.xticks(rotation=30); fig.tight_layout(); fig.savefig(out / "class_counts.png", dpi=150)

d = torch.load(ROOT / "data" / "features_train.pt")
fig, axes = plt.subplots(1, len(GENRES), figsize=(16, 3), sharey=True)
for ax, g in zip(axes, GENRES):
    i = (d["y"] == G2I[g]).nonzero()[0].item()
    ax.imshow(d["x"][i].float(), origin="lower", aspect="auto", cmap="magma", extent=[0, 30, 0, 128])
    ax.set_title(g); ax.set_xlabel("s")
axes[0].set_ylabel("mel bin"); fig.tight_layout(); fig.savefig(out / "example_spectrograms.png", dpi=150)
print("wrote", out)
