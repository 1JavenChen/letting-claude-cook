"""How much audio does the model need?  Test accuracy vs clip length (centre crop) for one model."""
import json, sys
from pathlib import Path
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GenreDataset, GenreCNN, device, SR, HOP, load_model
name = sys.argv[1]; dev = device()
model = load_model(name, dev)
ds = GenreDataset("test"); x, y = ds.x, ds.y
res = {}
for secs in [2, 3, 5, 10, 15, 30]:
    n = int(secs * SR / HOP); s = (x.shape[2] - n) // 2
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), 64):
            xb = x[i:i + 64, :, s:s + n].float().unsqueeze(1).to(dev)
            correct += (model(xb).argmax(1).cpu() == y[i:i + 64]).sum().item()
    res[secs] = correct / len(y); print(f"{secs:3d} s  acc {res[secs]:.3f}", flush=True)
json.dump(res, open(ROOT / "results" / name / "acc_vs_cliplen.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(4.5, 3)); ax.plot(list(res), list(res.values()), "o-")
ax.set_xlabel("clip length (s)"); ax.set_ylabel("test accuracy"); ax.set_xscale("log"); ax.set_xticks(list(res)); ax.set_xticklabels(list(res))
ax.set_title(f"{name}: accuracy vs clip length"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(ROOT / "results" / name / "acc_vs_cliplen.png", dpi=150)
