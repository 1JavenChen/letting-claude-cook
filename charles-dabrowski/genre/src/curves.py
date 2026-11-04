"""Training curves from results/<name>/train.json for one or more models."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
names = sys.argv[1:]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for n in names:
    h = json.load(open(ROOT / "results" / n / "train.json"))["history"]
    ax[0].plot([e["epoch"] for e in h], [e["loss"] for e in h], label=n)
    ax[1].plot([e["epoch"] for e in h], [e["val_acc"] for e in h], label=n)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy")
ax[1].legend(); fig.tight_layout(); fig.savefig(ROOT / "results" / ("curves_" + "_".join(names) + ".png"), dpi=150)
