"""Train a small classifier on frozen CLAP embeddings (data/features_clap.pt). Saves models/clap_head/best.pt."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES
d = torch.load(ROOT / "data" / "features_clap.pt"); X, y, sp = d["x"].float(), d["y"], np.array(d["split"])
mu, sd = X[sp == "train"].mean(0), X[sp == "train"].std(0) + 1e-6; Xn = (X - mu) / sd
tr, va, te = (torch.from_numpy(sp == s) for s in ("train", "val", "test"))
torch.manual_seed(0)
def run(hidden, wd, drop, epochs=300, lr=3e-3):
    net = nn.Sequential(nn.Dropout(drop), nn.Linear(512, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, len(GENRES))) if hidden else nn.Sequential(nn.Dropout(drop), nn.Linear(512, len(GENRES)))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 0, None
    for ep in range(epochs):
        net.train(); idx = torch.randperm(int(tr.sum()))
        xb, yb = Xn[tr][idx], y[tr][idx]
        for i in range(0, len(xb), 256):
            loss = nn.functional.cross_entropy(net(xb[i:i + 256]), yb[i:i + 256], label_smoothing=0.05); opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad(): acc = (net(Xn[va]).argmax(1) == y[va]).float().mean().item()
        if acc > best: best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); return best, net
results = []
for hidden in (0, 256, 512):
    for wd in (1e-2, 1e-1):
        for drop in (0.2, 0.5):
            acc, net = run(hidden, wd, drop); results.append((acc, hidden, wd, drop, net)); print(f"hidden {hidden:3d} wd {wd} drop {drop}: val {acc:.3f}", flush=True)
acc, hidden, wd, drop, net = max(results, key=lambda t: t[0])
with torch.no_grad(): test_acc = (net(Xn[te]).argmax(1) == y[te]).float().mean().item()
print(f"best: hidden {hidden} wd {wd} drop {drop} val {acc:.3f} test {test_acc:.3f}")
out = ROOT / "models" / "clap_head"; out.mkdir(parents=True, exist_ok=True)
torch.save({"state": net.state_dict(), "hidden": hidden, "mu": mu, "sd": sd, "val_acc": acc, "test_acc": test_acc}, out / "best.pt")
json.dump({"hidden": hidden, "wd": wd, "drop": drop, "val_acc": acc, "test_acc": test_acc, "n_train": int(tr.sum())}, open(out / "config.json", "w"), indent=1)
