"""Train the genre CNN. Usage: python src/train.py --name cnn_v1 --epochs 30"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, GenreDataset, GenreCNN, device

def evaluate(model, loader, dev):
    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(dev)).argmax(1).cpu()); ys.append(y)
    p, y = torch.cat(preds), torch.cat(ys)
    cm = np.zeros((len(GENRES), len(GENRES)), int)
    for a, b in zip(y.tolist(), p.tolist()):
        cm[a, b] += 1
    return (p == y).float().mean().item(), cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--crop", type=int, default=216)  # ~10 s
    ap.add_argument("--init", default=None, help="model name to initialise from (final layer re-initialised)")
    args = ap.parse_args()
    dev = device(); torch.manual_seed(0); np.random.seed(0)
    out = ROOT / "models" / args.name; out.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results" / args.name; res.mkdir(parents=True, exist_ok=True)

    tr = GenreDataset("train", crop=args.crop, augment=True)
    va = GenreDataset("val")
    counts = torch.bincount(tr.y, minlength=len(GENRES)).float()
    w = (counts.sum() / counts) ** 0.5; w = w / w.mean()
    print("train counts", counts.tolist(), "class weights", [round(v, 2) for v in w.tolist()], flush=True)
    tl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(va, 32, num_workers=2)

    model = GenreCNN(width=args.width).to(dev)
    if args.init:
        sd = torch.load(ROOT / "models" / args.init / "best.pt", map_location=dev)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.4.")}
        print("init from", args.init, model.load_state_dict(sd, strict=False), flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tl), pct_start=0.15)
    crit = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=0.1)
    best, hist = 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0, tot, n = time.time(), 0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        acc, _ = evaluate(model, vl, dev)
        hist.append({"epoch": ep, "loss": tot / n, "val_acc": acc})
        print(f"ep {ep:2d} loss {tot/n:.3f} val_acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc; torch.save(model.state_dict(), out / "best.pt")
    json.dump({"args": vars(args), "best_val_acc": best, "history": hist}, open(res / "train.json", "w"), indent=1)
    print("best val acc", best)

if __name__ == "__main__":
    main()
