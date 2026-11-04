"""Evaluate a trained model on the test split; writes metrics.json + confusion matrix png."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, GenreDataset, GenreCNN, device, load_model
from train import evaluate

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--width", type=int, default=32); args = ap.parse_args()
    dev = device(); model = load_model(args.name, dev)
    ds = GenreDataset(args.split); dl = torch.utils.data.DataLoader(ds, 32)
    acc, cm = evaluate(model, dl, dev)
    K = model.head[-1].out_features; cm = cm[:K, :K]; GENRES_ = GENRES[:K]
    per_class = {g: cm[i, i] / max(1, cm[i].sum()) for i, g in enumerate(GENRES_)}
    prec = {g: cm[i, i] / max(1, cm[:, i].sum()) for i, g in enumerate(GENRES_)}
    f1 = {g: 2 * per_class[g] * prec[g] / max(1e-9, per_class[g] + prec[g]) for g in GENRES_}
    res = ROOT / "results" / args.name
    json.dump({"split": args.split, "n": int(cm.sum()), "accuracy": acc, "recall": per_class, "precision": prec, "f1": f1,
               "macro_f1": float(np.mean(list(f1.values()))), "confusion": cm.tolist()}, open(res / f"metrics_{args.split}.json", "w"), indent=1)
    print(f"{args.split}: n={cm.sum()} acc={acc:.3f} macro_f1={np.mean(list(f1.values())):.3f}")
    for g in GENRES_:
        print(f"  {g:11s} recall {per_class[g]:.2f} precision {prec[g]:.2f} f1 {f1[g]:.2f}")
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(GENRES_, rotation=45, ha="right"); ax.set_yticklabels(GENRES_)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{args.name} — {args.split} acc {acc:.1%}")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(res / f"confusion_{args.split}.png", dpi=150)

if __name__ == "__main__":
    main()
