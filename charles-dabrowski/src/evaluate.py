"""
Evaluate a trained checkpoint on the test split and produce metrics + plots.

Usage:
    python src/evaluate.py --model models/cnn_v1.pt --data data/mixtures --out results/cnn_v1
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import LABEL_KEYS, N_CLASSES, MixtureDataset, SpeakerCountCNN, device


def predict(model, ds, dev, batch=128):
    dl = torch.utils.data.DataLoader(ds, batch_size=batch)
    preds = {k: [] for k in LABEL_KEYS}
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out = model(x.to(dev))
            for k in LABEL_KEYS:
                preds[k].append(out[k].argmax(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in preds.items()}


def accuracy_by(rows, correct, key_fn):
    groups = defaultdict(list)
    for r, c in zip(rows, correct):
        groups[key_fn(r)].append(c)
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(groups.items())}


def snr_bin(r):
    if r["snr_db"] == "":
        return "real recording" if r["noise_kind"].startswith("ami") else "noise only"
    s = float(r["snr_db"])
    lo = int(s // 5) * 5
    return f"{lo:02d}-{lo + 5:02d} dB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cnn_v1.pt")
    ap.add_argument("--data", default="data/mixtures")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/cnn_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()

    ckpt = torch.load(args.model, map_location=dev)
    model = SpeakerCountCNN(width=ckpt["width"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])

    ds = MixtureDataset(args.data, args.split, feature=ckpt.get("feature", "mel"))
    preds = predict(model, ds, dev)
    truth = {k: ds.labels[:, i].numpy() for i, k in enumerate(LABEL_KEYS)}
    rows = ds.rows

    # ---- headline metrics
    metrics = {}
    for k in LABEL_KEYS:
        metrics[k] = {
            "accuracy": float(np.mean(preds[k] == truth[k])),
            "mae": float(np.mean(np.abs(preds[k] - truth[k]))),
            "within_1": float(np.mean(np.abs(preds[k] - truth[k]) <= 1)),
        }
    correct = preds["n_speakers"] == truth["n_speakers"]
    metrics["count_accuracy_by_n_speakers"] = accuracy_by(rows, correct, lambda r: int(r["n_speakers"]))
    metrics["count_accuracy_by_snr"] = accuracy_by(rows, correct, snr_bin)
    metrics["count_accuracy_by_noise"] = accuracy_by(rows, correct, lambda r: r["noise_kind"])
    metrics["count_accuracy_by_reverb"] = accuracy_by(rows, correct, lambda r: "reverb" if r["reverb"] == "1" else "dry")
    metrics["n_test"] = len(rows)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)

    # ---- per-clip predictions (for human comparison / error analysis)
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", *[f"true_{k}" for k in LABEL_KEYS], *[f"pred_{k}" for k in LABEL_KEYS], "snr_db", "noise_kind", "reverb"])
        for i, r in enumerate(rows):
            w.writerow([r["file"], *[truth[k][i] for k in LABEL_KEYS], *[preds[k][i] for k in LABEL_KEYS],
                        r["snr_db"], r["noise_kind"], r["reverb"]])

    # ---- plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(truth["n_speakers"], preds["n_speakers"]):
        cm[t, p] += 1
    ax = axes[0]
    ax.imshow(cm, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax.set_xlabel("predicted speakers"); ax.set_ylabel("true speakers"); ax.set_title("Speaker count confusion")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))

    ax = axes[1]
    by_snr = {k: v for k, v in metrics["count_accuracy_by_snr"].items() if k != "noise only"}
    ax.bar(list(by_snr), [v[0] for v in by_snr.values()], color="#4c72b0")
    ax.set_ylim(0, 1); ax.set_ylabel("count accuracy"); ax.set_title("Accuracy vs SNR")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[2]
    by_n = metrics["count_accuracy_by_n_speakers"]
    ax.bar([str(k) for k in by_n], [v[0] for v in by_n.values()], color="#dd8452")
    ax.set_ylim(0, 1); ax.set_xlabel("true speakers"); ax.set_title("Accuracy vs number of speakers")

    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=130)

    # ---- print
    print(f"{args.split} set: {len(rows)} clips")
    for k in LABEL_KEYS:
        m = metrics[k]
        print(f"  {k:12s} acc {m['accuracy']:.3f}   MAE {m['mae']:.3f}   within±1 {m['within_1']:.3f}")
    print("  count acc by #speakers:", {k: f"{v[0]:.2f}" for k, v in by_n.items()})
    print("  count acc by SNR:      ", {k: f"{v[0]:.2f}" for k, v in metrics["count_accuracy_by_snr"].items()})
    print("  count acc by noise:    ", {k: f"{v[0]:.2f}" for k, v in metrics["count_accuracy_by_noise"].items()})
    print("  count acc by reverb:   ", {k: f"{v[0]:.2f}" for k, v in metrics["count_accuracy_by_reverb"].items()})
    print("saved", out / "metrics.json", out / "predictions.csv", out / "summary.png")


if __name__ == "__main__":
    main()
