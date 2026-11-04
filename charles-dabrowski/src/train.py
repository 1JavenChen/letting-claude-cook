"""
Train the speaker-count CNN.

Usage:
    python src/train.py --data data/mixtures --epochs 25 --out models/cnn_v1.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from common import LABEL_KEYS, MixtureDataset, SpeakerCountCNN, device


def run_epoch(model, loader, dev, optimizer=None, scheduler=None, ema=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, {k: 0 for k in LABEL_KEYS}, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = sum(F.cross_entropy(out[k], y[:, i], label_smoothing=0.05) for i, k in enumerate(LABEL_KEYS))
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                ema.update_parameters(model)
            total_loss += loss.item() * len(x)
            for i, k in enumerate(LABEL_KEYS):
                correct[k] += (out[k].argmax(1) == y[:, i]).sum().item()
            n += len(x)
    return total_loss / n, {k: c / n for k, c in correct.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mixtures")
    ap.add_argument("--out", default="models/cnn_v1.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to initialise weights from (fine-tuning)")
    ap.add_argument("--feature", default="mel", choices=["mel", "stft"],
                    help="input representation: log-mel (64 bins) or linear log-STFT (128 bins)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    print("device:", dev, flush=True)

    train_ds = MixtureDataset(args.data, "train", augment=True, feature=args.feature)
    val_ds = MixtureDataset(args.data, "val", feature=args.feature)
    print(f"feature: {args.feature}, input shape {tuple(train_ds[0][0].shape)}", flush=True)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch)

    model = SpeakerCountCNN(width=args.width).to(dev)
    if args.init:
        ckpt = torch.load(args.init, map_location=dev)
        model.load_state_dict(ckpt["state_dict"]); print("initialised from", args.init, flush=True)
    # EMA copy of the weights (and BatchNorm statistics): smoother, more reliable validation
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, dev, opt, sched, ema)
        va_loss, va_acc = run_epoch(ema.module, val_dl, dev)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        **{f"train_{k}": v for k, v in tr_acc.items()},
                        **{f"val_{k}": v for k, v in va_acc.items()}})
        print(f"ep {epoch:2d} | loss {tr_loss:.3f}/{va_loss:.3f} | "
              f"count acc {tr_acc['n_speakers']:.3f}/{va_acc['n_speakers']:.3f} | "
              f"male {va_acc['n_male']:.3f} female {va_acc['n_female']:.3f} | {time.time() - t0:.0f}s", flush=True)
        if va_acc["n_speakers"] > best:
            best = va_acc["n_speakers"]
            torch.save({"state_dict": ema.module.state_dict(), "width": args.width, "feature": args.feature,
                        "args": vars(args)}, out_path)

    json.dump(history, open(out_path.with_suffix(".history.json"), "w"), indent=1)
    print(f"best val count accuracy {best:.3f}; saved {out_path}")


if __name__ == "__main__":
    main()
