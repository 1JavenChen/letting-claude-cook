"""Train the mouth head on cached audio features (Option A).

    python train.py                       # defaults
    python train.py --steps 20000 --hidden 384

Saves the best checkpoint (by validation loss) to models/mouth_<encoder>.pt
and a training curve to out/train_<encoder>.png.
"""

import argparse
import math
import os
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS, MOUTH_SHAPES  # noqa: E402
from lipsync import rule_based  # noqa: E402
from model.data import WindowDataset, load_clips, split  # noqa: E402
from model.encoders import ENCODERS  # noqa: E402
from model.head import MouthHead  # noqa: E402

REPORT_SHAPES = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthStretchLeft", "mouthPressLeft"]


def masked_loss(pred, target, mask):
    """MSE on values plus MSE on frame-to-frame motion (so it learns timing,
    not just average poses), ignoring frames without trustworthy labels."""
    m = mask[..., None]
    pos = (((pred - target) ** 2) * m).sum() / (m.sum() * pred.shape[-1] + 1e-8)
    mv = m[:, 1:] * m[:, :-1]
    vel = ((((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])) ** 2) * mv).sum()
    vel = vel / (mv.sum() * pred.shape[-1] + 1e-8)
    return pos + 10.0 * vel


@torch.no_grad()
def evaluate(head, clips, val_ranges, train_mean, device):
    """Validation loss + per-shape correlation, vs. trivial and Phase 1 baselines."""
    head.eval()
    preds, targets, masks, rule_jaw = [], [], [], []
    for clip, (lo, hi) in zip(clips, val_ranges):
        feats = torch.from_numpy(np.asarray(clip.feats[lo:hi], np.float32))[None].to(device)
        preds.append(head(feats)[0].cpu().numpy())
        targets.append(clip.labels[lo:hi])
        masks.append(clip.keep[lo:hi])
        audio = clip.audio[lo * (rule_based.SR // FPS): hi * (rule_based.SR // FPS)]
        rb = rule_based.rules(rule_based.frame_features(audio))[: hi - lo]
        rule_jaw.append(np.pad(rb[:, 0], (0, hi - lo - len(rb))))
    head.train()
    pred, tgt, keep = np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)
    rule_jaw = np.concatenate(rule_jaw)

    def loss_of(p):
        t = torch.from_numpy
        return masked_loss(t(p)[None], t(tgt)[None], t(keep.astype(np.float32))[None]).item()

    def corr(a, b):
        a, b = a[keep], b[keep]
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 0.0

    return {
        "loss": loss_of(pred),
        "loss_mean_baseline": loss_of(np.broadcast_to(train_mean, tgt.shape).copy()),
        "corr": {s: corr(pred[:, MOUTH_SHAPES.index(s)], tgt[:, MOUTH_SHAPES.index(s)]) for s in REPORT_SHAPES},
        "rule_based_jaw_corr": corr(rule_jaw, tgt[:, 0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="wav2vec2", choices=ENCODERS)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--window", type=float, default=4.0, help="seconds per training window")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--clips", help="only clips whose name contains one of these (comma-separated)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    clips = load_clips(ROOT, args.encoder, args.clips)
    if not clips:
        sys.exit("No clips with cached features. Run tools/prepare_data.py then tools/extract_features.py.")
    splits = [split(c, args.val_frac) for c in clips]
    train_ranges = [(c, *tr) for c, (tr, _) in zip(clips, splits)]
    val_ranges = [va for _, va in splits]
    kept_min = sum(c.keep.sum() for c in clips) / FPS / 60
    print(f"{len(clips)} clips, {kept_min:.1f} min of usable labelled audio, training on {device}")

    window = int(args.window * FPS)
    ds = WindowDataset(train_ranges, window, samples=args.steps * args.batch)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch)
    train_mean = np.concatenate([c.labels[lo:hi][c.keep[lo:hi]] for c, lo, hi in train_ranges]).mean(0)

    head = MouthHead(ENCODERS[args.encoder].dim, len(MOUTH_SHAPES), args.hidden, args.blocks).to(device)
    print(f"Mouth head: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M trainable parameters")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    ckpt_path = os.path.join(ROOT, "models", f"mouth_{args.encoder}.pt")
    history, best, running, t0 = [], float("inf"), [], time.time()
    for step, (feats, labels, mask) in enumerate(loader, start=1):
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
        loss = masked_loss(head(feats), labels, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(head, clips, val_ranges, train_mean, device)
            history.append((step, np.mean(running), ev["loss"]))
            running = []
            flag = ""
            if ev["loss"] < best:
                best, flag = ev["loss"], "  * saved"
                torch.save({"state_dict": head.state_dict(), "config": head.config,
                            "encoder": args.encoder, "names": MOUTH_SHAPES, "fps": FPS}, ckpt_path)
            print(f"step {step:6d} | train {history[-1][1]:.4f} | val {ev['loss']:.4f} "
                  f"(always-average guess: {ev['loss_mean_baseline']:.4f}) | "
                  f"{time.time() - t0:.0f}s{flag}")

    head.load_state_dict(torch.load(ckpt_path)["state_dict"])
    ev = evaluate(head, clips, val_ranges, train_mean, device)
    print("\nBest model, held-out correlation with your real mouth (1.0 = perfect, 0 = unrelated):")
    for s, c in ev["corr"].items():
        print(f"  {s:18s} {c:+.2f}")
    print(f"  (Phase 1 rule-based jawOpen: {ev['rule_based_jaw_corr']:+.2f})")

    steps, tr, va = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, tr, label="train")
    plt.plot(steps, va, label="validation (unseen speech)")
    plt.axhline(ev["loss_mean_baseline"], ls="--", c="gray", label="always-average guess")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "out", f"train_{args.encoder}.png"), dpi=90)
    print(f"Saved {ckpt_path} and out/train_{args.encoder}.png")


if __name__ == "__main__":
    main()
