"""Train the audio + webcam fusion model (model/fusion.py).

Target: full-resolution face tracking (all blendshapes).
Input:  the audio features, plus the face tracking from a weaker camera - the
        same video tracked at low resolution (tools/degrade_track.py) or the
        full-res tracking with noise - with random stretches where the face is
        lost or the mouth is hidden, so the model learns to lean on the voice.

    python train_fusion.py --clips ai_is_dumb,synth_talk

Evaluates on the held-out last 10% of each clip under several conditions
(weak camera alone, fused, audio only, mouth hidden) and writes
out/fusion_eval.csv with per-blendshape results.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS  # noqa: E402
from model.data import WindowDataset, load_clips, split  # noqa: E402
from model.encoders import ENCODERS  # noqa: E402
from model.fusion import FusionHead  # noqa: E402
from train import masked_loss  # noqa: E402

GROUPS = {
    "jaw + mouth": lambda n: n.startswith(("jaw", "mouth")),
    "smile": lambda n: n.startswith("mouthSmile"),
    "brows": lambda n: n.startswith("brow"),
    "blinks": lambda n: n.startswith("eyeBlink"),
    "eyes (gaze, squint, wide)": lambda n: n.startswith("eye") and not n.startswith("eyeBlink"),
}


def mouth_columns(names):
    return np.array([n.startswith(("jaw", "mouth", "cheekPuff")) for n in names])


def random_spans(rng, n, count, lo_s, hi_s):
    m = np.ones(n, bool)
    for _ in range(count):
        length = int(rng.uniform(lo_s, hi_s) * FPS)
        s = rng.integers(0, max(1, n - length))
        m[s:s + length] = False
    return m


def make_input(clip, s, e, rng, mouth_cols, train=True):
    """Weak-camera tracking for frames s:e, plus seen-flags, with augmentation."""
    n = e - s
    if hasattr(clip, "labels_lo") and (not train or rng.random() < 0.6):
        video = clip.labels_lo[s:e].copy()
        face = clip.face_ok_lo[s:e].copy()
    else:
        video = clip.labels[s:e] + rng.normal(0, rng.uniform(0, 0.05), (n, clip.labels.shape[1]))
        video = np.clip(video, 0, 1).astype(np.float32)
        face = clip.face_ok[s:e].copy()
    mouth = np.ones(n, bool)
    if train:
        r = rng.random()
        if r < 0.15:
            face[:] = False  # camera off / face lost the whole time
        elif r < 0.5:
            face &= random_spans(rng, n, rng.integers(1, 4), 0.3, 2.5)
        if rng.random() < 0.25:
            mouth &= random_spans(rng, n, rng.integers(1, 3), 0.5, 3.0)
    mouth &= face
    video[~face] = 0
    video[np.ix_(~mouth, mouth_cols)] = 0
    return video, face.astype(np.float32), mouth.astype(np.float32)


class FusionDataset(WindowDataset):
    def __init__(self, ranges, window, samples, mouth_cols, seed=0):
        super().__init__(ranges, window, samples, seed=seed, with_clip=True)
        self.mouth_cols = mouth_cols

    def __getitem__(self, i):
        clip, s = super().__getitem__(i)
        e = s + self.window
        video, face, mouth = make_input(clip, s, e, self.rng, self.mouth_cols)
        t = torch.from_numpy
        return (t(np.asarray(clip.feats[s:e], np.float32)), t(video), t(face), t(mouth),
                t(clip.labels[s:e]), t(clip.keep[s:e].astype(np.float32)))


def weighted_loss(pred, target, mask, w):
    return masked_loss(pred * w, target * w, mask)


@torch.no_grad()
def predict(head, clip, lo, hi, device, video, face, mouth):
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None].to(device)  # noqa: E731
    return head(t(clip.feats[lo:hi]), t(video), t(face), t(mouth))[0].cpu().numpy()


@torch.no_grad()
def evaluate(head, clips, val, device, mouth_cols, train_mean):
    """Predictions and targets for each condition over all held-out frames."""
    head.eval()
    rng = np.random.default_rng(0)
    out = {k: [] for k in ["weak camera alone", "fused: weak camera + audio", "audio only",
                           "fused, mouth hidden", "always-average guess"]}
    targets, keeps = [], []
    for clip, (lo, hi) in zip(clips, val):
        video, face, mouth = make_input(clip, lo, hi, rng, mouth_cols, train=False)
        held = video.copy()  # camera alone: hold the last seen value when the face is lost
        for i in range(1, len(held)):
            if not face[i]:
                held[i] = held[i - 1]
        out["weak camera alone"].append(held)
        out["fused: weak camera + audio"].append(predict(head, clip, lo, hi, device, video, face, mouth))
        z = np.zeros_like(face)
        out["audio only"].append(predict(head, clip, lo, hi, device, np.zeros_like(video), z, z))
        vm = video.copy()
        vm[:, mouth_cols] = 0
        out["fused, mouth hidden"].append(predict(head, clip, lo, hi, device, vm, face, z))
        out["always-average guess"].append(np.broadcast_to(train_mean, video.shape))
        targets.append(clip.labels[lo:hi])
        keeps.append(clip.keep[lo:hi])
    head.train()
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(targets), np.concatenate(keeps)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 0.0


def vertex_error_mm(pred, target, names):
    """Mean distance (mm) between where the lips / eyelids land on the ICT 3D head under the
    predicted vs. the target weights - the error you actually see on the avatar."""
    from realtime.retarget import EYES, LIPS, LandmarkRetargeter
    rt = vertex_error_mm.rt = getattr(vertex_error_mm, "rt", None) or LandmarkRetargeter(list(names))
    K = len(rt.ok)
    Jp = rt.J[:3 * K] / rt.sw[:3 * K, None]                 # unweighted cm per unit weight, [3K, free]
    diff = (pred - target)[:, rt.free] @ Jp.T              # [T, 3K] cm
    dist = np.linalg.norm(diff.reshape(len(diff), K, 3), axis=2) * 10
    lips, eyes = np.isin(rt.ok, list(LIPS)), np.isin(rt.ok, list(EYES))
    return dist[:, lips].mean(), dist[:, eyes].mean()


def report(preds, target, keep, names, path):
    try:
        print("  3D error on the head (mm, lower is better):   lips / eyelids")
        for c in preds:
            lip, eye = vertex_error_mm(preds[c][keep], target[keep], names)
            print(f"    {c:40s} {lip:5.2f} / {eye:5.2f}")
    except Exception as e:  # noqa: BLE001  (missing ICT assets shouldn't break training)
        print(f"  (3D error unavailable: {e})")
    names = np.array(names)
    print(f"\nHeld-out results ({keep.sum() / FPS / 60:.1f} min). "
          "Correlation with full-quality tracking, averaged over each group (1 = perfect):")
    conds = list(preds)
    print(f"  {'':28s}" + "".join(f"{c[:22]:>24s}" for c in conds))
    for g, f in GROUPS.items():
        cols = [i for i, n in enumerate(names) if f(n)]
        row = [np.mean([corr(preds[c][keep, i], target[keep, i]) for i in cols]) for c in conds]
        print(f"  {g:28s}" + "".join(f"{v:24.2f}" for v in row))
    rows = []
    for i, n in enumerate(names):
        rows.append([n, f"{target[keep, i].std():.3f}"] +
                    [f"{corr(preds[c][keep, i], target[keep, i]):.3f}" for c in conds])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["blendshape", "target_std"] + conds)
        w.writerows(rows)
    print(f"Per-blendshape table: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True, help="clip name substrings, comma-separated")
    p.add_argument("--encoder", default="wav2vec2", choices=ENCODERS)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--lookahead", type=int, default=1, help="future frames per block")
    p.add_argument("--lookahead-total", type=int, help="total future frames instead (1 per block, "
                   "first blocks first), e.g. 2 = 67 ms")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", default=os.path.join(ROOT, "models", "fusion.pt"))
    p.add_argument("--eval-clips", help="clips whose held-out part picks the best checkpoint "
                   "(default: the training clips)")
    p.add_argument("--report-clips", help="extra clips to report on at the end, e.g. your own "
                   "clips after pretraining on public data (their held-out part only)")
    p.add_argument("--init", help="start from this checkpoint (fine-tuning)")
    p.add_argument("--target", default="labels", help="label set to learn: labels (MediaPipe) or "
                   "labels_hyb (hybrid geometric, tools/make_hybrid.py)")
    p.add_argument("--report", default=os.path.join(ROOT, "out", "fusion_eval.csv"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    patterns = ",".join(x for x in (args.clips, args.eval_clips, args.report_clips) if x)
    clips = load_clips(ROOT, args.encoder, patterns, shapes="all", target=args.target)
    if not clips:
        sys.exit("No matching clips with cached features.")
    matches = lambda c, pat: bool(pat) and any(x in c.name for x in pat.split(","))  # noqa: E731
    names = [n for n in clips[0].names if n != "_neutral"]
    keep_cols = [clips[0].names.index(n) for n in names]
    for c in clips:  # drop MediaPipe's "_neutral" score: not a face shape
        c.names = names
        c.labels = np.ascontiguousarray(c.labels[:, keep_cols])
        if hasattr(c, "labels_lo"):
            c.labels_lo = np.ascontiguousarray(c.labels_lo[:, keep_cols])
    mcols = mouth_columns(names)

    splits = {c.name: split(c, args.val_frac) for c in clips}
    train_clips = [c for c in clips if matches(c, args.clips)]
    eval_clips = [c for c in clips if matches(c, args.eval_clips or args.clips)]
    report_clips = [c for c in clips if matches(c, args.report_clips)]
    train_ranges = [(c, *splits[c.name][0]) for c in train_clips]
    val = [splits[c.name][1] for c in eval_clips]
    lo_note = sum(hasattr(c, "labels_lo") for c in train_clips)
    print(f"{len(train_clips)} training clips ({lo_note} with low-res tracking), "
          f"{sum(c.keep.sum() for c in train_clips) / FPS / 60:.1f} min usable, {len(names)} blendshapes; "
          f"{len(eval_clips)} clips for checkpoint selection")

    tr_labels = np.concatenate([c.labels[lo:hi][c.keep[lo:hi]] for c, lo, hi in train_ranges])
    train_mean = tr_labels.mean(0)
    std = tr_labels.std(0)
    w = 1 / (std + 0.02)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)  # every shape counts

    ds = FusionDataset(train_ranges, int(args.window * FPS), args.steps * args.batch, mcols)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch)
    head = FusionHead(ENCODERS[args.encoder].dim, len(names), args.hidden, args.blocks,
                      lookahead=args.lookahead, dropout=args.dropout, prior=train_mean,
                      mouth_cols=mcols, lookahead_total=args.lookahead_total).to(device)
    if args.init:
        ck = torch.load(args.init)
        assert ck["names"] == names, "checkpoint was trained on different blendshapes"
        head.load_state_dict(ck["state_dict"])
        head.prior.copy_(torch.as_tensor(train_mean))  # the new speaker's average face
        print(f"Starting from {args.init}")
    print(f"Fusion head: {sum(q.numel() for q in head.parameters()) / 1e6:.2f}M parameters, "
          f"look-ahead {head.lookahead_frames / FPS * 1000:.0f} ms, training on {device}")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    best, running, t0 = float("inf"), [], time.time()
    for step, batch in enumerate(loader, start=1):
        feats, video, face, mouth, target, mask = (x.to(device) for x in batch)
        loss = weighted_loss(head(feats, video, face, mouth), target, mask, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            preds, target_np, keep = evaluate(head, eval_clips, val, device, mcols, train_mean)
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(device)  # noqa: E731
            vl = weighted_loss(t(preds["fused: weak camera + audio"]), t(target_np), t(keep), w).item()
            al = weighted_loss(t(preds["audio only"]), t(target_np), t(keep), w).item()
            flag = ""
            if vl < best:
                best, flag = vl, "  * saved"
                torch.save({"state_dict": head.state_dict(), "config": head.config, "names": names,
                            "encoder": args.encoder, "fps": FPS, "target": args.target}, args.out)
            print(f"step {step:6d} | train {np.mean(running):.4f} | val fused {vl:.4f} "
                  f"audio-only {al:.4f} | {time.time() - t0:.0f}s{flag}")
            running = []

    head.load_state_dict(torch.load(args.out)["state_dict"])
    preds, target_np, keep = evaluate(head, eval_clips, val, device, mcols, train_mean)
    report(preds, target_np, keep, names, args.report)
    if report_clips:
        print(f"\n--- On {len(report_clips)} report clip(s): {args.report_clips} ---")
        rval = [splits[c.name][1] for c in report_clips]
        rmean = np.concatenate([c.labels[c.keep] for c in report_clips]).mean(0)
        preds, target_np, keep = evaluate(head, report_clips, rval, device, mcols, rmean)
        report(preds, target_np, keep, names, args.report.replace(".csv", "_report.csv"))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
