"""Voiceprints: group a clip's audio into voices with an open speaker-ID model.

Uses SpeechBrain's ECAPA model (ungated, Apache-2.0) to turn every 1.5 s
window of audio into a 192-number voiceprint, then groups similar voiceprints
with spherical k-means. Which group belongs to which person is decided later
(pair_speakers.py) by checking whose mouth moves during that group's windows.

    python tools/voices.py data/processed/video_top.npz [--listen out/voices]

--listen writes a few short .wav samples per group so you can hear what each
group is (e.g. you / your friend / game audio).
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.blendshapes import FPS  # noqa: E402

SR = 16000
WIN_S, HOP_S = 1.5, 0.5


def load_encoder(device):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(ROOT, "models", "ecapa"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,  # Windows: avoid symlinks needing admin
    )


@torch.no_grad()
def embed(audio, device="cuda", batch=256):
    """Return (window_centres_s, embeddings[N, 192] unit-length, window_db)."""
    enc = load_encoder(device)
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    starts = np.arange(0, max(1, len(audio) - win + 1), hop)
    embs, dbs = [], []
    for i in range(0, len(starts), batch):
        segs = np.stack([audio[s: s + win] for s in starts[i: i + batch]])
        dbs.append(20 * np.log10(np.sqrt((segs ** 2).mean(1)) + 1e-6))
        e = enc.encode_batch(torch.from_numpy(segs).to(device)).squeeze(1)
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return (starts + win / 2) / SR, np.concatenate(embs), np.concatenate(dbs)


def kmeans(x, k, iters=50, seed=0):
    """Spherical k-means (cosine similarity) with k-means++ style seeding."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = np.clip(1 - np.max(x @ np.array(centres).T, axis=1), 0, None) + 1e-12
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(centres)
    for _ in range(iters):
        assign = np.argmax(x @ c.T, axis=1)
        for j in range(k):
            if (assign == j).any():
                m = x[assign == j].mean(0)
                c[j] = m / (np.linalg.norm(m) + 1e-9)
    return np.argmax(x @ c.T, axis=1), c


def voice_groups(audio, k=4, device="cuda"):
    """Group windows into k voice clusters; quiet windows get group -1."""
    centres_s, emb, db = embed(audio, device)
    loud = db > np.percentile(db, 95) - 35
    groups = np.full(len(emb), -1)
    groups[loud], _ = kmeans(emb[loud], k)
    return centres_s, groups


def to_frames(centres_s, groups, n_frames):
    """Each 30 fps frame takes the group of the nearest window centre."""
    t = np.arange(n_frames) / FPS
    idx = np.clip(np.searchsorted(centres_s, t), 0, len(centres_s) - 1)
    prev = np.clip(idx - 1, 0, len(centres_s) - 1)
    nearer = np.where(np.abs(centres_s[prev] - t) < np.abs(centres_s[idx] - t), prev, idx)
    return groups[nearer]


def write_samples(audio, centres_s, groups, out_dir, per_group=4, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for g in sorted(set(groups) - {-1}):
        idx = np.nonzero(groups == g)[0]
        for i, w in enumerate(rng.choice(idx, min(per_group, len(idx)), replace=False)):
            c = int(centres_s[w] * SR)
            clip = audio[max(0, c - 2 * SR): c + 2 * SR]
            sf.write(os.path.join(out_dir, f"group{g}_sample{i}_{centres_s[w]:.0f}s.wav"), clip, SR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--listen", help="folder to write sample .wavs per voice group")
    args = p.parse_args()
    audio = np.load(args.npz)["audio"]
    centres_s, groups = voice_groups(audio, args.k)
    for g in sorted(set(groups)):
        print(f"group {g:2d}: {np.mean(groups == g):5.1%} of windows" + ("  (quiet)" if g == -1 else ""))
    if args.listen:
        write_samples(audio, centres_s, groups, args.listen)
        print(f"Samples written to {args.listen}")


if __name__ == "__main__":
    main()
