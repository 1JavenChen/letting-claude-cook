"""Decide who is talking when two webcams share one audio track.

After running prepare_data.py twice on the same video (two crops, two tags):

  1. Voiceprints (tools/voices.py) group the audio into k voice groups.
  2. Each group is assigned by whose mouth moves during it:
       - one person's mouth clearly moves more   -> that person's voice
       - neither mouth moves much               -> background (game, quiet)
       - both move / unclear                    -> mixed
  3. Each person's `keep` mask becomes: face found AND (their own voice OR
     background). The other person's voice and mixed stretches are dropped.
  4. Mouth veto, frame by frame: voiceprints work on 1.5 s windows, so they
     miss short interjections and crosstalk. A frame is also dropped if the
     other person's mouth is moving about as much as this person's.

Mouth movement is measured relative to each person's own typical level,
because small webcams have tracking jitter.

    python tools/pair_speakers.py data/processed/video_top.npz data/processed/video_bottom.npz
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import FPS, audio_db, rolling_mean  # noqa: E402
from voices import to_frames, voice_groups  # noqa: E402

ARTICULATION = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
                "mouthLowerDownLeft", "mouthLowerDownRight"]


def rolling_std(x, width):
    m = rolling_mean(x, width)
    return np.sqrt(np.maximum(rolling_mean(x * x, width) - m * m, 0))


def activity(data):
    """Mouth movement over ~0.5 s, as a multiple of this person's typical level."""
    names = list(data["names"])
    labels, face_ok = data["labels"], data["face_ok"]
    a = sum(rolling_std(labels[:, names.index(s)], FPS // 2) for s in ARTICULATION)
    a = np.where(face_ok, a, 0.0)
    return a / (np.median(a[face_ok]) + 1e-6)


def assign_groups(groups, act_a, act_b, ratio=1.5, moving=1.2, still=1.0):
    """Map each voice group to 'a', 'b', 'background' or 'mixed'."""
    owners = {}
    for g in sorted(set(groups)):
        m = groups == g
        ma, mb = np.median(act_a[m]), np.median(act_b[m])
        if g == -1 or (ma < still and mb < still):
            owner = "background"
        elif ma >= moving and ma >= ratio * mb:
            owner = "a"
        elif mb >= moving and mb >= ratio * ma:
            owner = "b"
        else:
            owner = "mixed"
        owners[g] = (owner, m.mean(), ma, mb)
    return owners


def write_kept_samples(audio, talking, name, out_dir, n=6, min_s=1.5, max_s=4.0, seed=0):
    """Write random stretches of exactly the frames kept as this person talking."""
    os.makedirs(out_dir, exist_ok=True)
    edges = np.diff(np.r_[0, talking.astype(np.int8), 0])
    runs = [(s, e) for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0])
            if e - s >= min_s * FPS]
    rng = np.random.default_rng(seed)
    hop = 16000 // FPS
    for i, r in enumerate(sorted(rng.choice(len(runs), min(n, len(runs)), replace=False))):
        s, e = runs[r]
        e = min(e, s + int(max_s * FPS))
        sf.write(os.path.join(out_dir, f"{name}_sample{i}_{s / FPS:.0f}s.wav"), audio[s * hop: e * hop], 16000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--k", type=int, default=6, help="number of voice groups to look for")
    p.add_argument("--listen", help="folder to write sample .wavs of exactly the frames "
                   "kept as each person talking")
    p.add_argument("--moving", type=float, default=1.5,
                   help="mouth veto: other person counts as talking above this x their typical level")
    p.add_argument("--ratio", type=float, default=0.7,
                   help="...and at least this x my own movement")
    args = p.parse_args()

    da, db_ = dict(np.load(args.a)), dict(np.load(args.b))
    n = min(len(da["labels"]), len(db_["labels"]))
    for d in (da, db_):
        for k in ("labels", "keep", "face_ok"):
            d[k] = d[k][:n]
        d["audio"] = d["audio"][: n * (16000 // FPS)]
    tag_a, tag_b = (os.path.basename(x)[:-4].split("_")[-1] for x in (args.a, args.b))

    act_a, act_b = activity(da), activity(db_)
    centres, win_groups = voice_groups(da["audio"], args.k)
    groups = to_frames(centres, win_groups, n)
    owners = assign_groups(groups, act_a, act_b)
    names = {"a": tag_a, "b": tag_b, "background": "background", "mixed": "mixed"}
    print("voice groups (mouth movement: 1.0 = that person's typical level):")
    for g, (owner, share, ma, mb) in owners.items():
        print(f"  group {g:2d}: {share:5.1%} of time | {tag_a} {ma:4.2f} | {tag_b} {mb:4.2f} -> {names[owner]}")
    owner_of = np.array([owners[g][0] for g in groups])

    results = []
    for d, me, mine, other, path in ((da, "a", act_a, act_b, args.a), (db_, "b", act_b, act_a, args.b)):
        other_talking = (other >= args.moving) & (other >= args.ratio * mine)
        keep = d["face_ok"] & ((owner_of == me) | (owner_of == "background")) & ~other_talking
        d["keep_solo"] = d.get("keep_solo", d["keep"])
        d["keep"] = keep
        d["voice_owner"] = owner_of
        d["role"] = me  # which side of the pair this person is ("a" or "b")
        np.savez_compressed(path, **d)
        talking = (owner_of == me) & keep
        results.append((names[me], keep, talking))
        print(f"{os.path.basename(path)}: own voice {talking.mean():.0%} of time, kept {keep.mean():.0%} of frames "
              f"({keep.sum() / FPS / 60:.1f} min; {(keep & talking).sum() / FPS / 60:.1f} min of them talking)")

    if args.listen:
        for name, _, talking in results:
            write_kept_samples(da["audio"], talking, name, args.listen)
        print(f"Listen to check: {args.listen}")

    level = audio_db(da["audio"], n)
    t = np.arange(n) / FPS
    span = slice(0, min(n, 300 * FPS))  # plot the first 5 minutes
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t[span], level[span], lw=0.5, color="0.3")
    axes[0].set_ylabel("audio dB")
    for ax, d, (name, keep, talk) in zip(axes[1:], (da, db_), results):
        jaw = d["labels"][:, list(d["names"]).index("jawOpen")]
        ax.plot(t[span], jaw[span], lw=0.6, color="tab:red")
        ax.fill_between(t[span], 0, 1, where=talk[span], transform=ax.get_xaxis_transform(),
                        color="tab:green", alpha=0.25, lw=0, label="own voice (kept)")
        ax.fill_between(t[span], 0, 1, where=~keep[span], transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.3, lw=0, label="dropped")
        ax.set_ylabel(f"{name}\njawOpen")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(args.a), "pair_" + os.path.basename(args.a)[:-4] + ".png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"Plot (first 5 min): {out}")


if __name__ == "__main__":
    main()
