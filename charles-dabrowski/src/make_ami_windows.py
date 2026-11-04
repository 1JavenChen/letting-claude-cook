"""
Build a real-recording dataset from the AMI Meeting Corpus.

Slices each meeting into 4 s windows and labels every window from the manual
speaker-turn annotations (pyannote AMI-diarization-setup RTTMs, "only_words"):
a speaker counts if they talk for >= MIN_SPEECH seconds inside the window;
sex comes from AMI participants.xml (fallback: first letter of the speaker id).

Writes data/ami/labels.csv in the same schema as make_mixtures.py and the
log-mel feature caches directly (features_<split>.pt), so train.py / evaluate.py
work unchanged. No per-window wav files are written.

Splits (pyannote standard meeting split):
  train     balanced: up to --cap windows per class per mic, hop 2 s
  val       balanced: up to --cap_val per class, hop 2 s, first mic
  test      natural distribution, hop 4 s (non-overlapping), first mic
  test_bal  balanced subset of test, up to --cap_val per class

Usage:
    python src/make_ami_windows.py --mics Array1-01,Mix-Headset
"""
import argparse
import collections
import csv
import glob
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from common import SR, log_mel

WIN, MIN_SPEECH = 4.0, 0.5


def read_rttm(path):
    segs = collections.defaultdict(list)
    for line in open(path):
        p = line.split()
        if p and p[0] == "SPEAKER":
            segs[p[7]].append((float(p[3]), float(p[3]) + float(p[4])))
    return segs


def windows_for_meeting(segs, sex, hop, n_samples):
    """-> list of (start_sec, n_spk, n_male, n_female, speaker_ids)."""
    end = min(n_samples / SR, max(e for v in segs.values() for _, e in v) + 2.0)
    starts = np.arange(0.0, end - WIN + 1e-6, hop)
    spks = list(segs)
    active = np.zeros((len(starts), len(spks)))
    for j, spk in enumerate(spks):
        for s, e in segs[spk]:
            lo = np.maximum(starts, s); hi = np.minimum(starts + WIN, e)
            active[:, j] += np.clip(hi - lo, 0, None)
    on = active >= MIN_SPEECH
    out = []
    for w, st in enumerate(starts):
        ids = [spks[j] for j in range(len(spks)) if on[w, j]]
        n = len(ids)
        if n > 4:
            continue
        m = sum(sex[i] == "M" for i in ids); f = sum(sex[i] == "F" for i in ids)
        out.append((float(st), n, m, f, ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/ami")
    ap.add_argument("--out", default="data/ami")
    ap.add_argument("--mics", default="Array1-01,Mix-Headset", help="train on these mics; val/test use the first")
    ap.add_argument("--cap", type=int, default=9000, help="max train windows per class per mic")
    ap.add_argument("--cap_val", type=int, default=400, help="max val / test_bal windows per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meetings", default=None, help="comma-separated subset (for testing the script)")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    mics = args.mics.split(",")
    sex = dict(re.findall(r'nite:id="([A-Z0-9]+)" sex="([MF])"', open(raw / "annotations/corpusResources/participants.xml", encoding="latin-1").read()))
    sex_of = lambda s: sex.get(s) or s[0]  # ids start with M/F

    split_map = {"train": "train", "dev": "val", "test": "test"}
    rows, feats = [], collections.defaultdict(list)

    for rsplit, split in split_map.items():
        meetings = sorted(Path(p).stem for p in glob.glob(str(raw / f"setup/only_words/rttms/{rsplit}/*.rttm")))
        if args.meetings:
            meetings = [m for m in meetings if m in args.meetings.split(",")]
        use_mics = mics if split == "train" else mics[:1]
        hop = 4.0 if split == "test" else 2.0
        cands = collections.defaultdict(list)  # class -> [(meeting, mic, start, n, m, f, ids)]
        for m in meetings:
            segs = read_rttm(raw / f"setup/only_words/rttms/{rsplit}/{m}.rttm")
            sx = {s: sex_of(s) for s in segs}
            for mic in use_mics:
                wav = raw / "audio" / f"{m}.{mic}.wav"
                if not wav.exists():
                    print(f"  missing {wav.name}, skipping"); continue
                info = sf.info(wav)
                for st, n, mm, ff, ids in windows_for_meeting(segs, sx, hop, info.frames):
                    cands[n].append((m, mic, st, n, mm, ff, ids))
        if not cands:
            continue
        print(f"{split}: candidate windows per class", {k: len(v) for k, v in sorted(cands.items())}, flush=True)

        if split == "test":
            chosen = {"test": [w for v in cands.values() for w in v]}
            bal = []
            for k, v in cands.items():
                bal += rng.sample(v, min(len(v), args.cap_val))
            chosen["test_bal"] = bal
        else:
            cap = args.cap * len(use_mics) if split == "train" else args.cap_val
            chosen = {split: [w for k, v in cands.items() for w in rng.sample(v, min(len(v), cap))]}

        for sp, wins in chosen.items():
            wins.sort()
            print(f"  {sp}: {len(wins)} windows -> computing log-mel", flush=True)
            cur, audio = None, None
            for (m, mic, st, n, mm, ff, ids) in wins:
                key = (m, mic)
                if key != cur:
                    audio, sr = sf.read(raw / "audio" / f"{m}.{mic}.wav", dtype="float32"); assert sr == SR
                    if audio.ndim > 1:
                        audio = audio.mean(1)
                    cur = key
                a = int(st * SR); seg = audio[a:a + int(WIN * SR)]
                if len(seg) < int(WIN * SR):
                    seg = np.pad(seg, (0, int(WIN * SR) - len(seg)))
                feats[sp].append(log_mel(seg).half())
                rows.append({"file": f"{m}.{mic}@{st:.1f}", "split": sp, "n_speakers": n, "n_male": mm, "n_female": ff,
                             "snr_db": "", "reverb": 0, "noise_kind": f"ami-{mic}", "speaker_ids": " ".join(ids)})
            torch.save(torch.stack(feats[sp]), out / f"features_{sp}.pt")
            dist = collections.Counter(r["n_speakers"] for r in rows if r["split"] == sp)
            print(f"  {sp}: saved {len(feats[sp])} windows, class counts {dict(sorted(dist.items()))}", flush=True)

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "n_speakers", "n_male", "n_female", "snr_db", "reverb", "noise_kind", "speaker_ids"])
        w.writeheader(); w.writerows(rows)
    print("wrote", out / "labels.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
