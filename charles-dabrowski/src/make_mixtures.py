"""
Generate labeled "noisy room" clips by overlaying clean LibriSpeech utterances.

Each output clip is a fixed-length mono wav plus a row in a CSV with exact labels:
    n_speakers, n_male, n_female, snr_db, reverb, speaker_ids

Speakers are split into train / val / test by speaker ID so the model never
hears a test speaker during training.

Usage:
    python src/make_mixtures.py --n_train 4000 --n_val 500 --n_test 500
    python src/make_mixtures.py --n_train 20 --n_val 0 --n_test 0 --out data/demo   # quick listen
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
CLIP_SEC = 4.0
CLIP_LEN = int(SR * CLIP_SEC)


# ----------------------------------------------------------------------------- data index
def load_speaker_sex(librispeech_root: Path) -> dict[int, str]:
    """Parse SPEAKERS.TXT -> {speaker_id: 'M' | 'F'}."""
    sex = {}
    with open(librispeech_root / "SPEAKERS.TXT") as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2 and parts[0].isdigit():
                sex[int(parts[0])] = parts[1]
    return sex


def index_utterances(subset_dir: Path) -> dict[int, list[Path]]:
    """{speaker_id: [flac paths]} for one LibriSpeech subset (e.g. dev-clean)."""
    utts: dict[int, list[Path]] = {}
    for flac in sorted(subset_dir.rglob("*.flac")):
        spk = int(flac.parts[-3])
        utts.setdefault(spk, []).append(flac)
    return utts


def split_speakers(speakers: list[int], rng: random.Random, frac=(0.7, 0.15, 0.15)):
    spk = speakers[:]
    rng.shuffle(spk)
    n = len(spk)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return {"train": spk[:a], "val": spk[a:b], "test": spk[b:]}


# ----------------------------------------------------------------------------- audio utils
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def random_speech_segment(paths: list[Path], rng: random.Random) -> np.ndarray:
    """Pick a random CLIP_LEN window that actually contains speech (not mostly silence)."""
    for _ in range(20):
        audio, sr = sf.read(rng.choice(paths), dtype="float32")
        assert sr == SR, f"expected {SR} Hz, got {sr}"
        if len(audio) < CLIP_LEN:
            audio = np.pad(audio, (0, CLIP_LEN - len(audio)))
        start = rng.randint(0, len(audio) - CLIP_LEN)
        seg = audio[start : start + CLIP_LEN]
        # reject windows that are mostly silence relative to the whole utterance
        if rms(seg) > 0.5 * rms(audio):
            return seg / rms(seg)
    return seg / rms(seg)


def babble_noise(utts: dict[int, list[Path]], speakers: list[int], rng: random.Random, n_voices=12) -> np.ndarray:
    """Cocktail-party background: many faint voices summed together."""
    chosen = rng.sample(speakers, min(n_voices, len(speakers)))
    noise = sum(random_speech_segment(utts[s], rng) for s in chosen)
    return noise / rms(noise)


def colored_noise(rng: random.Random, kind: str) -> np.ndarray:
    """White or pink (1/f) noise, unit RMS."""
    g = np.random.default_rng(rng.getrandbits(32))
    white = g.standard_normal(CLIP_LEN).astype(np.float32)
    if kind == "white":
        return white / rms(white)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(CLIP_LEN)
    spec[1:] /= np.sqrt(freqs[1:])  # 1/f power -> 1/sqrt(f) amplitude
    pink = np.fft.irfft(spec, n=CLIP_LEN).astype(np.float32)
    return pink / rms(pink)


def apply_reverb(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Cheap synthetic room: exponentially decaying noise impulse response."""
    rt60 = rng.uniform(0.2, 0.8)  # seconds for -60 dB
    n = int(rt60 * SR)
    g = np.random.default_rng(rng.getrandbits(32))
    ir = g.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0  # direct path
    wet = np.convolve(x, ir)[:CLIP_LEN]
    return wet / rms(wet) * rms(x)


# ----------------------------------------------------------------------------- mixing
def make_mixture(utts, speakers, sex, rng, max_speakers, snr_range, p_reverb, noise_kinds):
    n_spk = rng.randint(0, max_speakers)
    chosen = rng.sample(speakers, n_spk)

    mix = np.zeros(CLIP_LEN, dtype=np.float32)
    for s in chosen:
        gain_db = rng.uniform(-3, 3)  # some people talk louder than others
        mix += random_speech_segment(utts[s], rng) * 10 ** (gain_db / 20)

    reverb = rng.random() < p_reverb
    if reverb and n_spk > 0:
        mix = apply_reverb(mix, rng)

    noise_kind = rng.choice(noise_kinds)
    if noise_kind == "babble":
        others = [s for s in speakers if s not in chosen]
        noise = babble_noise(utts, others, rng)
    else:
        noise = colored_noise(rng, noise_kind)

    snr_db = rng.uniform(*snr_range)
    if n_spk > 0:
        noise *= rms(mix) / 10 ** (snr_db / 20)
    else:
        noise *= 0.1  # noise-only clip: arbitrary reference level
        snr_db = float("-inf")
    out = mix + noise

    # same loudness for every clip so overall level carries no information about the label
    out = out / rms(out) * 0.05
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out / peak * 0.99

    return out.astype(np.float32), {
        "n_speakers": n_spk,
        "n_male": sum(sex[s] == "M" for s in chosen),
        "n_female": sum(sex[s] == "F" for s in chosen),
        "snr_db": round(snr_db, 2) if np.isfinite(snr_db) else "",
        "reverb": int(reverb and n_spk > 0),
        "noise_kind": noise_kind,
        "speaker_ids": " ".join(map(str, chosen)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech", default="data/raw/LibriSpeech")
    ap.add_argument("--subset", default="dev-clean", help="comma-separated LibriSpeech subsets to pool")
    ap.add_argument("--out", default="data/mixtures")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--p_reverb", type=float, default=0.5)
    ap.add_argument("--noise", default="babble,white,pink", help="comma-separated noise kinds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sex", default=None, choices=["M", "F"], help="restrict the speaker pool to one sex (single-sex dataset)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.librispeech)
    sex = load_speaker_sex(root)
    utts = {}
    for subset in args.subset.split(","):
        utts.update(index_utterances(root / subset))
    if args.sex:
        utts = {s: u for s, u in utts.items() if sex.get(s) == args.sex}
        print(f"speaker pool restricted to sex={args.sex}: {len(utts)} speakers")
    splits = split_speakers(sorted(utts), rng)
    noise_kinds = args.noise.split(",")

    out_root = Path(args.out)
    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    fields = ["file", "split", "n_speakers", "n_male", "n_female", "snr_db", "reverb", "noise_kind", "speaker_ids"]

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, n in counts.items():
            (out_root / split).mkdir(exist_ok=True)
            for i in range(n):
                audio, meta = make_mixture(
                    utts, splits[split], sex, rng, args.max_speakers,
                    (args.snr_min, args.snr_max), args.p_reverb, noise_kinds,
                )
                name = f"{split}/{split}_{i:05d}.wav"
                sf.write(out_root / name, audio, SR)
                writer.writerow({"file": name, "split": split, **meta})
                if (i + 1) % 500 == 0:
                    print(f"{split}: {i + 1}/{n}")

    print(f"wrote {sum(counts.values())} clips to {out_root}/ with labels.csv")
    print("speakers per split:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
