"""Phase 1 baseline: hand-written rules mapping audio features to blendshapes.

This is intentionally simple -- loudness opens the jaw, low-frequency-heavy
sounds (oo/oh) round the lips, bright sounds (ee) stretch them, hiss (s/sh/f)
narrows the opening. It's the yardstick the Phase 2 neural model has to beat.

Usage:
    python lipsync/rule_based.py data/test.wav out/test.json
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blendshapes import FPS, MOUTH_SHAPES, save_animation  # noqa: E402

SR = 16000


def load_mono(path):
    sr, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g)
    return audio


def frame_features(audio):
    hop = SR // FPS
    win = int(0.05 * SR)
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    low = freqs < 900
    mid = (freqs >= 900) & (freqs < 3500)
    high = freqs >= 3500

    n_frames = int(np.ceil(len(audio) / hop))
    padded = np.pad(audio, (win // 2, win))
    feats = np.zeros((n_frames, 4), np.float32)  # rms_db, low, mid, high ratios
    for i in range(n_frames):
        seg = padded[i * hop: i * hop + win] * window
        spec = np.abs(np.fft.rfft(seg)) ** 2
        total = spec.sum() + 1e-10
        rms = np.sqrt(np.mean(seg ** 2))
        feats[i] = (20 * np.log10(rms + 1e-6), spec[low].sum() / total,
                    spec[mid].sum() / total, spec[high].sum() / total)
    return feats


def smooth(x, attack=0.6, release=0.25):
    """Asymmetric one-pole smoothing: mouths open fast, close a bit slower."""
    out = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        k = attack if v > prev else release
        prev = prev + k * (v - prev)
        out[i] = prev
    return out


def rules(feats):
    db, low, mid, high = feats.T
    # Map loudness to 0..1 relative to this clip's speech (ignore near-silence,
    # which would otherwise drag the floor down and make everything look loud).
    speech = db[db > db.max() - 50]
    floor, ceil = np.percentile(speech, 15), np.percentile(speech, 97)
    loud = np.clip((db - floor) / (ceil - floor + 1e-6), 0, 1) ** 1.3
    voiced = db > floor - 6
    hiss = np.clip((high - 0.3) / 0.4, 0, 1) * voiced
    rounded = np.clip((low - 0.85) / 0.12, 0, 1) * (1 - hiss)
    bright = np.clip((mid - 0.12) / 0.3, 0, 1) * (1 - hiss)

    w = {name: np.zeros(len(feats), np.float32) for name in MOUTH_SHAPES}
    w["jawOpen"] = smooth(0.75 * loud * (1 - 0.7 * hiss))
    w["mouthFunnel"] = smooth(0.6 * rounded * loud)
    w["mouthPucker"] = smooth(0.5 * rounded * (1 - loud))
    for side in ("Left", "Right"):
        w["mouthStretch" + side] = smooth(0.5 * bright * loud + 0.3 * hiss)
        w["mouthSmile" + side] = smooth(0.15 * bright)
        w["mouthUpperUp" + side] = smooth(0.4 * hiss)
        w["mouthLowerDown" + side] = smooth(0.35 * hiss + 0.2 * loud * bright)
        # A short silence between voiced frames is likely a plosive (p/b/m): press.
        gap = (~voiced) & np.roll(voiced, 2) & np.roll(voiced, -2)
        w["mouthPress" + side] = smooth(0.6 * gap.astype(np.float32), 0.8, 0.4)
    return np.clip(np.stack([w[n] for n in MOUTH_SHAPES], axis=1), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("out")
    args = p.parse_args()
    frames = rules(frame_features(load_mono(args.audio)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_animation(args.out, frames)
    print(f"Wrote {len(frames)} frames ({len(frames) / FPS:.1f}s) to {args.out}")


if __name__ == "__main__":
    main()
