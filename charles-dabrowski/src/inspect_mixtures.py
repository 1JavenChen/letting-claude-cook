"""
Sanity-check a mixtures folder: label distribution + a spectrogram grid.

Usage:
    python src/inspect_mixtures.py data/demo
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram


def main(folder: str):
    root = Path(folder)
    rows = list(csv.DictReader(open(root / "labels.csv")))
    print(f"{len(rows)} clips")
    print("n_speakers:", dict(sorted(Counter(r["n_speakers"] for r in rows).items())))
    print("noise_kind:", dict(Counter(r["noise_kind"] for r in rows)))
    print("reverb:    ", dict(Counter(r["reverb"] for r in rows)))
    snrs = [float(r["snr_db"]) for r in rows if r["snr_db"]]
    if snrs:
        print(f"snr_db:     {min(snrs):.1f} .. {max(snrs):.1f} (mean {np.mean(snrs):.1f})")

    # one example per speaker count
    examples = {}
    for r in rows:
        examples.setdefault(int(r["n_speakers"]), r)
    examples = [examples[k] for k in sorted(examples)]

    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 3.5), squeeze=False)
    for ax, r in zip(axes[0], examples):
        audio, sr = sf.read(root / r["file"])
        f, t, S = spectrogram(audio, sr, nperseg=512, noverlap=384)
        ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-10), shading="gouraud", cmap="magma", vmin=-110, vmax=-40)
        ax.set_ylim(0, 8)
        ax.set_title(f"{r['n_speakers']} spk ({r['n_male']}M/{r['n_female']}F)  snr={r['snr_db'] or 'n/a'}  {r['noise_kind']}", fontsize=9)
        ax.set_xlabel("time (s)")
    axes[0][0].set_ylabel("kHz")
    plt.tight_layout()
    out = root / "examples.png"
    plt.savefig(out, dpi=120)
    print("saved", out)
    print("example files:", [r["file"] for r in examples])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/demo")
