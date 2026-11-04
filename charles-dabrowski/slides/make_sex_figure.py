"""Male vs female voice, made legible: one clean voice of each sex, spectrogram 0-1 kHz with the harmonic spacing marked,
plus the spectrum at one instant where the harmonics appear as evenly spaced peaks. Writes slides/assets/male_vs_female_light.png"""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src"); from common import SR

ROOT = Path("."); LS = ROOT / "data/raw/LibriSpeech"
TEXT, MUTED, GRID, ORANGE, MAGENTA = "#1E1838", "#5E5878", "#DDD8EA", "#D96F1A", "#B82E5C"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.titlesize": 13, "axes.titleweight": "bold"})
N_FFT, HOP, FMAX, DUR = 2048, 160, 1000, 1.2
spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, win_length=1600, hop_length=HOP, power=2.0)
HZ = np.arange(N_FFT // 2 + 1) * SR / N_FFT

def sex_of():
    d = {}
    for ln in open(LS / "SPEAKERS.TXT"):
        if ln.startswith(";"): continue
        p = [x.strip() for x in ln.split("|")]; d[p[0]] = p[1]
    return d

def best_segment(sex, target_f0, n_speakers=12):
    """pick (audio segment, f0) with the most stable voicing and pitch closest to target"""
    sx = sex_of(); best = None
    for spk in sorted(p.name for p in (LS / "dev-clean").iterdir() if sx.get(p.name) == sex)[:n_speakers]:
        for fl in sorted((LS / "dev-clean" / spk).rglob("*.flac"))[:3]:
            a, sr = sf.read(fl, dtype="float32"); assert sr == SR
            f0 = torchaudio.functional.detect_pitch_frequency(torch.from_numpy(a), SR, frame_time=0.02, freq_low=60, freq_high=400).numpy()
            step = int(0.02 * SR); win = int(DUR / 0.02)
            for i in range(0, len(f0) - win, 5):
                seg = f0[i:i + win]; med = np.median(seg)
                voiced = np.mean(np.abs(seg - med) < 0.15 * med)        # fraction of frames near the median pitch
                energy = float(np.mean(a[i * step:(i + win) * step] ** 2))
                score = voiced - 0.004 * abs(med - target_f0) + (0 if energy > 1e-3 else -1)
                if best is None or score > best[0]:
                    best = (score, a[i * step:(i + win) * step], med, spk)
    return best[1], best[2], best[3]

fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.09, wspace=0.18, hspace=0.42)
for i, (sex, target, col) in enumerate([("M", 110, ORANGE), ("F", 210, MAGENTA)]):
    a, f0, spk = best_segment(sex, target); word = "male" if sex == "M" else "female"
    S = torch.log(spec(torch.from_numpy(a)) + 1e-6).numpy(); nb = int(FMAX / (SR / N_FFT)) + 1
    ax = axes[i, 0]
    ax.imshow(S[:nb], origin="lower", aspect="auto", cmap="magma", extent=[0, DUR, 0, FMAX], vmin=S[:nb].max() - 11, vmax=S[:nb].max())
    ax.set_title(f"one {word} voice, 0-1 kHz   (pitch {f0:.0f} Hz)", color=col, loc="left")
    ax.set_ylabel("Hz"); ax.set_xlabel("seconds")
    for k in range(1, int(FMAX / f0) + 1):                             # tick every harmonic on the right edge
        ax.plot([DUR - 0.03, DUR], [k * f0, k * f0], color="white", lw=1.5)
    # bracket between harmonic 1 and 2 with the spacing
    bracket = (f0, col)
    # spectrum at the loudest instant
    f0t = torchaudio.functional.detect_pitch_frequency(torch.from_numpy(a), SR, frame_time=0.01, freq_low=60, freq_high=400).numpy()
    en = S[:nb].sum(0)[:len(f0t)]; ok = en > np.percentile(en, 60)
    lo, hi = int(0.12 * SR / HOP), int((DUR - 0.12) * SR / HOP); ok[:lo] = False; ok[hi:] = False
    cand = np.where(ok)[0]; frame = int(cand[np.argmin(np.abs(f0t[cand] - f0))]); sl = S[:nb, frame]
    t_star = frame * HOP / SR
    axes[i, 0].axvline(t_star, color="white", lw=1.4, ls="--")
    ax = axes[i, 1]
    ax.fill_between(HZ[:nb], sl.min(), sl, color=col, alpha=0.25); ax.plot(HZ[:nb], sl, color=col, lw=1.8)
    for k in range(1, int(FMAX / f0) + 1):
        ax.axvline(k * f0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, FMAX); ax.set_yticks([]); ax.set_xlabel("Hz")
    ax.set_title(f"spectrum at the dashed line: peaks every {f0:.0f} Hz", color=col, loc="left", fontsize=12)
    ax0 = axes[i, 0]; x = t_star + 0.04 if t_star < DUR - 0.35 else t_star - 0.04
    ax0.annotate("", xy=(x, 2 * f0), xytext=(x, f0), arrowprops=dict(arrowstyle="<->", color="white", lw=1.8))
    ax0.text(x + (0.03 if t_star < DUR - 0.35 else -0.03), 1.5 * f0, f"gap = {f0:.0f} Hz", color="white", fontsize=12, fontweight="bold",
             va="center", ha="left" if t_star < DUR - 0.35 else "right", bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none", alpha=0.9))
    ax.text(0.98, 0.92, f"{int(FMAX / f0)} harmonics below 1 kHz", transform=ax.transAxes, ha="right", va="top", fontsize=12, color=TEXT)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    print(word, "speaker", spk, "f0", round(float(f0)))
fig.savefig("slides/assets/male_vs_female_light.png", dpi=170)
print("saved")
