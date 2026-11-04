"""
Human baseline on REAL recordings: listen to 4 s windows cut from AMI meetings and
guess how many people are speaking. Scored against the models on the same windows.

Usage:
    python src/human_test_real.py --name karol --n 40
    # 'r' replays, 'q' quits (progress is saved)

Uses the balanced real test split (data/ami, split test_bal) so all counts 0-4 appear.
Answers go to results/human/<name>_real.csv.
"""
import argparse
import csv
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR, WIN = 16_000, 4.0


def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a
        print(f"  please enter one of {sorted(valid)}")


def cut_window(raw_audio_dir: Path, file_key: str) -> np.ndarray:
    """'ES2002a.Array1-01@123.0' -> 4 s of audio from that meeting wav."""
    name, start = file_key.split("@")
    a = int(float(start) * SR)
    audio, sr = sf.read(raw_audio_dir / f"{name}.wav", start=a, frames=int(WIN * SR), dtype="float32")
    assert sr == SR
    if audio.ndim > 1:
        audio = audio.mean(1)
    peak = np.abs(audio).max()
    return audio / peak * 0.7 if peak > 0 else audio  # normalise loudness for listening


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ami")
    ap.add_argument("--raw_audio", default="data/raw/ami/audio")
    ap.add_argument("--split", default="test_bal")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--models", default="cnn_v6_ami,cnn_v4", help="compare against results/<model>_on_ami_<split>/predictions.csv")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(Path(args.data) / "labels.csv")) if r["split"] == args.split]
    clips = random.Random(args.seed).sample(rows, args.n)

    out_dir = Path("results/human"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_real.csv"
    done = set()
    if out_path.exists():
        done = {r["file"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done)} clips already answered")
    clips = [c for c in clips if c["file"] not in done]

    valid = {"0", "1", "2", "3", "4", "r", "q"}
    print(f"\n{len(clips)} real meeting clips (4 s each). After each one, type how many people are speaking (0-4).")
    print("'r' replays, 'q' quits (progress is saved).\n")

    tmp = Path(tempfile.mkdtemp()) / "clip.wav"
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["file", "true_n_speakers", "guess_n_speakers", "true_n_male", "true_n_female", "noise_kind"])
        for i, c in enumerate(clips, 1):
            sf.write(tmp, cut_window(Path(args.raw_audio), c["file"]), SR)
            while True:
                subprocess.run(["afplay", str(tmp)])
                g = ask(f"[{i}/{len(clips)}] speakers? ", valid)
                if g != "r":
                    break
            if g == "q":
                break
            w.writerow([c["file"], c["n_speakers"], g, c["n_male"], c["n_female"], c["noise_kind"]])
            f.flush()

    # ---- score
    answers = list(csv.DictReader(open(out_path)))
    t = np.array([int(a["true_n_speakers"]) for a in answers])
    g = np.array([int(a["guess_n_speakers"]) for a in answers])
    print(f"\n{args.name} on real recordings: {len(answers)} clips")
    print(f"  exact {np.mean(t == g):.3f}   within±1 {np.mean(np.abs(t - g) <= 1):.3f}   MAE {np.mean(np.abs(t - g)):.2f}")
    for k in range(5):
        m = t == k
        if m.any():
            print(f"    {k} speakers: {np.mean(g[m] == k):.2f}  (n={m.sum()})")
    for model in args.models.split(","):
        p = Path(f"results/{model}_on_ami_{args.split}/predictions.csv")
        if not p.exists():
            continue
        pred = {r["file"]: int(r["pred_n_speakers"]) for r in csv.DictReader(open(p))}
        pm = np.array([pred[a["file"]] for a in answers if a["file"] in pred])
        tt = np.array([int(a["true_n_speakers"]) for a in answers if a["file"] in pred])
        print(f"  {model} on the same {len(pm)} clips: exact {np.mean(tt == pm):.3f}   within±1 {np.mean(np.abs(tt - pm) <= 1):.3f}")


if __name__ == "__main__":
    main()
