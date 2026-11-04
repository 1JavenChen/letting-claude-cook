"""
Human baseline: listen to test clips and guess the number of speakers.

Plays clips with macOS `afplay`, records your answers, and scores you on the
same clips the model was scored on (so results are directly comparable).

Usage:
    python src/human_test.py --data data/mixtures_v2 --n 40 --name karol
    # type 'r' to replay a clip, 'q' to quit early

Results are appended to results/human/<name>.csv; a summary prints at the end.
Pass --preds results/cnn_v2/predictions.csv to see the model's score on the same clips.
"""

import argparse
import csv
import random
import subprocess
from pathlib import Path

import numpy as np


def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a
        print(f"  please enter one of {sorted(valid)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mixtures_v2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--name", required=True, help="listener id, e.g. your first name")
    ap.add_argument("--preds", default=None, help="model predictions.csv to compare against")
    ap.add_argument("--gender", action="store_true", help="also ask for #male / #female")
    ap.add_argument("--seed", type=int, default=123, help="same seed = same clips for every listener")
    args = ap.parse_args()

    root = Path(args.data)
    rows = [r for r in csv.DictReader(open(root / "labels.csv")) if r["split"] == "test"]
    rng = random.Random(args.seed)
    clips = rng.sample(rows, args.n)

    out_dir = Path("results/human")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.csv"
    done = set()
    if out_path.exists():
        done = {r["file"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done)} clips already answered")
    clips = [c for c in clips if c["file"] not in done]

    valid = {"0", "1", "2", "3", "4", "r", "q"}
    print(f"\n{len(clips)} clips. After each one, type how many people are speaking (0-4).")
    print("'r' replays, 'q' quits (progress is saved).\n")

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["file", "true_n_speakers", "guess_n_speakers", "true_n_male", "guess_n_male",
                        "true_n_female", "guess_n_female", "snr_db", "noise_kind", "reverb"])
        for i, c in enumerate(clips, 1):
            path = root / c["file"]
            while True:
                subprocess.run(["afplay", str(path)])
                g = ask(f"[{i}/{len(clips)}] speakers? ", valid)
                if g != "r":
                    break
            if g == "q":
                break
            gm = gf = ""
            if args.gender and g != "0":
                gm = ask("   how many male? ", {"0", "1", "2", "3", "4"})
                gf = ask("   how many female? ", {"0", "1", "2", "3", "4"})
            w.writerow([c["file"], c["n_speakers"], g, c["n_male"], gm, c["n_female"], gf,
                        c["snr_db"], c["noise_kind"], c["reverb"]])
            f.flush()

    # ---- score
    answers = list(csv.DictReader(open(out_path)))
    t = np.array([int(a["true_n_speakers"]) for a in answers])
    g = np.array([int(a["guess_n_speakers"]) for a in answers])
    print(f"\n{args.name}: {len(answers)} clips")
    print(f"  count accuracy {np.mean(t == g):.3f}   MAE {np.mean(np.abs(t - g)):.3f}   within±1 {np.mean(np.abs(t - g) <= 1):.3f}")
    for k in range(5):
        m = t == k
        if m.any():
            print(f"    {k} speakers: {np.mean(g[m] == k):.2f}  (n={m.sum()})")

    if args.preds:
        model = {r["file"]: int(r["pred_n_speakers"]) for r in csv.DictReader(open(args.preds))}
        p = np.array([model[a["file"]] for a in answers if a["file"] in model])
        tt = np.array([int(a["true_n_speakers"]) for a in answers if a["file"] in model])
        print(f"  model on the same {len(p)} clips: accuracy {np.mean(tt == p):.3f}   MAE {np.mean(np.abs(tt - p)):.3f}")


if __name__ == "__main__":
    main()
