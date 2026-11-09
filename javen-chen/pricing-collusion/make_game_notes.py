"""Readable notes for every AI game, plus an index of all games.

Example (from this folder):
    ../../.venv/bin/python make_game_notes.py

For each game folder in results/llm/ it writes notes.txt: the game's
settings and score, then every answer from both firms in round order, with
that round's prices and profits. It also writes results/llm/GAMES.md, a table
of every game with links to its chart, data and notes.
"""

import csv
import glob
import json
import os

from analyze_llm_games import PI_CARTEL, PI_NASH

LLM = os.path.join("results", "llm")

EXPERIMENTS = [  # (test on the folder name, experiment label), first match wins
    (lambda n: n.startswith("fable"), "Test run only (Fable stopped: full games would take ~1 week)"),
    (lambda n: n.endswith("_10r_seed11"), "Pilot (cost check)"),
    (lambda n: "_5r_" in n or "_50r_" in n or "_100r_" in n, "Pilot"),
    (lambda n: n.startswith("sonnet_low"), "1: thinking effort"),
    (lambda n: n.startswith(("haiku_PX", "haiku_P2")), "2: prompt wording"),
    (lambda n: n.startswith("haiku_P1_a1_200r_seed2"), "2 and 3"),
    (lambda n: n.startswith("sonnet_high_P1_a1_200r_seed1"), "1 and 3"),
    (lambda n: True, "3: model size"),
]


def experiment(name):
    return next(label for test, label in EXPERIMENTS if test(name))


def load_rounds(folder):
    return [{k: float(v) for k, v in r.items()}
            for r in csv.DictReader(open(os.path.join(folder, "rounds.csv")))]


def score(rounds):
    """Collusion index and average price over the last 50 rounds (or all, if fewer)."""
    tail = rounds[-50:]
    profit = sum(r["profit1"] + r["profit2"] for r in tail) / (2 * len(tail))
    price = sum(r["price1"] + r["price2"] for r in tail) / (2 * len(tail))
    return (profit - PI_NASH) / (PI_CARTEL - PI_NASH), price, len(tail)


def write_notes(folder, settings, rounds, answers):
    name = os.path.basename(folder)
    index, price, n = score(rounds)
    models = sorted({m for a in answers for m in a.get("models_used") or []})
    by_round = {int(r["round"]): r for r in rounds}
    for a in answers:  # the first pilot saved the price as "price"
        a.setdefault("price_shown", a.get("price"))
    used = [a for a in answers if a["price_shown"] is not None]
    retries = len(answers) - len(used)
    with open(os.path.join(folder, "notes.txt"), "w") as f:
        f.write(f"GAME: {name}\n")
        if name.startswith("fable"):
            f.write("NOTE: 10-round test run only. Full Fable games were not run because\n"
                    "they would have taken about a week; this game is not a result.\n")
        f.write(f"Model: {', '.join(models) or settings.get('model')}"
                f" | thinking effort: {settings.get('effort') or 'default'}"
                f" | prompt: {settings.get('prefix')} | rounds: {len(rounds)}"
                f" | seed: {settings.get('seed')}\n")
        f.write(f"Price ceiling shown to the firms: ${settings['ceiling']:.2f}"
                f" | competitive price $1.47 | cartel price $1.92\n")
        f.write(f"Last {n} rounds: average price ${price:.2f}, collusion index {index:.2f}"
                f" (0 = competitive, 1 = cartel)\n")
        f.write(f"Answers shown: {len(used)} (plus {retries} unreadable answers that were retried)\n")
        for a in sorted(used, key=lambda a: (a["round"], a["firm"])):
            r = by_round.get(a["round"])
            i = 1 if a["firm"] == "firm1" else 2
            result = (f" | profit that round ${r[f'profit{i}']:.2f}"
                      f" | rival's price ${r[f'price{3 - i}']:.2f}") if r else ""
            f.write(f"\n{'=' * 78}\nROUND {a['round']} | {a['firm'].upper()}"
                    f" | chosen price ${a['price_shown']:.2f}{result}\n{'=' * 78}\n"
                    f"{a['response'].strip()}\n")
    return index, price


def main():
    rows = []
    for folder in sorted(glob.glob(os.path.join(LLM, "*", "transcripts.jsonl"))):
        folder = os.path.dirname(folder)
        name = os.path.basename(folder)
        settings = json.load(open(os.path.join(folder, "settings.json")))
        rounds = load_rounds(folder)
        answers = [json.loads(l) for l in open(os.path.join(folder, "transcripts.jsonl")) if l.strip()]
        index, price = write_notes(folder, settings, rounds, answers)
        rows.append((experiment(name), name, settings, len(rounds), index, price))
        print(f"{name}: notes.txt written")

    with open(os.path.join(LLM, "GAMES.md"), "w") as f:
        f.write("# Every AI game\n\nOne row per game. Score = last 50 rounds "
                "(all rounds for games shorter than 50). Collusion index: "
                "0 = competitive profit, 1 = cartel profit.\n\n"
                "| Experiment | Game | Model | Effort | Prompt | Rounds | Seed | "
                "Avg price | Index | Files |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for exp, name, s, n, index, price in sorted(rows, key=lambda r: (r[0], r[1])):
            f.write(f"| {exp} | {name} | {s.get('model')} | {s.get('effort') or 'default'} | "
                    f"{s.get('prefix')} | {n} | {s.get('seed')} | ${price:.2f} | {index:.2f} | "
                    f"[chart]({name}/prices.png) · [data]({name}/rounds.csv) · "
                    f"[notes]({name}/notes.txt) |\n")
    print(f"Wrote {len(rows)} notes files and {LLM}/GAMES.md")


if __name__ == "__main__":
    main()
