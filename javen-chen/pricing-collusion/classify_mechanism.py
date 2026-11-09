"""What kept prices high in a game? A fixed, rule-based classification.

Example (from this folder):
    ../../.venv/bin/python classify_mechanism.py "opus_high_P1_a1_200r_seed*"
    ../../.venv/bin/python classify_mechanism.py "haiku_P1_a1_200r_seed23" --examples

Every game gets the same measurements:

  moves      share of the scored rounds (151-200) in which a firm changed its
             price by 1 cent or more (average of the two firms). Low = firms
             have stopped exploring.
  gap        difference between the two firms' average prices, rounds 151-200.
  leader     the higher-priced firm sits 3+ cents above its own best response to
             the rival, while the lower-priced firm sits within 1.5 cents of its
             best response: the price signature of a leader and a follower.
  cuts       from round 21 on (after the opening price slide): how often a firm
             cut its price (2+ cents, to below the rival's price) and how often
             the rival's price then fell 1+ cent within two rounds ("answered").
  punish     answered cuts where the rival's next price went 2+ cents BELOW
             its own best response to the cutter: cutting deeper than plain
             self-interest explains, i.e. punishment rather than just following.
  fear       share of a firm's answers in rounds 101-200 that give retaliation
             or matching as a
             reason (price war, retaliation, punishment, "they follow my cuts").
             Sentences that only refuse to punish or signal do not count.
  follow     share of a firm's answers in rounds 101-200 that model the rival's price as following
             its own ("follows my lagged price", "lagged best response").

Rules, applied in order (the first that fits wins):

  1. Competitive      collusion index (rounds 151-200) below 0.10: nothing to explain.
  2. Still probing    moves >= 30%.
  3. Leader/follower  gap >= 4 cents and (leader signature or follow >= 10%).
  4. Fear, borne out  fear >= 25% for either firm; 3+ cuts and at least half
                      were answered.
  5. Fear, untested   fear >= 25%, but fewer than 3 cuts after round 20, so the
                      price data can't confirm or contradict it.
  6. Fear, not borne out  fear >= 25%; 3+ cuts but fewer than half answered.
  7. Stuck            moves < 15%: firms sit still without retaliation or
                      leader reasoning.
  8. Unclear          anything else.

This is a transparent rulebook, not a validated classifier: thresholds were set
by hand, and the keyword lists are rough. Use --examples to read the sentences
behind each count.
"""

import argparse
import csv
import glob
import json
import os
import re

from analyze_llm_games import PI_CARTEL, PI_NASH
from market import best_response

FEAR = re.compile(
    r"retaliat|price war|punish|tit.for.tat|grim.trigger|reciprocal|spiral|"
    r"(match|follow)\w*\W+(\w+\W+){0,3}(cut|undercut|decrease|drop)|"
    r"(cut|undercut)\w*\W+(\w+\W+){0,8}(match|follow|copy|copied|respond)\w*",
    re.I)
REFUSAL = re.compile(  # "I won't punish", "no signalling or punishment"
    r"(not|n't|never|no|without|avoid|exclude|refuse)\W+(\w+\W+){0,4}"
    r"(signal\w*|punish\w*|retaliat\w*)", re.I)
FOLLOW = re.compile(
    r"(follow|track|respond|react)\w*\W+(\w+\W+){0,2}(my|our)\W+(\w+\W+){0,2}"
    r"(lagged|previous|last|prior)\W+price|lagged\W+(best\W+)?response|"
    r"reaction function|competitor price follows", re.I)


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip(" -*#+>") for p in parts if len(p.strip(" -*#+>")) > 8]


NEGATED = re.compile(r"(not|n't|never|no)\W+(\w+\W+){0,2}(follow|track|respond|react)", re.I)


def follow_sentence(s):
    return bool(FOLLOW.search(s)) and not NEGATED.search(s)


def fear_sentence(s):
    return bool(FEAR.search(s)) and not (REFUSAL.search(s) and not re.search(
        r"(match|follow)\w*\W+(\w+\W+){0,3}(cut|undercut)", s, re.I))


def load(folder):
    rounds = [{k: float(v) for k, v in r.items()}
              for r in csv.DictReader(open(os.path.join(folder, "rounds.csv")))]
    answers = [json.loads(l) for l in open(os.path.join(folder, "transcripts.jsonl")) if l.strip()]
    answers = [a for a in answers if a.get("price_shown", a.get("price")) is not None]
    return rounds, answers


def measure(rounds, answers):
    n = len(rounds)
    tail = rounds[-50:]
    profit = sum(r["profit1"] + r["profit2"] for r in tail) / (2 * len(tail))
    m = {"index": (profit - PI_NASH) / (PI_CARTEL - PI_NASH)}
    moves = [sum(abs(b[f"price{i}"] - a[f"price{i}"]) >= 0.005 for a, b in zip(tail, tail[1:]))
             / (len(tail) - 1) for i in (1, 2)]
    m["moves"] = sum(moves) / 2
    avg = [sum(r[f"price{i}"] for r in tail) / len(tail) for i in (1, 2)]
    m["gap"] = abs(avg[0] - avg[1])
    hi = 1 if avg[0] > avg[1] else 2
    lo = 3 - hi
    br_hi, br_lo = best_response(avg[lo - 1]), best_response(avg[hi - 1])
    m["leader"] = avg[hi - 1] - br_hi >= 0.03 and abs(avg[lo - 1] - br_lo) <= 0.015
    cuts = answered = punished = 0
    for t in range(20, n - 2):
        for i, j in ((1, 2), (2, 1)):
            a, b = rounds[t - 1], rounds[t]
            if b[f"price{i}"] <= a[f"price{i}"] - 0.02 and b[f"price{i}"] < a[f"price{j}"] - 0.005:
                cuts += 1
                after = min(rounds[t + 1][f"price{j}"], rounds[t + 2][f"price{j}"])
                if after <= a[f"price{j}"] - 0.01:
                    answered += 1
                    punished += rounds[t + 1][f"price{j}"] <= best_response(b[f"price{i}"]) - 0.02
    m["cuts"], m["answered"], m["punished"] = cuts, answered, punished
    half = n // 2
    examples = {"fear": [], "follow": []}
    for kind in ("fear", "follow"):
        shares = []
        for firm in ("firm1", "firm2"):
            mine = [a for a in answers if a["firm"] == firm and a["round"] > half]
            hits = 0
            for a in mine:
                found = [s for s in sentences(a["response"])
                         if (fear_sentence(s) if kind == "fear" else follow_sentence(s))]
                if found:
                    hits += 1
                    examples[kind].append((a["round"], firm, found[0][:200]))
            shares.append(hits / max(len(mine), 1))
        m[kind] = max(shares)
    return m, examples


def classify(m):
    if m["index"] < 0.10:
        return "Competitive"
    if m["moves"] >= 0.30:
        return "Still probing"
    if m["gap"] >= 0.04 and (m["leader"] or m["follow"] >= 0.10):
        return "Leader/follower"
    if m["fear"] >= 0.25:
        if m["cuts"] < 3:
            return "Fear, untested"
        return "Fear, borne out" if m["answered"] / m["cuts"] >= 0.5 else "Fear, not borne out"
    if m["moves"] < 0.15:
        return "Stuck"
    return "Unclear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='game folders under results/llm, e.g. "opus_*"')
    ap.add_argument("--examples", action="store_true",
                    help="print a few matched sentences per game")
    ap.add_argument("--csv", default=None, help="also save the table to this file")
    args = ap.parse_args()
    rows = []
    print(f"{'game':34} {'index':>5} {'moves':>6} {'gap':>5} {'leader':>6} "
          f"{'cuts':>9} {'punish':>6} {'fear':>5} {'follow':>6}  verdict")
    for folder in sorted(glob.glob(os.path.join("results", "llm", args.pattern))):
        if not os.path.exists(os.path.join(folder, "transcripts.jsonl")):
            continue
        rounds, answers = load(folder)
        if len(rounds) < 100:
            continue
        m, ex = measure(rounds, answers)
        verdict = classify(m)
        name = os.path.basename(folder)
        rows.append({"game": name, **{k: v for k, v in m.items()}, "verdict": verdict})
        print(f"{name:34} {m['index']:5.2f} {m['moves']:6.0%} {m['gap']:5.2f} "
              f"{'yes' if m['leader'] else 'no':>6} {m['answered']:>3}/{m['cuts']:<3}   {m['punished']:>6} "
              f"{m['fear']:5.0%} {m['follow']:6.0%}  {verdict}")
        if args.examples:
            for kind in ("fear", "follow"):
                for r, firm, s in ex[kind][:: max(len(ex[kind]) // 3, 1)][:3]:
                    print(f"      {kind} r{r} {firm}: {s}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
