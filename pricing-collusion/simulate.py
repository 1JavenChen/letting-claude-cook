"""Run a repeated pricing game between two bots and record every round."""

import csv

from market import demand, profits


def play(bot1, bot2, rounds=100):
    """Play `rounds` rounds. Each round both bots set a price at the same
    time, then both see the result. Returns one dict per round."""
    history1, history2 = [], []   # each bot's view: (my price, rival price)
    log = []
    for r in range(1, rounds + 1):
        p1 = bot1(r, history1)
        p2 = bot2(r, history2)
        q1, q2 = demand([p1, p2])
        pi1, pi2 = profits([p1, p2])
        history1.append((p1, p2))
        history2.append((p2, p1))
        log.append({"round": r, "price1": p1, "price2": p2,
                    "quantity1": q1, "quantity2": q2,
                    "profit1": pi1, "profit2": pi2})
    return log


def save_csv(log, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
