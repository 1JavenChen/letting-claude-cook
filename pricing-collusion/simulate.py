"""Run a repeated pricing game between two bots and record every round."""

import csv
from concurrent.futures import ThreadPoolExecutor

from market import demand, profits


def play(bot1, bot2, rounds=100, on_round=None):
    """Play `rounds` rounds. Each round both bots set a price at the same
    time, then both see the result. Returns one dict per round.

    Each bot's history is a list of (my price, rival price, my quantity,
    my profit). `on_round(log)` is called after every round, if given."""
    history1, history2 = [], []
    log = []
    # Ask both bots at once: AI bots take seconds per answer.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for r in range(1, rounds + 1):
            f1 = pool.submit(bot1, r, list(history1))
            f2 = pool.submit(bot2, r, list(history2))
            p1, p2 = f1.result(), f2.result()
            q1, q2 = demand([p1, p2])
            pi1, pi2 = profits([p1, p2])
            history1.append((p1, p2, q1, pi1))
            history2.append((p2, p1, q2, pi2))
            log.append({"round": r, "price1": p1, "price2": p2,
                        "quantity1": q1, "quantity2": q2,
                        "profit1": pi1, "profit2": pi2})
            if on_round:
                on_round(log)
    return log


def save_csv(log, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
