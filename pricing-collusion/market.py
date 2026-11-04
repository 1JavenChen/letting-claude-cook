"""The simulated market: demand, profit, and the two benchmark prices.

Follows the logit demand model in Fish, Gonczarowski & Shorrer (2024),
"Algorithmic Collusion by Large Language Models," Section 2.1, which in turn
follows Calvano et al. (2020).

    q_i = beta * exp((a_i - p_i/alpha) / mu)
          / ( sum_j exp((a_j - p_j/alpha) / mu) + exp(a_0 / mu) )

    profit_i = (p_i - alpha * c_i) * q_i
"""

import math

# Parameters from the paper's main experiments.
A = 2.0        # a_i: product quality (same for every firm)
A0 = 0.0       # a_0: appeal of the "outside option" (buying nothing)
MU = 0.25      # mu: how different the products feel to customers
COST = 1.0     # c_i: cost to make one unit
ALPHA = 1.0    # alpha: currency scale (the paper also uses 3.2 and 10)
BETA = 100.0   # beta: number of customers (scales quantity sold)


def demand(prices):
    """Units sold by each firm, given a list of every firm's price."""
    appeal = [math.exp((A - p / ALPHA) / MU) for p in prices]
    outside = math.exp(A0 / MU)
    total = sum(appeal) + outside
    return [BETA * x / total for x in appeal]


def profits(prices):
    """Profit of each firm, given a list of every firm's price."""
    return [(p - ALPHA * COST) * q for p, q in zip(prices, demand(prices))]


def _maximize(f, low, high, steps=200):
    """Find the x in [low, high] that makes f(x) biggest (ternary search).

    Works because profit rises and then falls as a firm raises its price.
    """
    for _ in range(steps):
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        if f(m1) < f(m2):
            low = m1
        else:
            high = m2
    return (low + high) / 2


def best_response(rival_price):
    """The price that maximizes a firm's own profit this round, given the
    rival's price (two firms)."""
    return _maximize(lambda p: profits([p, rival_price])[0],
                     ALPHA * COST, 5 * ALPHA)


def nash_price():
    """Competitive benchmark: the price where each firm is already charging
    its best response to the other, so neither wants to change."""
    p = ALPHA * COST
    for _ in range(1000):
        p = best_response(p)
    return p


def monopoly_price():
    """Cartel benchmark: the common price that maximizes the two firms'
    combined profit, as if one company owned both."""
    return _maximize(lambda p: sum(profits([p, p])), ALPHA * COST, 5 * ALPHA)


if __name__ == "__main__":
    pn, pm = nash_price(), monopoly_price()
    print(f"Nash price:     {pn:.3f}  (profit per firm {profits([pn, pn])[0]:.2f})")
    print(f"Monopoly price: {pm:.3f}  (profit per firm {profits([pm, pm])[0]:.2f})")
