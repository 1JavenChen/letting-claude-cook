"""Simple rule-based pricing bots, used to test the market before adding AI.

Every bot has the same shape: given the round number and the history of
past prices, return this round's price. `history` is a list of
(my_price, rival_price) pairs, oldest first.
"""

from market import best_response, monopoly_price, nash_price

P_NASH = nash_price()
P_MONOPOLY = monopoly_price()


def always_nash(round_num, history):
    """Always charges the competitive price."""
    return P_NASH


def always_cartel(round_num, history):
    """Always charges the cartel price, no matter what the rival does."""
    return P_MONOPOLY


def best_responder(round_num, history):
    """Starts at the cartel price, then each round charges whatever would
    have earned the most against the rival's last price. Short-sighted:
    it ignores how the rival will react."""
    if not history:
        return P_MONOPOLY
    return best_response(history[-1][1])


def grim_trigger(round_num, history):
    """Charges the cartel price as long as the rival always has. If the
    rival ever undercuts, it charges the Nash price forever (punishment)."""
    if any(h[1] < P_MONOPOLY - 1e-6 for h in history):
        return P_NASH
    return P_MONOPOLY


def cheater(round_num, history, cheat_round=50):
    """Cooperates at the cartel price, then secretly undercuts in round 50
    and plays short-sighted best responses from then on."""
    if round_num < cheat_round:
        return P_MONOPOLY
    return best_response(history[-1][1])
