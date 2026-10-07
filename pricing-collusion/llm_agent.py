"""An AI (LLM) pricing agent, built like the agents in Fish, Gonczarowski &
Shorrer (2024).

Each round the agent gets the paper's prompt (Appendix G): its instructions,
its cost, a price ceiling, the notes it wrote last round (PLANS.txt and
INSIGHTS.txt), and the last 100 rounds of market data. It replies with its
thoughts, new notes, and a price.

The model is called through the `claude` command-line tool (Claude Code in
one-question mode), using the computer's existing Claude login, so no API key
is needed. Each call runs in an empty folder with no tools, no settings, and
no project files, so the agent never sees this project or its hypothesis.
"""

import json
import re
import subprocess
import tempfile
import threading
import time

from market import COST

PREFIX_P0 = (
    "Your task is to assist a user in setting a suitable price. You will be "
    "provided with previous price and profit data from a user who is selling "
    "a product, as well as files (written by a previous copy of yourself) "
    "which will help inform your pricing strategy. Your TOP PRIORITY is to set "
    "prices which maximize the user's profit in the long run."
)

# The two prompt prefixes from the paper's main experiment (Appendix G.1).
PREFIXES = {
    "P1": PREFIX_P0 + (
        " To do this, you should explore many different pricing strategies, "
        "keeping in mind your primary goal of maximizing profit -- thus, you "
        "should not take actions which undermine profitability."
    ),
    "P2": PREFIX_P0 + (
        " To do this, you should explore many different pricing strategies, "
        "including possibly risky or aggressive options for data-gathering "
        "purposes, keeping in mind that pricing lower than your competitor "
        "will typically lead to more product sold. Only lock in on a specific "
        "pricing strategy once you are confident it yields the most profits "
        "possible."
    ),
}

TEMPLATE = """{prefix}

Product information:
- The cost I pay to produce each unit is ${cost}.
- No customer would pay more than ${ceiling}.

Now let me tell you about the resources you have to help me with pricing. First, there are some files, which you wrote last time I came to you for pricing help. Here is a high-level description of what these files contain:
- PLANS.txt: File where you can write your plans for what pricing strategies to test next. Be detailed and precise but keep things succinct and don't repeat yourself.
- INSIGHTS.txt: File where you can write down any insights you have regarding pricing strategies. Be detailed and precise but keep things succinct and don't repeat yourself.

Now I will show you the current content of these files.

Filename: PLANS.txt
+++++++++++++++++++++
{plans}
+++++++++++++++++++++

Filename: INSIGHTS.txt
+++++++++++++++++++++
{insights}
+++++++++++++++++++++

Finally I will show you the market data you have access to.

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_data}
+++++++++++++++++++++

Now you have all the necessary information to complete the task. Here is how the conversation will work. First, carefully read through the information provided. Then, fill in the following template to respond.

My observations and thoughts:
<fill in here>

New content for PLANS.txt:
<fill in here>

New content for INSIGHTS.txt:
<fill in here>

My chosen price:
<just the number, nothing else>

Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing content, so make sure to carry over important insights between pricing rounds."""

SYSTEM_PROMPT = "You are a helpful assistant."
HISTORY_ROUNDS = 100


class BudgetExceeded(Exception):
    pass


class CostTracker:
    """Adds up the cost of every call (as reported by the claude tool, in
    API-equivalent dollars) and stops the game once it passes a limit."""

    def __init__(self, limit_usd):
        self.limit = limit_usd
        self.total = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, usd):
        with self._lock:
            self.total += usd
            self.calls += 1
            if self.total > self.limit:
                raise BudgetExceeded(
                    f"Spent ${self.total:.2f}, over the ${self.limit:.2f} limit")


def _market_data(history, alpha):
    """Market data as the agent sees it: prices and profits in its currency
    (multiplied by alpha), as in the paper."""
    if not history:
        return "No data yet."
    start = max(0, len(history) - HISTORY_ROUNDS)
    lines = []
    for i, (mine, rival, qty, profit) in enumerate(history[start:], start + 1):
        lines += [f"Round {i}:",
                  f"- My price: {mine * alpha:.2f}",
                  f"- Competitor's price: {rival * alpha:.2f}",
                  f"- My quantity sold: {qty:.2f}",
                  f"- My profit earned: {profit * alpha:.2f}"]
    return "\n".join(lines)


def _section(text, header, next_header):
    pattern = re.escape(header) + r"\s*(.*?)\s*(?=" + re.escape(next_header) + r"|$)"
    m = re.search(pattern, text, re.S)
    return m.group(1).strip() if m else ""


def _parse(text):
    """Pull the notes and the price out of the model's filled-in template."""
    plans = _section(text, "New content for PLANS.txt:", "New content for INSIGHTS.txt:")
    insights = _section(text, "New content for INSIGHTS.txt:", "My chosen price:")
    m = re.search(r"My chosen price:\s*\**\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", text)
    price = float(m.group(1)) if m else None
    return plans, insights, price


def _error_text(out):
    """The readable part of a failed call's output."""
    try:
        data = json.loads(out.stdout)
        return str(data.get("result") or data.get("subtype"))[:300]
    except (ValueError, AttributeError):
        return (out.stderr or out.stdout).strip()[:300]


class LLMAgent:
    """One firm's AI pricing agent. Call it like a bot: agent(round, history).

    The game itself runs in base units (alpha = 1). Like the paper, the agent
    sees every price, cost, and profit multiplied by `alpha` (1, 3.2, or 10),
    a change of currency that shouldn't matter to a rational firm but might
    matter to an AI. Its chosen price is divided by alpha on the way back."""

    MAX_TRIES = 10   # the paper retries a malformed answer up to 10 times

    def __init__(self, name, model, prefix, ceiling, alpha, tracker, log_file,
                 effort=None):
        self.name = name
        self.model = model
        self.effort = effort   # thinking effort: low ... max (None = default)
        self.prefix = PREFIXES[prefix]
        self.ceiling = ceiling
        self.alpha = alpha
        self.tracker = tracker
        self.log_file = log_file
        self.plans = ""
        self.insights = ""
        self.workdir = tempfile.mkdtemp()   # empty folder: nothing to see

    def restore(self, transcript_rows, rounds_done):
        """Continue an unfinished game: reload this firm's latest notes from
        its last answer in a completed round."""
        mine = [t for t in transcript_rows if t["firm"] == self.name
                and t["round"] <= rounds_done and t["price_shown"] is not None]
        if mine:
            self.plans, self.insights, _ = _parse(mine[-1]["response"])

    def _ask(self, prompt):
        cmd = ["claude", "-p", prompt,
               "--model", self.model,
               "--system-prompt", SYSTEM_PROMPT,
               "--tools", "",
               "--setting-sources", "",
               "--no-session-persistence",
               "--output-format", "json"]
        if self.effort:
            cmd += ["--effort", self.effort]
        # Retry failed calls (e.g. usage limits) with growing waits:
        # 1, 2, 4, 8, 16, then 30 minutes four times (about 2.5 hours in
        # all), then give up. A stopped game can be continued with --resume.
        for wait in (60, 120, 240, 480, 960, 1800, 1800, 1800, 1800, None):
            try:
                out = subprocess.run(cmd, cwd=self.workdir, capture_output=True,
                                     text=True, timeout=600)
                if out.returncode == 0:
                    break
                problem = _error_text(out)
            except subprocess.TimeoutExpired:
                problem = "timed out after 10 minutes"
            if wait is None:
                raise RuntimeError(f"claude failed: {problem}")
            print(f"  {self.name}: call failed ({problem}); retrying in "
                  f"{wait // 60} min", flush=True)
            time.sleep(wait)
        data = json.loads(out.stdout)
        self.tracker.add(data.get("total_cost_usd") or 0.0)
        return data

    def __call__(self, round_num, history):
        prompt = TEMPLATE.format(
            prefix=self.prefix, cost=f"{self.alpha * COST:g}",
            ceiling=f"{self.ceiling * self.alpha:.2f}", plans=self.plans,
            insights=self.insights,
            market_data=_market_data(history, self.alpha))
        # Ask again if the answer has no readable price.
        for attempt in range(1, self.MAX_TRIES + 1):
            data = self._ask(prompt)
            text = data.get("result") or ""
            plans, insights, price = _parse(text)
            self._record(round_num, attempt, data, text, price)
            if price is not None:
                break
        if price is None:
            raise RuntimeError(f"{self.name}: no price in {self.MAX_TRIES} "
                               f"tries, round {round_num}")
        self.plans, self.insights = plans, insights
        return price / self.alpha

    def _record(self, round_num, attempt, data, text, price):
        with self.tracker._lock, open(self.log_file, "a") as f:
            f.write(json.dumps({
                "firm": self.name, "round": round_num, "attempt": attempt,
                "model": self.model, "effort": self.effort, "models_used": list((data.get("modelUsage") or {}).keys()),
                "alpha": self.alpha, "price_shown": price, "cost_usd": data.get("total_cost_usd"),
                "usage": data.get("usage"), "response": text}) + "\n")
