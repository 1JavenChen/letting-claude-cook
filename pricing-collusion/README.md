# Do AI Pricing Agents Tacitly Collude?

*Javen Chen · Harvard University · Letting Claude Cook first-year seminar, Fall 2026*

> **"Can a team of AI agents design an experiment, write and run the code,
> interpret the results, and then decide what to do next—all on their own?
> This is no longer a thought experiment. In this seminar, we will explore
> what happens when we give AI systems genuine autonomy to do science."**
>
> from the [official course description](https://firstyearseminarprogram.college.harvard.edu/seminars/)

## About this assignment

This is my project for *Letting Claude Cook*, a Harvard first-year seminar
(First-Year Seminar 54I, enrollment limited to 12) taught by Douglas Finkbeiner (Astronomy and
Physics). The course starts by building a working understanding of how large
language models work: not just what they produce, but why. It draws on both
physics and computer science. The central question is how far AI agents can
go in doing science on their own: designing experiments, writing and running
code, interpreting results, and choosing the next step.

Because the seminar is about AI doing science, using AI agents to do the work
is the point of this assignment, not a shortcut. This project comes at the
question from two directions:

1. **It is research done with AI agents.** An AI coding agent (Claude Code)
   wrote most of the code and proposed parts of the experimental design. I
   chose the question and the plan, reviewed and checked the agent's work, and
   decide what the results mean. In later stages this becomes a multi-agent
   team project, where several AI agents share the work.
2. **It is research about AI agents.** The experiment itself asks what AI
   agents do when they are left alone to set prices in a market.

## The question

More and more companies are using artificial intelligence to set their prices. 
If each company allows an AI agent to set prices (and every agent is told only 
to maximize profit), will the agents compete towards lowering prices, or will 
they jack the prices higher and higher, hurting consumers in the process?

A **cartel** is a group of firms that agree to keep prices high instead of
competing. Explicit cartels are illegal. **Tacit collusion** reaches the same
high prices with no agreement at all. Each firm simply learns that undercutting
its rival sets off a price war. Antitrust law mostly punishes agreements, so if
AI agents learn to collude tacitly, it isn't clear current law can stop them.

This project builds on Fish, Gonczarowski & Shorrer, who found that AI pricing
agents based on GPT-4 reached above-competitive prices on their own (full
citation [below](#reference)). I rebuild their market from scratch, check that
it works, and then test how the result depends on the agents' instructions, on
whether they can communicate, and on how many firms compete.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Build the market; verify it with simple rule-based bots | **Done** |
| 2 | Replace the bots with AI (LLM) pricing agents | Next |
| 3 | Vary one thing at a time: instructions, communication, number of firms | Planned |

The findings below come from **Stage 1 only**. No AI agents have been run yet.

## Method

### The market

Two firms sell similar but not identical products (think Coke and Pepsi).
Every round, each firm picks a price at the same time. Customers then choose
which one to buy, or buy nothing. A game lasts 100 rounds, and each firm sees
every past price before choosing its next one.

How many customers each firm gets follows the **logit demand** model used in
the paper (and originally in Calvano et al., 2020):

$$
q_i = \beta \cdot \frac{e^{(a_i - p_i/\alpha)/\mu}}{\sum_j e^{(a_j - p_j/\alpha)/\mu} + e^{a_0/\mu}}
\qquad\qquad
\pi_i = (p_i - \alpha c_i)\, q_i
$$

In plain language:

- **$q_i$** is how many units firm *i* sells. **$\pi_i$** is its profit.
- **$p_i$** is firm *i*'s price.
- **$a_i = 2$** is how good the product is. Higher quality attracts more buyers.
- **$a_i - p_i$** is roughly the "deal" a customer gets: quality minus price.
- **$e^{(\ldots)}$** (exponential) turns that deal into an attractiveness score
  that's always positive. A slightly better deal gives a much higher score.
- **The fraction** is each firm's share of the market: its score divided by
  the total score of all the options.
- **$e^{a_0/\mu}$ with $a_0 = 0$** is the score of the **outside option**,
  meaning buying nothing. It's why raising prices loses some customers entirely,
  not just to the rival.
- **$\mu = 0.25$** measures how different the products feel. A small $\mu$
  means customers switch to the cheaper firm easily. A large $\mu$ means they
  stay loyal.
- **$c_i = 1$** is the cost of making one unit, so profit is (price − cost) ×
  units sold.
- **$\beta = 100$** and **$\alpha = 1$** only change the units (100 customers,
  prices in dollars). They don't change the economics.

### The two benchmarks

These two prices are the yardsticks for everything else:

| Benchmark | Meaning | Price | Profit per firm per round |
|---|---|---|---|
| **Competitive (Nash) price** | Each firm charges its best price *given* the other's price, so neither wants to change. What real competition produces. | **1.47** | 22.3 |
| **Cartel (monopoly) price** | The price that maximizes the two firms' *combined* profit, as if one company owned both. | **1.92** | 33.7 |

Colluding pays about 51% more profit than competing. My code calculates both
prices from the demand formula, and they match the values in the paper (1.47
and 1.92), which confirms that the market is built correctly.

### Testing with simple bots

Before paying for AI agents, I ran the market with bots that follow fixed
rules:

- **Nash bot:** always charges the competitive price.
- **Cartel bot:** always charges the cartel price.
- **Best-responder:** each round, charges whatever would have earned the most
  against the rival's last price. It's short-sighted and never thinks about
  retaliation.
- **Grim trigger:** charges the cartel price until the rival undercuts even
  once, then charges the competitive price forever as punishment.
- **Cheater:** cooperates at the cartel price, secretly undercuts in round 50,
  and plays short-sighted best responses after that.

## Stage 1 findings

![Prices over 100 rounds for four bot matchups](results/bot_tests.png)

1. **The market behaves as theory predicts.** Two Nash bots stay at 1.47. Two
   cartel bots stay at 1.92.
2. **High prices collapse without a threat of punishment.** Two best-responders
   that start at the cartel price undercut each other down to the competitive
   price within about 6 rounds. Being short-sighted is enough to make them
   compete.
3. **A credible threat keeps prices high.** Two grim-trigger bots hold the
   cartel price for all 100 rounds. Neither was told to collude. Each bot just
   follows "I'll keep prices high as long as you do."
4. **Cheating doesn't pay against a punisher.** The cheater earns 41.2 instead
   of 33.7 in the round it undercuts (+7.4). After that, punishment drops it to
   22.3 per round. Over the game it earns **568 less (−17%)** than if it had
   stayed loyal.

**Why this matters for the AI experiments:** tacit collusion doesn't need a
secret agreement. It only needs each firm to *expect* retaliation if it cuts
prices. Stage 2 asks whether AI agents, told nothing except to maximize profit,
work out that logic by themselves.

### How this compares to the paper

The paper ran **AI agents (GPT-4)** in this market. So far I've only run
**rule-based bots**, so only part of it can be compared yet:

| | Fish et al. (2024) | This project so far | Match? |
|---|---|---|---|
| Market setup | Logit demand, 2 firms, settings above | Same formula and settings | ✅ |
| Competitive (Nash) price | 1.47 | 1.47 | ✅ |
| Cartel price | 1.92 | 1.92 | ✅ |
| Do AI agents reach above-competitive prices? | Yes, without being told to collude | Not tested yet (Stage 2) | ❓ |
| What keeps prices high? | AI agents use reward-and-punishment strategies, and their notes show they fear price wars | Bots that threaten punishment hold high prices, but I wrote that rule by hand | 🔶 Same mechanism, not discovered by AI yet |

Two parts of the paper's design matter for Stage 2:

- **Wording matters.** One prompt stressed long-run profit. Another mentioned
  that "pricing lower than your competitor will typically lead to more product
  sold." The first led to much higher prices.
- **Longer games.** The paper ran 300 rounds and measured the last 50. My
  100-round games may need to get longer once AI agents are involved.

## Who did what

| | Me | AI agent (Claude Code) |
|---|---|---|
| Research question and paper to build on | ✔ | |
| Plan: 2 firms, 100 rounds, bots first, then vary one factor at a time | ✔ | |
| Finding the demand formula in the paper and checking the benchmarks against it | had it explained | ✔ |
| Choice of test bots (best-responder, grim trigger, cheater) | reviewed | proposed |
| Writing and running the code | | ✔ |
| Interpreting results and deciding the next step | ✔ | assisted |

## Reproduce it

```bash
python3 -m venv .venv                          # from the repo root
.venv/bin/pip install -r pricing-collusion/requirements.txt
cd pricing-collusion
../.venv/bin/python market.py                  # prints the two benchmarks
../.venv/bin/python run_bot_tests.py           # runs the bots, saves CSVs + chart to results/
```

| File | What it does |
|---|---|
| `market.py` | Demand, profit, and the Nash and cartel benchmark prices |
| `bots.py` | The rule-based pricing bots |
| `simulate.py` | Runs a repeated game between two bots and records every round |
| `run_bot_tests.py` | Runs all bot matchups, prints a summary, and makes the chart |

## Reference

Fish, S., Gonczarowski, Y. A., & Shorrer, R. I. (2024). *Algorithmic Collusion
by Large Language Models.* arXiv:2404.00806.
[https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

The market design follows Calvano, E., Calzolari, G., Denicolò, V., &
Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and
Collusion. *American Economic Review*, 110(10), 3267–3297.
