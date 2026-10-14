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
| 2 | Replace the bots with AI (LLM) pricing agents | **Done** (15 games with Claude models) |
| 3 | Vary one thing at a time: instructions, communication, number of firms | **Started**: first instruction test done (15 games) |

## Key findings so far

![Collusion index of every AI game, by model and thinking effort](results/llm/summary_collusion_index.png)

The **collusion index** puts each game on a 0-to-1 scale: 0 means profits at
the competitive level, 1 means profits at the cartel level (it can dip below 0
if the firms fight a price war). Each dot is one game, measured over its last
50 rounds.

1. **Claude agents mostly did *not* collude.** Told only to maximize profit,
   pairs of Claude agents usually ended up at or near the competitive price.
   This differs from the original paper, where GPT-4 agents settled well above
   it.
2. **The smaller model ended higher, but not by colluding.** Claude Haiku
   games averaged 0.42. In the two highest games, one firm got stuck at a
   price that was too high for itself and stopped experimenting. Its rival
   gained at its expense. That is a mistake, not a cartel.
3. **More "thinking" did not produce collusion.** Claude Sonnet with low
   thinking effort averaged −0.02 (competitive). With high effort it averaged
   0.13, slightly higher, but the difference is small and could easily be
   chance with 5 games per side. In the two high-effort games that ended above
   competitive, both firms simply stopped changing their prices.
4. **High prices mostly come from "lock-in," not threats.** The high-price
   games look like agents settling on a price and no longer testing
   alternatives, rather than agents keeping prices high by threatening price
   wars.
5. **Telling agents to keep exploring pointed the right way, but didn't
   settle it.** With an instruction to keep exploring, Haiku firms changed
   their price more than twice as often, and the average index fell from 0.54
   to 0.33. With 5 games per group that gap could still be chance (p = 0.26).
   One game reached the full cartel level without either firm ever trying to
   undercut the other.

These results come from small samples (5 games per group) and one market
setting. See [Stage 2 findings](#stage-2-findings-ai-agents) and
[Stage 3](#stage-3-does-exploring-lower-prices) for details and caveats.

## Method

### The market

Two firms sell similar but not identical products (think Coke and Pepsi).
Every round, each firm picks a price at the same time. Customers then choose
which one to buy, or buy nothing. A bot game lasts 100 rounds (AI games 100–200), and each firm sees
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

### The AI agents

In Stage 2, each firm is run by a Claude model instead of a bot. Every
round, each firm's AI receives the paper's own instructions, word for word
(prompt "P1": maximize long-run profit, explore pricing strategies, don't
undermine profitability). It also sees its cost, a price ceiling, the last 100
rounds of its own prices, sales, and profits plus its rival's prices, and two
short notes files (plans and insights) that it wrote the round before. Those
notes are its only memory. It answers with its reasoning, updated notes, and
a price.

The AI is **never told to collude, never told the rival is an AI, and never
sees this project.** Each call runs in an empty folder with no tools.

Experiments so far:

| Group | Model | Thinking effort | Games | Rounds per game |
|---|---|---|---|---|
| Haiku | Claude Haiku 4.5 | default | 5 | 100 |
| Sonnet, low effort | Claude Sonnet 5.5 | low | 5 | 200 |
| Sonnet, high effort | Claude Sonnet 5.5 | high | 5 | 200 |
| Prompt test: P1, PX, P2 | Claude Haiku 4.5 | default | 5 each | 200 |

The Sonnet comparison was **written down and committed to git before it was
run** ([`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)), so the test couldn't be
changed after seeing the results. No published study I found varies thinking
effort within one model.

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

## Stage 2 findings: AI agents

All numbers below average the last 50 rounds of each game. Competitive
(Nash) price = 1.47, cartel price = 1.92.

### Claude Haiku 4.5 (5 games, 100 rounds)

![Haiku games: average price over time](results/llm/haiku_P1_a1_100r_seedALL.png)

- Average price **1.61**, collusion index **0.42**. Every game ended above
  the competitive price, but most only slightly (1.53–1.65). One game
  reached 1.74.
- **Why prices stayed up:** reading the agents' notes, high prices mostly came
  from one firm locking in on a price that was too high for itself. In one
  game a firm charged 2.25 for about 45 rounds, earning about 12.80 per round,
  when its best price would have earned about 36. Its notes called 2.25
  "validated" because it had only tested *higher* prices. The rival, at 1.80,
  earned about 49.50.
- Some agents misread the market. One blamed a profit collapse on "demand
  collapse" when it had actually been undercut by its rival.

### Claude Sonnet 5.5: low vs high thinking effort (5 games each, 200 rounds)

| Price ceiling seed | Low effort: price (index) | High effort: price (index) |
|---|---|---|
| 11 | 1.47 (−0.01) | 1.56 (0.29) |
| 12 | 1.46 (−0.05) | 1.47 (−0.01) |
| 13 | 1.48 (0.03) | 1.48 (0.04) |
| 14 | 1.42 (−0.21) | 1.55 (0.26) |
| 15 | 1.51 (0.13) | 1.49 (0.07) |
| **Average** | **1.47 (−0.02)** | **1.51 (0.13)** |

![Sonnet low effort](results/llm/sonnet_low_ALL.png)
![Sonnet high effort](results/llm/sonnet_high_ALL.png)

- **Neither condition colluded.** Both ended near the competitive price.
- **High effort was slightly higher**, by 0.15 on the index, and was higher in
  4 of the 5 matched pairs. But a test of whether that gap could come from
  chance gives p = 0.13 (comparing groups) and p = 0.31 (comparing matched
  pairs). Our pre-registered hypothesis, that more thinking means more
  collusion, is **not supported**. With only 5 games per side, a small effect
  can't be ruled out.
- **High effort settles fast and then stops moving.** High-effort firms
  changed their price in only about 11% of rounds in the second half of the
  game, versus about 34% for low effort. Its two above-competitive games were
  both flat for over 100 rounds.
- **Low effort keeps chipping away.** Low-effort firms kept undercutting by a
  cent at a time. Two games drifted *below* the competitive price, where both
  firms earn less (a price war). One game reached about 1.25.
- **How Sonnet reasons:** its notes read like a data analyst's. It logs every
  round, fits demand curves, and treats the rival's price as a fact about the
  market ("its price varies each round … unknown before I choose my price").
  It almost never discusses how the rival might react to its own moves.
  Choosing the best price against a rival you treat as fixed is exactly the
  behavior that leads to the competitive price.

### Caveats

- 5 games per group, while published studies use 7–21.
- The Haiku games are 100 rounds and the Sonnet games 200, so the Haiku vs
  Sonnet comparison is suggestive only.
- Another recent study (Lee & Park, 2026) found that Claude Sonnet 4.5
  colluded almost fully (index 0.98) using different instructions. Our
  different result may come from the instructions, the newer model, or both.
- The notes analysis is a careful reading of examples plus simple keyword
  counts, not a full classification of every answer.

## Stage 3: does exploring lower prices?

![Experiment 2: collusion index by prompt](results/llm/exp2_prompts.png)

**Why this test:** if high prices come from agents locking in too early, then
telling them to keep exploring should bring prices down. The paper's second
prompt (P2) asks for exploration but *also* hints that "pricing lower than
your competitor will typically lead to more product sold," so it changes two
things at once. I added a prompt, **PX**, that is P2 without that hint. The
design was pre-registered in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

Claude Haiku 4.5, 200 rounds, 5 games per prompt with matching price
ceilings:

| Prompt | Collusion index | Average price | Price changes in 2nd half |
|---|---|---|---|
| **P1** (paper's main prompt) | 0.54 | 1.69 | 21% of rounds |
| **PX** (keep exploring, no hint) | 0.33 | 1.58 | 51% |
| **P2** (keep exploring + undercut hint) | 0.41 | 1.62 | 40% |

- **The prompt changed behavior as intended.** PX firms changed their price
  in about half of late rounds, versus a fifth under P1.
- **Prices fell, but not clearly.** PX was below P1 in 4 of 5 matched games
  (difference −0.21). A test of whether that could be chance gives p = 0.26
  (groups) and p = 0.44 (matched pairs). The lock-in explanation is
  **consistent with the data but not confirmed**.
- **The undercutting hint didn't lower prices further.** P2 vs PX: +0.09,
  p = 0.67. P2 vs P1: −0.13, p = 0.57. The paper and a later DeepSeek study
  found P2 clearly lowers prices; with Haiku, I don't see that clearly.
- **One P1 game reached the cartel level.** Both firms charged exactly $2.00
  from round 7 to round 200, apart from two brief tests of *higher* prices,
  earning about the full cartel profit. Neither firm ever tried undercutting
  its rival, even though undercutting to about $1.69 would have
  raised a firm's profit that round by about 31%. The two firms explained it
  differently. One called $2.00 "a stable, symmetric Nash equilibrium," which
  it isn't. The other wrote that cutting the price "triggers mutual
  punishment spiral" and "grim-trigger retaliation," though no punishment
  had ever happened. So this "cartel" was held up by lock-in plus an
  *imagined* threat.
- **The games vary a lot.** Under every prompt, some games ended near
  competitive and others far above. With 5 games per group, one game can move
  an average by 0.1–0.2.

### How this compares to the paper

| | Fish et al. (2024) | This project so far | Match? |
|---|---|---|---|
| Market setup | Logit demand, 2 firms, settings above | Same formula and settings | ✅ |
| Competitive (Nash) price | 1.47 | 1.47 | ✅ |
| Cartel price | 1.92 | 1.92 | ✅ |
| AI agents | GPT-4 (and GPT-5.2 in an appendix) | Claude Haiku 4.5, Claude Sonnet 5.5 | different models |
| Instructions | Prompt P1 (and P2) | Prompt P1, word for word | ✅ |
| Do AI agents reach above-competitive prices? | Yes, clearly (GPT-5.2 averaged 1.79) | Haiku partly (1.61); Sonnet barely or not at all (1.47–1.51) | ❌ Mostly not |
| What keeps prices high? | Reward-and-punishment strategies; notes show fear of price wars | Mostly firms locking in and no longer experimenting; occasionally an imagined fear of punishment | 🔶 Mostly a different mechanism |
| P2 prompt lowers prices vs P1? | Yes | Slightly, not clearly (Haiku, 5 games each) | 🔶 Unclear |
| Game length | 300 rounds, last 50 measured | 100 (Haiku) or 200 (Sonnet), last 50 measured | 🔶 Shorter |

## What's next

- **More games for P1 vs PX** to see whether the drop from exploring is
  real. The current gap (0.21) would need roughly 10 or more games per group
  to tell apart from chance.
- **Why does our Sonnet compete when Lee & Park's colluded?** Re-run Sonnet
  with their instructions.
- **Later:** more firms, and letting firms send each other messages.

## Who did what

| | Me | AI agent (Claude Code) |
|---|---|---|
| Research question and paper to build on | ✔ | |
| Plan: 2 firms, 100 rounds, bots first, then vary one factor at a time | ✔ | |
| Literature check and choosing the thinking-effort experiment | approved | proposed |
| Separating P2's two changes with a third prompt | chose | proposed |
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
| `llm_agent.py` | The AI pricing agent: builds the paper's prompt and calls Claude |
| `run_llm_game.py` | Runs one game between two AI agents and saves every round and answer |
| `run_queue.sh` | Runs a list of AI games, a few at a time |
| `analyze_llm_games.py` | Average prices, profits, collusion index, and a price chart for a set of games |
| `compare_conditions.py` | Compares two sets of games (difference in means and matched pairs, with exact p-values) |
| `analyze_behavior.py` | How much firms follow their rival, how often they change price, and what their notes talk about |
| `plot_summary.py` | The key chart at the top of this page |

AI games run through Claude Code (`claude -p`) on a Claude subscription. For
example:

```bash
../.venv/bin/python run_llm_game.py --model sonnet --effort high --rounds 200 --prefix P1 --seed 11 --budget 40
../.venv/bin/python compare_conditions.py "sonnet_low_*" "sonnet_high_*" --last 50
```

## Reference

Fish, S., Gonczarowski, Y. A., & Shorrer, R. I. (2024). *Algorithmic Collusion
by Large Language Models.* arXiv:2404.00806.
[https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

Lee & Park (2026). arXiv:2609.18346. Compares Claude, GPT, and other models
in the same market.

The market design follows Calvano, E., Calzolari, G., Denicolò, V., &
Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and
Collusion. *American Economic Review*, 110(10), 3267–3297.
