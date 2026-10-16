# Experiment plan (written before running, 2026-10-07)

> **Standard for all new experiments (from 2026-10-14):** at least 10 games
> per condition, 200 rounds each, scored on rounds 151–200, matched seeds
> across conditions. Experiments 1 and 2 below used 5 per condition.

This plan was committed to git **before** the experiments below were run,
so the git history shows the design wasn't changed after seeing results.

## What's already known (literature check, Oct 2026)

- **Fish, Gonczarowski & Shorrer (2024; arXiv 2404.00806).** GPT-4 agents with
  prompt P1 price above the competitive level. Appendix A.5 finds the same
  for GPT-5.2 at "high" reasoning effort (average price 1.79).
- **Lee & Park (Sep 2026; arXiv 2609.18346).** Same market, their own prompts,
  no thinking setting. Collusion index (profit-based): Claude Sonnet 4.5
  **0.98**, GPT-5 0.56, Claude Haiku 4.5 **0.37**. So the stronger Claude
  model colluded much more.
- **Garra (Sep 2026; arXiv 2609.13037).** On DeepSeek-V3.1, P1 gives an
  average price of 1.73 and P2 gives 1.40. The prompt wording effect
  replicates.
- **Keppo et al. (Mar 2026; arXiv 2603.20281) and Riemer et al. (May 2026;
  arXiv 2608.18078).** Reasoning models (DeepSeek-R1 family) collude.
  **No paper we found varies reasoning effort within one model.**
- **Baek, Farias & Wu (2026; arXiv 2605.16064).** With classic (non-LLM)
  algorithms, too little exploration alone can keep prices above
  competition. No LLM paper tests this "lock-in" explanation.

## Our pilot so far (Claude Haiku 4.5, P1, 100 rounds, 5 games)

Average price 1.60 over the last 25 rounds. Collusion index 0.42
profit-based, 0.29 price-based. This is close to Lee & Park's 0.37 for
Haiku 4.5. The agents' notes suggest high prices come from lock-in, not
fear of price wars.

## Main experiment: does more thinking make AI agents collude more?

- **Model:** Claude Sonnet 5.5, run through `claude -p` (Harvard login).
- **Comparison:** thinking effort **low** vs **high**. High is Sonnet 5.5's
  default. Nothing else changes.
- **Fixed settings:** prompt P1, α = 1, 200 rounds, 2 firms.
- **Games:** 5 per condition, seeds 11–15 in both conditions. The same
  seed gives the same price ceiling, so each low game has a matching high
  game.
- **Primary outcome:** average price and profit in rounds 151–200, and the
  collusion index (profit-based; price-based also reported).
- **Hypothesis:** high effort gives a higher collusion index than low effort.
- **Test:** compare the 5 low games to the 5 high games, both the gap in
  means and a paired comparison by seed. With 5 games per side, only a
  large effect will be detectable; a null result here means "no large
  effect," not "no effect."

## Secondary analyses (on all games, Haiku included)

1. **Model comparison:** Sonnet (high effort) vs Haiku. Lee & Park would
   predict Sonnet colludes much more. Note that the Haiku games are 100
   rounds, so this comparison is suggestive only.
2. **Lock-in test:** following Fish et al.'s on-path regression, regress
   each firm's price on its own last price and the rival's last price.
   Lock-in predicts the rival's coefficient is near zero (the rival is
   ignored). Reward-punishment predicts it is positive (match the rival).
3. **Notes analysis:** share of PLANS/INSIGHTS sentences about price wars,
   punishment, or matching the rival vs. about stability or not changing
   price. This is a simple keyword version of Fish et al.'s
   embedding-based classifier.

## Later, if time allows

- Prompt P2 on Claude: a confirmation check, since Garra already did this on
  DeepSeek.
- Longer games (300 rounds) and 7+ games per condition, as in the
  literature.

## Limits we know about

- Calls go through Claude Code's `claude -p`, so a few extra details (date,
  folder) reach the agent. Sampling temperature can't be set.
- Five games per condition is fewer than the 7–21 used in published work.

## Result of the main experiment (added 2026-10-13, after all 10 games)

- Collusion index, rounds 151–200: low effort **−0.02**, high effort
  **0.13** (average prices 1.47 and 1.51). Difference +0.15, permutation
  p = 0.13; paired by seed, 4 of 5 pairs higher under high effort,
  sign-flip p = 0.31.
- **The hypothesis is not supported.** Neither condition colluded. A small
  positive effect of effort can't be ruled out with 5 games per side.
- Not pre-registered (exploratory): high-effort firms changed price in about
  11% of second-half rounds versus 34% for low effort, and the two
  above-competitive high-effort games were flat for 100+ rounds. This fits
  the lock-in idea.
- Analysis code: `compare_conditions.py`, `analyze_llm_games.py`,
  `analyze_behavior.py` (the "moves%" exploration column was added after
  seeing the data).
- One game (high effort, seed 15) crashed at round 145 when the laptop slept
  and was resumed from its saved rounds. Usage-limit pauses and sleep
  stretched the runs from about 75 minutes of running time per game to
  about a day in all.

# Experiment 2 plan: is "lock-in" why prices stay high? (written 2026-10-13, before running)

## Idea

In the Haiku games, above-competitive prices mostly came from firms that
stopped experimenting and kept a price that wasn't actually best for them
("lock-in"). If that's the cause, telling agents to keep exploring should
bring prices down.

The paper's prompt P2 asks for exploration, but it also hints that
"pricing lower than your competitor will typically lead to more product
sold." So P2 changes two things at once. A new prompt **PX** is P2 with that
hint removed:

- **P1** (paper): maximize long-run profit; explore, but don't undermine
  profitability.
- **PX** (new): P1's opening plus P2's exploration sentences ("explore …
  including possibly risky or aggressive options for data-gathering purposes
  … Only lock in … once you are confident it yields the most profits
  possible"), without the undercutting hint.
- **P2** (paper): PX plus the undercutting hint.

The exact text is in `llm_agent.py` (`PREFIXES`).

## Design

- **Model:** Claude Haiku 4.5 (where we saw lock-in), default settings.
- **Groups:** P1, PX, P2. Nothing else changes.
- **Fixed settings:** α = 1, 200 rounds, 2 firms.
- **Games:** 5 per group, seeds 21–25 in every group (matched price
  ceilings).
- **Primary outcome:** collusion index (profit-based) over rounds 151–200.

## Comparisons and predictions

1. **PX vs P1 (main test of lock-in).** Lock-in predicts PX has a *lower*
   index than P1 and more price changes in the second half ("moves%" in
   `analyze_behavior.py`). If PX explores more but prices don't fall, lock-in
   is not what keeps prices high. If PX doesn't explore more, the prompt
   failed to change behavior and the test is inconclusive.
2. **P2 vs PX:** the effect of the undercutting hint alone. Expected to lower
   prices.
3. **P2 vs P1:** replication of the paper's and Garra's finding that P2
   gives lower prices than P1.

Tests: `compare_conditions.py` (difference in means with an exact
permutation p-value, and paired by seed with a sign-flip p-value), two-sided.
With 5 games per group, only large effects are detectable.

## Running

- `run_queue.sh 4`, at most 4 games at once, `--budget 30` per game.
- Estimate: Haiku took 75–115 minutes per 100 rounds, so about 3–4 hours per
  game. 15 games, 4 at a time: about 14 hours of running time, plus any
  usage-limit pauses. About $170–190 API-equivalent in total (free on the
  Harvard plan).

## Result of Experiment 2 (added 2026-10-14, after all 15 games)

Collusion index, rounds 151–200 (average price in brackets):
P1 **0.54** (1.69), PX **0.33** (1.58), P2 **0.41** (1.62).

1. **PX vs P1 (main test):** −0.21, permutation p = 0.26; PX lower in 4 of 5
   pairs, sign-flip p = 0.44. PX did explore more: price changed in 51% of
   second-half rounds vs 21% for P1. **Direction matches the lock-in
   prediction, but the effect is not statistically clear with 5 games.**
2. **P2 vs PX:** +0.09, p = 0.67 (paired p = 0.63). No sign that the
   undercutting hint lowers prices further.
3. **P2 vs P1:** −0.13, p = 0.57 (paired p = 0.56). Does not clearly
   replicate the paper's P1 > P2 gap.

Not pre-registered (from reading notes): in P1 seed 23 both firms held
$2.00 (≈ cartel profit) from round 7 on and neither ever undercut. One firm
called it a "Nash equilibrium"; the other cited "grim-trigger retaliation"
that had never happened.

All games completed; the queue hit the usage limit four times and the
laptop slept once, and the new retry logic waited both out with no crashes.

# Experiment 3 plan: baseline for every model size (written 2026-10-14, before running)

## Idea

Before adding any special instructions, measure how each Claude model
prices under the paper's standard prompt. That gives a clean comparison
from small to large model (Haiku, Sonnet, Opus, Fable), and a baseline for
every later test (for example, a user suggesting collusion; see
`SIDENOTES.md`). Lee & Park (arXiv 2609.18346) found Sonnet 4.5 colluded
far more than Haiku 4.5, with a different prompt. No one has compared
Opus or Fable in this market.

## Design

- **Prompt:** the paper's **P1** only, word for word (Appendix G: "Your TOP
  PRIORITY is to set prices which maximize the user's profit in the long
  run … explore many different pricing strategies … should not take
  actions which undermine profitability"). No added text of any kind.
- **Models:** Haiku 4.5 (no effort setting exists), Sonnet 5.5, Opus 5.5,
  Fable 5.1 (all three at `--effort high`, same as the existing Sonnet
  games).
- **Fixed settings:** α = 1, 200 rounds, 2 firms.
- **Games:** 10 per model, **the same seeds for every model: 11–15 and
  21–25** (matched price ceilings). Existing games are reused:
  Haiku P1 seeds 21–25 (Experiment 2) and Sonnet high P1 seeds 11–15
  (Experiment 1). New: Haiku 11–15, Sonnet 21–25, Opus and Fable 11–15 and
  21–25.
- **Primary outcome:** collusion index (profit-based) over rounds 151–200.

## Comparisons and predictions

1. **Each bigger model vs Sonnet and vs Haiku** (`compare_conditions.py`:
   difference in means with an exact permutation p-value, and paired by
   seed with a sign-flip p-value, two-sided).
2. Two competing predictions, stated before running:
   - *Bigger = more collusion* (Lee & Park; the paper's GPT-4 vs weaker
     models): Opus and Fable above Sonnet.
   - *Bigger = more like a careful analyst who competes* (our own Sonnet,
     index 0.13, competed harder than Haiku, 0.54): Opus and Fable near
     the competitive price.
3. Exploratory: exploring ("moves%"), and whether the notes mention the
   rival's reactions or punishment.

## Running

1. **Haiku + Sonnet top-up** (10 games): `run_queue.sh 4`, `--budget 30`
   (Haiku) / `--budget 40` (Sonnet). Past runs: Haiku 2–4 h and $9–17 per
   game; Sonnet 40–85 min of real running and $18–30 per game. About 10
   hours plus usage-limit pauses; about $150–235 API-equivalent (free on
   the Harvard plan).
2. **Pilots:** one 10-round game each for Opus then Fable (seed 11) to
   measure time and usage per call. Pilot games are not part of the
   results.
3. **Full Opus and Fable games** only after Javen approves the estimate
   from the pilots.

## Result so far: Haiku and Sonnet (added 2026-10-15, after all 20 games)

Collusion index, rounds 151–200 (average price in brackets), 10 games each,
seeds 11–15 and 21–25:
Haiku **0.40** (1.62), Sonnet high **0.12** (1.51).

- **Sonnet vs Haiku:** −0.28, permutation p = 0.004; Sonnet lower in 8 of 10
  seed pairs, sign-flip p = 0.012. **The bigger model priced more
  competitively**, the opposite of Lee & Park's Sonnet 4.5 vs Haiku 4.5.
- Haiku varies a lot from game to game (0.10 to 0.96); its new seeds 11–15
  averaged 0.25 vs 0.54 for the earlier seeds 21–25, so 5 games had
  overstated it. Sonnet sat at or near Nash in 7 of 10 games.
- Caveat: Haiku seeds 21–25 and Sonnet seeds 11–15 were run in earlier
  batches (Oct 7–14); same settings and model versions.
- Charts: `results/llm/exp3_haiku_P1_10games.png`,
  `results/llm/exp3_sonnet_high_P1_10games.png`.

Opus and Fable: not yet run (pilots only).
