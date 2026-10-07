# Experiment plan (written before running, 2026-10-07)

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
