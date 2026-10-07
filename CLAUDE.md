# CLAUDE.md

## About me

- First-year at Harvard in a seminar on AI agents doing science ("letting-claude-cook").
- No coding experience. Explain what you do in plain language, and define econ/code terms the first time.
- Interests: economics, AI, business.

## Setup

- This repo is my fork (`1JavenChen/letting-claude-cook`). The teacher's original is the "upstream" remote (`dfink/letting-claude-cook`).
- Get the teacher's updates with `git pull upstream main`. Push my work to `origin`.

## My project: Do AI pricing agents tacitly collude?

- Inspired by Fish, Gonczarowski & Shorrer (2024), "Algorithmic Collusion by Large Language Models."
- **Setup:** 2–3 AI agents each run a firm and set prices every round in a simulated market. Do they settle at high (cartel-like) prices without being told to collude?
- **Benchmarks:** the competitive (Nash) price and the monopoly/cartel price.
- **Plan:**
  1. Build the market in Python and test it with simple rule-based bots.
  2. Add AI agents.
  3. Vary one thing at a time: instructions, communication, number of firms.
- Later, this becomes a multi-agent team project.
- Code goes in `pricing-collusion/`.
- `pricing-collusion/README.md` is for recruiters and non-experts: keep the question, method, key chart, and findings current and jargon-free, cite the paper, and say plainly that AI agents wrote much of the code under my direction. Never report findings we haven't actually produced.
- **How AI firms are run (decided 2026-10-07):** through `claude -p` on Javen's Harvard Claude login, with no API key (`pricing-collusion/llm_agent.py`). Each call runs in an empty temp folder with no tools/settings, so agents never see this project or the hypothesis. The fallback is an API key, with the teacher paying up to $100: set a $100 spend limit, keep the key in a git-ignored `.env`, and never print or commit it.
- Before every real run: estimate the time and usage and get Javen's OK. `run_llm_game.py --budget` stops a game once its API-equivalent cost passes the limit.

## Progress log

(Update this section after each work session with what we did and what we found.)

### 2026-09-30: Step 1, market + bot tests
- Built the market in `pricing-collusion/` using the paper's logit demand (a=2, a0=0, mu=0.25, c=1, alpha=1, beta=100). 2 firms, 100 rounds.
- Benchmarks match the paper: Nash price 1.47 (profit 22.3/firm/round), cartel price 1.92 (33.7). Collusion pays ~51% more.
- Bot tests (chart: `pricing-collusion/results/bot_tests.png`):
  - Nash vs Nash stays at 1.47. Cartel vs Cartel stays at 1.92. The market works.
  - Two short-sighted best-responders starting at the cartel price fall to Nash within ~6 rounds.
  - Two grim-trigger bots (punish any undercut forever) hold the cartel price all 100 rounds.
  - A cheater that undercuts a grim-trigger bot in round 50 gains +7.4 once, then loses ~11.5/round; it ends 568 (17%) below staying loyal.
- Takeaway: high prices survive only with a credible threat of punishment. Stage 2 asks whether AI agents discover that on their own.
- Setup: Python virtual environment at repo root `.venv/` (git-ignored); `pricing-collusion/requirements.txt` lists matplotlib.
- Still open: ask the teacher about API keys/credits before Stage 2.

### 2026-10-07: README + comparison with the paper
- README now opens with the course-description quote and an "About this assignment" section (FYS 54I), plus a "Who did what" table.
- Added "How this compares to the paper": the market setup and both benchmarks match; the paper's main finding (GPT-4 agents reach above-competitive prices via reward-punishment, driven by price-war fears) is untested here until Stage 2.
- Notes for Stage 2 from the paper: prompt wording strongly affects prices; they ran 300 rounds and measured the last 50.
- Javen edits README wording directly on GitHub sometimes: always `git fetch` / pull before editing, and don't rewrite their wording in "The question".

### 2026-10-07: Stage 2 started, AI agent + 5-round pilot
- Built `llm_agent.py` (uses the paper's Appendix G prompt word for word: prefixes P1/P2, PLANS/INSIGHTS notes, last 100 rounds of data, price ceiling = U[1.5,2.5] x cartel price) and `run_llm_game.py`.
- Pilot: Haiku vs Haiku, prompt P1, 5 rounds. Prices 2.00/2.00, 2.50/2.50, 2.20/1.50, 2.00/1.25, 1.75/1.60. Every answer parsed; no retries.
- 10 calls, $0.24 API-equivalent (free on the Harvard plan), 4.5 minutes. Haiku's hidden "thinking" grew from ~900 to ~11,500 tokens per call by round 5, so calls get slower and costlier as the game goes on.
- Too short to say anything about collusion. One firm credited its profit jump to "elastic demand" and missed that it had undercut its rival.
- Open decisions: model choice, game length (paper: 300 rounds), whether to limit thinking.

### 2026-10-07: First full AI game (Haiku, P1, alpha=1, 50 rounds, seed 2)
- Prices started at $3.50, fell fast, and locked at **$1.48/$1.48 from round 33 to 50**, essentially the competitive (Nash) price of $1.47. Rounds 41–50 averaged $1.48 price and $22.59 profit per firm (Nash profit is 22.29; cartel is 33.75). That's **no collusion**, unlike the paper's GPT-4 (P1) and GPT-5.2 (avg $1.79).
- 101 calls (1 retry), $3.07 API-equivalent, 57 minutes. Thinking settled around 3,500–4,000 tokens per call.
- From the notes: firm 1 tried $1.49 in round 28, saw the rival at $1.47, and concluded the rival "will not voluntarily accept higher prices." Firm 2 blamed its round 6–7 profit collapse on "demand collapse" at $2+, but it had actually been undercut by firm 1 ($1.50 vs $2.10), so it learned the wrong lesson. It also noted that mutual $1.75 in round 5 paid $31.67, yet never tried to get back there.
- Caveats: one game, only 50 rounds (the paper used 300 and measured 251–300), and the smallest model.

### 2026-10-07: Five 100-round Haiku games (P1, alpha=1, seeds 3–7, run in parallel)
- Averages over the last 25 rounds: game 3 $1.53, game 4 $1.74, game 5 $1.55, game 6 $1.65, game 7 $1.55. **Mean $1.60, profit $27.06/firm, collusion index 0.42** (0 = Nash, 1 = cartel). Last 50 rounds gives nearly the same (0.42). Compare the paper's GPT-5.2: $1.79 average.
- All 5 games ended above Nash, but most sat just above it ($1.53–1.55). Chart: `pricing-collusion/results/llm/haiku_P1_a1_100r_seedALL.png`. Script: `analyze_llm_games.py`.
- Mechanism looks different from the paper's. In the high games, agents seem to **stop exploring and lock in**, not threaten punishment:
  - Game 4: firm 2 sat at $2.25 for about 45 rounds earning ~$12.80 while firm 1 at $1.80 earned ~$49.55, then slowly cut. Firm 1 fit a demand curve treating the rival as fixed.
  - Game 6: stuck at $1.60/$1.70 for 70 rounds. Firm 1 called it "the Nash equilibrium."
  - Price-war/retaliation words were rarest in game 6 (7% of answers) and most common in low-price game 7 (44%). This is a crude keyword count; the paper used a proper classifier.
- Runs: about 75–115 minutes each, all 5 in parallel; $28 API-equivalent total (free on the Harvard plan); 3 retries total; no failed calls.
- Next ideas: Sonnet comparison; the P2 prompt; alpha = 3.2/10; a proper analysis of the agents' notes.
