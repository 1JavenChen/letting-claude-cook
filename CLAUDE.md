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
- **Open question:** running AI agents inside the simulation needs API access. Check with the teacher how the class handles API keys/credits.

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
