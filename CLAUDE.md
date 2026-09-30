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
- **Open question:** running AI agents inside the simulation needs API access. Check with the teacher how the class handles API keys/credits.

## Progress log

(Update this section after each work session with what we did and what we found.)
