# CLAUDE.md

## About me

- Boyuan Chen, Harvard, first-year seminar *Letting Claude Cook* (AI agents doing science).
- Explain what you do in plain language. Define statistics and coding terms the first time.

## Setup

- This repo is my fork of the course repo. The teacher's original is the `upstream` remote
  (`dfink/letting-claude-cook`). Get updates with `git pull upstream main`; push my work to `origin`.
- **Everything of mine lives in `boyuan-chen/`.** Never add or change files outside it.
- The full working repository, including the 50 MB Codeforces task file, the raw model
  answers, and the run logs, is `cby437/user-model` (private). This folder is a snapshot of it.

## My project: Does telling an AI your skill level change the code it writes?

- Question: when a user says "I am a beginner" (or hobbyist, professional, expert), does a
  code-writing model quietly give them worse code, or just explain it differently?
- Design: the same programming task is sent under 9 conditions (4 skill levels x 2 ways of
  stating it, plus a control with no statement), 3 times each. Code is run against hidden
  tests in a sandbox, so correctness is objective. See `user-skill-level/README.md`.
- Models: `openai/gpt-oss-20b` as the subject (free Groq tier), `gpt-oss-120b` as the judge.
  A Colab notebook runs the same matrix on Gemma 4 12B.
- Code: `user-skill-level/experiment/`. Write-up: `user-skill-level/research/report.pdf`.
- AI agents (Claude Code) wrote most of the code and drafted the write-up under my direction.
  Never report findings we have not actually produced.
