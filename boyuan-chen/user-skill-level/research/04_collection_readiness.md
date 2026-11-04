# Stage 5 — Collection Readiness (blocked on credentials)

Everything for data collection is built and verified; the only missing piece is
`GROQ_API_KEY`, which exists in the user's terminal but not in the automated sessions.

## To collect (user action, one of)

**Option A — run it yourself** (resumable; rerun the same command after any interruption):

```bash
cd /Users/boyuanchen/user-model/.claude/worktrees/education-level-genai-experiment-27e674
export GROQ_API_KEY=...   # if not already set in this terminal
.venv/bin/python experiment/run_experiment.py --suite skill
```

**Option B — let the session loop run it**: add `export GROQ_API_KEY=...` to
`~/.bash_profile`; the local 10-minute loop will detect it and start collection
automatically.

## What the run is

648 requests (24 tasks × 9 conditions × 3 reps) on `openai/gpt-oss-20b` via the Groq
free tier, shuffled order (seed 20261014), checkpointed to
`experiment/out/responses_skill.jsonl`. Expected duration 1–4 h under free-tier 429
pacing. Optional pilot first: same command with `--reps 1` (216 requests).

## After collection (automated; the loop or data-collector agent handles it)

```bash
.venv/bin/python experiment/score.py        # -> experiment/out/scores_skill.csv
```

Then stage 5's integrity checks and `research/04_collection_log.md` per
`.claude/agents/data-collector.md`.
