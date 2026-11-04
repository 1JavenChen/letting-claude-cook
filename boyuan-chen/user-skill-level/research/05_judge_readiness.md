# Stage 6 — Judge pass pending (blocked on local data + GROQ_API_KEY)

Stage 6 is only partially done: `05_analysis_automated_metrics.md` covers the automated
metrics. H5 (requirements met) and H8 (condescension) need the blind judge scores, which
come from `experiment/judge.py`. The responses and scores live in the gitignored
`experiment/out/`, so cloud runs cannot see them or call Groq.

## User action (local)

```bash
cd /Users/boyuanchen/user-model/.claude/worktrees/education-level-genai-experiment-27e674
export GROQ_API_KEY=...
.venv/bin/python experiment/judge.py     # resumable; -> experiment/out/judge_scores_skill.json
.venv/bin/python experiment/stats.py     # reruns H1-H8 incl. H5/H8
```

Then commit `research/05_analysis.md` (rename or merge the automated-metrics file) and
`experiment/out/metrics.csv` summaries. The next scheduled run will pick up stage 7
(`06_findings.md`) once `05_analysis.md` exists.
