# Research Team: Skill Level → Claude's Code Output

The experiment pivots from *education level → general answers* (see `EXPERIMENT.md`, kept
as the v1 study) to **how a user's stated programming skill level affects the code an
LLM (Claude) produces**. The study is run by a team of seven agents defined in
`.claude/agents/`, one per stage of the standard research procedure, chained through
numbered artifacts in this directory.

## The team

| # | Stage (research procedure) | Agent | Reads | Writes |
|---|---|---|---|---|
| 1 | Review the literature | `literature-reviewer` | the web | `01_literature_review.md` |
| 2–3 | Research questions / hypotheses + research design | `research-designer` | 01, `EXPERIMENT.md` | `02_hypotheses_and_design.md` |
| 4 | Methods, logistics, ethics | `methods-engineer` | 02, `experiment/` | `03_methods_plan.md`, `experiment/tasks.json`, `experiment/conditions_skill.json`, runner + scoring harness |
| 5 | Collect data | `data-collector` | 03 | `experiment/out/responses_skill.jsonl`, `04_collection_log.md` |
| 6 | Analyze data | `data-analyst` | 02, 03, data | `experiment/out/metrics.csv`, `05_analysis.md` |
| 7 | Interpret findings | `findings-interpreter` | 01, 02, 05 | `06_findings.md` |
| 8 | Write and cite the report | `report-writer` | 01–06 | `07_report.md` |

## How to run a stage

Ask Claude Code to run the stage's agent, e.g. "run the literature-reviewer agent" or
"have the research-designer produce the design". Stages are sequential — each agent
reads its predecessors' artifacts — but stage 1 and the task-battery drafting half of
stage 4 can run in parallel once stage 2–3 has fixed the task spec.

**Human decision points** (don't skip): approve `02_hypotheses_and_design.md` before
methods engineering, and approve `03_methods_plan.md` (including backend/cost) before
collection.

## Standing constraints

- **Subject model:** the study targets Claude, which needs Anthropic API credits
  (`--backend anthropic`). Without credits, the pipeline runs on the Groq free tier
  (`openai/gpt-oss-20b`) — fine for piloting the pipeline, but the headline claims are
  then about that model, not Claude. Decide before stage 5.
- **Pipeline lessons from the v1 study** (`EXPERIMENT.md` §10): resolve model IDs
  against the account's live catalog, checkpoint every response, retry errors on
  rerun, abort on systemic failure, keep task text byte-identical across conditions.
- **Ethics:** synthetic personas, no human subjects — no IRB needed; keep the equity
  framing responsible (report magnitudes, not just significance).
