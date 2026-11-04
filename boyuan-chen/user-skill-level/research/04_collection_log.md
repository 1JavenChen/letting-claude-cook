# Stage 5 — Collection Log

## Provenance

- **Dataset:** `experiment/out/responses_skill.jsonl` (local, gitignored), 648 records.
- **Subject model:** `openai/gpt-oss-20b` for all 648 responses (verified in records),
  via Groq free developer tier; temperature 0.7, max_tokens 2400,
  `reasoning_effort: low`; shuffled execution order, seed 20261014.
- **Collection window:** 2026-10-14 → 2026-10-16 (paced by free-tier quota).
- **Token usage:** 115,281 input / 415,415 output tokens. Cost: $0 (free tier).

## Integrity checks (all pass)

| Check | Result |
|---|---|
| Record count matches matrix | 648/648 |
| Duplicate custom_ids | 0 |
| Per-condition balance | exactly 72 per condition (9 conditions) |
| Result types | 648 succeeded, 0 errored |
| Truncation (`stop_reason=length`) | 0 |
| Code extraction | 648/648 `labeled` python fences |
| Sandbox execution | 634 ok, 14 `code_error` (scored as failing tests) |
| Overall mean pass rate | 0.9645 — **ceiling risk for H3 confirmed**; per-tier and solvability-band sensitivity analyses are mandatory per the methods plan |

## Deviations / incidents

1. **Free-tier quota pacing.** Collection took ~2 days instead of hours; long
   Retry-After waits (up to ~7 min) honored throughout. No data impact.
2. **Mid-run pause (2026-10-14).** User-requested pause at 100/648 for a synthetic
   pipeline dress rehearsal; resumed from checkpoint. No data impact (shuffled order,
   checkpoint resume).
3. **Silent stalls (2):** Groq soft-throttling held connections open past the original
   600s socket timeout, wedging the runner twice (at 324 and 609). Fixes pushed
   mid-collection (`8438e69`): 180s timeout + retry on read stalls; both restarts
   resumed cleanly from the checkpoint. Post-fix, stalls self-cleared.
4. One education-suite partial file (`responses.jsonl`, 191 records of the v1 study,
   2026-10-07) coexists in `out/`; it belongs to the v1 pilot, not this dataset.

## Judge pass

Launched after collection (2026-10-16): `judge.py`, `openai/gpt-oss-120b` via Groq,
condition-blind prompts, checkpointed to `out/judge_scores_skill.json` every 25.

## Addendum (2026-10-21): stall root cause

The incidents in item 3 above, plus two judge-pass stalls (at 375 scored), trace to the
host Mac idle-sleeping (1-minute timeout) during long Retry-After sleeps, not to the
API. Judge pass relaunched under `caffeinate -i -s` from the 375 checkpoint; see
`EXPERIMENT.md` §10 (2026-10-21 entry).
