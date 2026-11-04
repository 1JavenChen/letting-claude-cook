# Stage 4 — Methods, Logistics, and Ethics

**Study:** How a user's *stated* programming skill level affects the code an LLM produces.
**Author:** methods-engineer. **Date:** 2026-10-14.
**Inputs:** `research/02_hypotheses_and_design.md` (governing design), `research/01_literature_review.md` §4,
`EXPERIMENT.md` §10 (pipeline lessons), existing `experiment/` pipeline.
**Status:** pipeline built and verified offline (dry-runs, sandboxed fixture tests, reference-solution
runs). No paid API calls were made in this stage. Ready for stage 5 (collection).

---

## 1. What was built

| Artifact | Contents | Verified by |
|---|---|---|
| `experiment/tasks.json` | 24 Python coding tasks: id, language, difficulty tier (A/B/C, 8 each), hardening flag + CWE tag (8 tasks), prompt, entry point, 3–6 hidden atomic requirements, 15–21 hidden assert-style tests (414 tests total incl. robustness), reference solution | every test executed against the reference solution, both in-process and through the real sandbox (`score.py --verify-tasks`: ALL PASS) |
| `experiment/conditions_skill.json` | frozen skill-level wordings + disclosure templates, mirroring `conditions.json` | dry-run renders inspected; byte-identity check below |
| `experiment/run_experiment.py` | extended with `--suite skill` (same backends, checkpointing, model resolution, backoff as v1); shuffled execution order with recorded seed; per-suite completion budget; temperature recorded per response | `--suite skill --dry-run` → 216 cells × 3 reps = 648 requests; built-in byte-identical-task-text check: OK; education suite unchanged (270 × 3 = 810) |
| `experiment/score.py` | sandboxed test runner + stdlib-only static metrics + prose metrics; writes `out/scores_skill.csv` + per-test detail JSONL | `--verify-tasks` (24/24 pass); end-to-end on 5 synthetic fixtures (`experiment/fixtures/fixture_responses.jsonl`) covering correct / buggy / no-code / hardened / infinite-loop responses — all scored as expected |

## 2. Design implementation

Matrix per the design doc §3/§10: **24 tasks × (4 skill levels × 2 channels + 1 control) × 3
replicates = 648 subject requests.** Cells per task = 9, matching the v1 layout.

- **Base system prompt** constant: `You are a helpful AI assistant.`
- **Skill levels** (frozen wording, length-matched at 11–12 words, stated as fact, no tone request):
  - beginner — "just starting to learn programming, currently taking a first programming course"
  - hobbyist — "a hobbyist programmer who has been coding casually for about a year"
  - professional — "a professional software engineer with about five years of industry experience"
  - expert — "a staff-level software engineer with fifteen-plus years of experience maintaining large systems"
- **Channels:** `system` → "You are a helpful AI assistant. The user is {level}." (third person);
  `message` → user turn prefixed "A bit about me: I am {level}.\n\n{task}" (first person).
  Control: base system prompt + bare task.
- **Task text byte-identical** across all nine cells (asserted automatically in `--dry-run`).
- **Sampling:** temperature 0.7 (the pipeline default, now recorded per response),
  `max_tokens` 2400 for the skill suite (code + explanation headroom; truncation is observable
  via the recorded `stop_reason`). gpt-oss reasoning kept at `reasoning_effort: low` with
  `<think>` blocks stripped (v1 behaviour, unchanged).
- **Order randomisation:** all 648 requests are shuffled with seed 20261014 (CLI-overridable,
  recorded in every response record), so rate-limit pacing and drift cannot alias with condition
  (design §6).
- **Checkpoint/resume/abort:** inherited from v1 — every success appended to
  `out/responses_skill.jsonl`, errors to `out/errors_skill.log` and retried on rerun, abort after
  5 consecutive failures, Groq model resolved against the account's live `/models` catalog at startup.

## 3. Task battery

8 tasks per tier; 8 hardening (security/robustness-sensitive) tasks with CWE tags and targeted
robustness tests. All prompts are register-neutral (no "simple"/"advanced" vocabulary), contain no
skill-revealing cues, and leave docstrings, type hints, validation, and error handling optional
unless the behaviour itself is the task (t17 cycle error, t19/t20 security contract, t24 invalid
expression), per design §4.4/§4.7. Tests check behaviour only, never style (§4.8). Most tasks are
freshly written for this study; a few reuse well-known task *shapes* (LIS, anagram grouping) with
new wording — contamination is mitigated rather than eliminated (design §4.6 "at least half fresh" is met).

| id | tier | task | entry point | hardening/CWE |
|---|---|---|---|---|
| t01 | A | running total | `running_total` | — |
| t02 | A | letter frequencies | `letter_frequencies` | — |
| t03 | A | chunk list | `chunk_list` | CWE-20 |
| t04 | A | second largest distinct | `second_largest` | — |
| t05 | A | palindrome phrase | `is_palindrome_phrase` | — |
| t06 | A | merge count dicts | `merge_counts` | — |
| t07 | A | clamp values | `clamp_values` | — |
| t08 | A | snake_case → camelCase | `snake_to_camel` | — |
| t09 | B | duration string → seconds | `parse_duration` | CWE-20 |
| t10 | B | group anagrams | `group_anagrams` | — |
| t11 | B | URL query-string parser | `parse_query` | — |
| t12 | B | business days between dates | `business_days_between` | CWE-20 |
| t13 | B | flatten nested lists | `flatten` | — |
| t14 | B | human-readable file size | `format_size` | — |
| t15 | B | top-n frequent words | `top_words` | — |
| t16 | B | merge intervals | `merge_intervals` | — |
| t17 | C | lexicographic topological sort | `topo_sort` | — |
| t18 | C | sliding-window rate limiter | `allowed_requests` | — |
| t19 | C | safe path resolution | `resolve_path` | CWE-22 |
| t20 | C | parameterized SQL builder | `build_query` | CWE-89 |
| t21 | C | template renderer | `render` | CWE-20 |
| t22 | C | LIS with performance pressure | `lis_length` | — |
| t23 | C | config parser with coercion | `parse_config` | CWE-20 |
| t24 | C | arithmetic expression evaluator (no eval) | `eval_expr` | CWE-95 |

Test strength (design §4.3 asks ≥15 cases incl. edge/boundary/invalid): every task has 15–21
hidden tests; hardening tasks add 3–6 robustness tests; t22 includes a 20,000-element performance
case that an O(n²) solution cannot finish inside the per-test time limit.

**Solvability screen (§4.2, 0.3–0.9 control pass rate):** executed *at stage 5* — the design's
pre-pilot screen needs model calls this stage may not make. Decision: run the full matrix, compute
control-cell pass rates first, and report any at-ceiling/at-floor tasks as a pre-registered
sensitivity analysis (primary models rerun excluding them) instead of replacing tasks mid-run.
A cheap preview is available as a `--reps 1` run before committing to the full one.

## 4. Measures: construct → column mapping

`score.py` writes one row per response to `experiment/out/scores_skill.csv`
(and per-test booleans to `scores_skill.detail.jsonl`). Primary metric per hypothesis in **bold**.

| Hypothesis | Columns |
|---|---|
| H1 style | **`comment_ratio`** = (comment + docstring lines) / code lines; `prose_words` (words outside fences), `docstring_lines`, `has_docstring`, `md_headers`, `md_bullets`, `identifier_len_mean`, `response_chars`, `output_tokens` |
| H2 complexity | **`cyclomatic_mean`** (1 + decision points per function: if/elif/for/while/except/assert/ternary, boolean operators, comprehension clauses); `cyclomatic_total`, `max_nesting`, `advanced_idioms` (comprehensions + lambdas + decorators + yields + walrus + type hints), `type_hints`, `loc`, `n_functions` |
| H3 correctness | **`pass_rate`** (fraction of hidden tests passed; extraction failure ⇒ 0 and `all_pass` = 0, per the pre-registered rule); `all_pass`, `extraction` (labeled/unlabeled/extraction_fail), `exec_status` |
| H4 robustness | **`validation_checks`** (AST count of `try` + `raise` + `isinstance`/`issubclass`); `try_count`, `raise_count`, `isinstance_count`, `robust_rate` (hardening tasks: handled-invalid-input rate), `security_flags` (eval/exec/`__import__`/os.system/pickle.load/shell=True count — stdlib Bandit-lite) |
| H5 substance | `requirements_met` — **blind-judge pass, stage 5/6** (see §5); `tasks.json` ships the hidden per-task requirement checklists |
| H6 channel | `channel` column interacts with all of the above |
| H7 default | control rows vs. level rows on H1/H2 primaries |
| H8 tone | `condescension` — blind-judge pass (see §5); `leakage` regex flag already computed |
| Manipulation check | `leakage` (response prose mentions the stated level) |

**Pre-registered extraction rule:** first fenced block labeled `python`/`py`/`python3`; else the
first unlabeled fenced block that compiles as Python; else `extraction_fail` (a recorded outcome,
analysed, not dropped). Unclosed trailing fences (truncated responses) are still extracted.

**Robustness ("handled invalid input") criterion:** on a hardening task's invalid input, the call
must raise `ValueError` or `TypeError` — a deliberate validation signal. Returning garbage, hanging,
or crashing with incidental exceptions (IndexError, KeyError, …) fails. *Known measurement noise:*
some unguarded implementations happen to raise ValueError incidentally (e.g. `range(..., 0)`,
`date.fromisoformat` on bad input), inflating robustness for non-validating code on t03/t12. This
is why `validation_checks` (AST-based) is H4's primary and `robust_rate` is supporting evidence.

**Sandbox:** each response runs once in `python -I` (isolated mode), empty environment, closed
stdin, temp working directory, 5 s wall-clock alarm per test (fresh globals copy per test), and an
overall subprocess timeout. Module-level code errors ⇒ `code_error`; missing entry point recorded;
no network access is assumed or needed. The fixture suite confirms an infinite-loop submission
costs per-test timeouts only and cannot stall the scorer.

## 5. Blind judge (specified here, executed in stages 5–6)

One judge request per response (648): judge sees **only** the bare task text, the hidden
requirements checklist, and the response — never the persona (v1 protocol). Judge model =
`openai/gpt-oss-120b` on Groq (subject model ≠ judge model); instructed to ignore any audience
references. Outputs JSON: `requirements_met` (checklist booleans), `condescension` 1–5,
`assumed_knowledge` 1–5, `hedging`/`refusal` flags. Responses whose `leakage` flag is set get a
pre-registered robustness re-analysis excluding them; a 10% stratified sample is hand-validated.
The judge runner is a small extension of the existing collection machinery (same backoff/
checkpoint loop over `scores_skill.csv` rows); implementing it is the first task of stage 5/6.

## 6. Decision log (design §11 open decisions + methods-level calls)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Backend = Groq free tier, `openai/gpt-oss-20b`** for the v2 run; judge `openai/gpt-oss-120b`. | No Anthropic credits on the account (EXPERIMENT.md §10). Pipeline is model-agnostic: a Claude rerun is `--backend anthropic` (one flag; model resolved per backend; README constraint that headline claims then name the model actually run **stands — findings from this run are about gpt-oss-20b, not Claude**). |
| 2 | **Skill-level wording** frozen as in design §3, length-matched 11–12 words. | Design-doc wording adopted verbatim in spirit; phrasing valid in both first and third person so the channel factor changes only person, not content. |
| 3 | **Full 24-task battery, 3 reps.** A pilot is the same matrix with `--reps 1` (216 requests) if wanted before the full run. | Design's power analysis fixed 24 tasks; a separate 12-task pilot would double collection passes on a free tier. |
| 4 | **Self-repair probe skipped in v1** (future work), as is the optional wording-robustness check. | Design lists both as optional/under-budget extras; "keep moving" directive. |
| 5 | **All 24 tasks are Python** — *deviation* from design §4.1 (16 Py / 4 JS / 4 other). | One sandbox (python -I) with per-test alarms covers correctness *and* the full static-metric battery for every task; JS/SQL/Go runners would add Node/DB/Go runtime dependencies and give correctness-only scoring for 8 tasks. Cost: cross-language generalisation is lost — flagged for the report's limitations and as future work. |
| 6 | **Solvability screen moved to stage 5** (sensitivity analysis instead of task replacement). | Screening requires model calls; replacement loops conflict with a pre-registered battery and with running on a free tier. See §3. |
| 7 | **Robustness criterion** = clean `ValueError`/`TypeError` on invalid input; `validation_checks` is H4's primary. | Automatable and frozen in advance; incidental-exception noise documented in §4. |
| 8 | **Extraction in `score.py`, not the runner.** | The runner stays a faithful collector of raw text; extraction is deterministic and re-runnable, so the frozen rule can be audited without re-collecting. |
| 9 | **Cyclomatic proxy is stdlib-AST based** (no Radon/Bandit deps). | `requirements.txt` stays minimal; the proxy counts the standard decision points; `security_flags` covers the Bandit-class calls relevant to these tasks. |
| 10 | **Shuffle seed 20261014**, temperature 0.7, max_tokens 2400 — all recorded per response. | Design §6 fixed-params + recorded-seed requirements. |
| 11 | **Judge implementation deferred to stage 5/6**, protocol frozen in §5. | Keeps stage 4 within scope; H5/H8 need the judge, all other hypotheses are scoreable today. |

## 7. Exact run commands (stage 5)

```bash
cd experiment

# 0. sanity check, no API calls
../.venv/bin/python run_experiment.py --suite skill --dry-run

# 1. (optional) one-rep pilot: 216 requests, same matrix
GROQ_API_KEY=... ../.venv/bin/python run_experiment.py --suite skill --reps 1

# 2. full collection: 648 requests -> out/responses_skill.jsonl
#    (rerun the same command to resume after rate limits / aborts)
GROQ_API_KEY=... ../.venv/bin/python run_experiment.py --suite skill

# 3. score everything -> out/scores_skill.csv + out/scores_skill.detail.jsonl
../.venv/bin/python score.py

# regression checks at any time (no API calls):
../.venv/bin/python score.py --verify-tasks
../.venv/bin/python score.py --responses fixtures/fixture_responses.jsonl \
    --out out/scores_fixture.csv --per-test-seconds 1

# Claude rerun (needs ANTHROPIC_API_KEY + credits; claims then are about Claude):
ANTHROPIC_API_KEY=... ../.venv/bin/python run_experiment.py --suite skill --backend anthropic
```

Note: a `--reps 1` pilot and the full run share `out/responses_skill.jsonl` — the full run resumes
past the pilot's 216 records (reps 2–3 only), so the pilot's cost is never wasted. Move the file
aside instead if a clean separation is wanted.

## 8. Runtime and request estimates

- **Subject run:** 648 requests. v1 collected 810 shorter responses in ~1–2 h on the Groq free
  tier with 429 pacing; code answers are longer (max_tokens 2400), so estimate **2–4 h**
  wall-clock, resumable. Zero dollars.
- **Judge run (stage 5/6):** 648 requests on `gpt-oss-120b`, similar pacing: ~2 h. Zero dollars.
- **Scoring:** local only; fixture timing extrapolates to roughly **5–15 min** for 648 responses
  (well-behaved code runs in <1 s; each hanging test costs its 5 s alarm).
- **Timeline:** collection (incl. reruns for rate limits) day 1; judge + scoring day 1–2;
  analysis (stage 6) day 2+.

## 9. Ethics note

No human subjects: all personas are synthetic credential sentences; no IRB review is required.
The equity framing (RQ3) is reported responsibly per the standing README constraint: raw
low-vs-high-skill gaps with 95% CIs and effect sizes, not bare significance verdicts; claims are
about a model's response to a *stated credential*, never about any population of programmers.
Because decision 1 runs the collection on `openai/gpt-oss-20b`, every report from this run must
name that model and must not attribute findings to Claude. Sandboxed execution of model-generated
code uses an isolated interpreter with no environment, no stdin, and timeouts, so untrusted code
cannot touch credentials or the network path of the collection machinery.

## 10. Verification record (what was actually run in stage 4)

1. Task builder: all 414 tests executed in-process against the 24 reference solutions — 0 failures
   (one hand-computed expectation for t18 was wrong against the verified reference and was fixed
   before freezing `tasks.json`).
2. `score.py --verify-tasks`: 24/24 reference solutions pass all main + robustness tests through
   the real sandbox (`python -I` subprocess).
3. `run_experiment.py --suite skill --dry-run`: 216 cells × 3 reps = 648; example cells rendered;
   byte-identical task text across conditions: OK. Education suite dry-run unchanged (810).
4. Persona-cell render check: beginner/system and expert/message cells produce the intended
   system prompt and message prefix; 648 unique custom_ids after shuffling.
5. Fixture end-to-end: 5 synthetic responses scored — correct (pass_rate 1.0, docstring + type
   hints detected, leakage flag correctly fires on persona-echoing prose), buggy (0.25),
   no-code (`extraction_fail`), hardened (robust_rate 1.0, validation_checks 4),
   infinite loop (0.0 via per-test timeouts; scorer completes).

No paid or external API call was made during stage 4.

## 11. Pre-analysis audit resolutions (2026-10-14, before confirmatory stats)

- **`comment_ratio` definition confirmed, not a bug.** Implementation matches the
  frozen definition: (comment + docstring lines) / code lines, where code lines
  exclude comments and docstrings. Range is [0, ∞); interim real values of 4–6 mean
  the subject model writes several documentation lines per code line. Retained
  unchanged for H1; interpret as documentation density, not a proportion.
- **`security_flags` detector verified live.** A synthetic snippet with `eval`,
  `os.system(...)`, and `pickle.load(...)` scores 3; clean code scores 0. All-zero
  values on real responses therefore indicate clean generated code, not a scoring gap.
