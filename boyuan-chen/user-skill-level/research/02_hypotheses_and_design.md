# Stages 2–3 — Research Questions, Hypotheses, and Research Design

**Study:** How a user's *stated* programming skill level affects the code an LLM (Claude)
produces. **Author:** research-designer. **Date:** 2026-10-14.
**Inputs:** `research/01_literature_review.md` (gap numbers G1–G6 below refer to its §3),
`EXPERIMENT.md` (v1 education-level study; design vocabulary reused).
**Status:** awaiting human approval before stage 4 (see `research/README.md`).

---

## 1. Research questions

| RQ | Question | Gap |
|---|---|---|
| **RQ1 Style adaptation** | Does stated skill level change the *form* of generated code and prose — comments/docstrings, naming, structural complexity, explanation length, jargon? | G1, G3 |
| **RQ2 Substance & quality** | Holding the task fixed, does stated skill level change *what is delivered*: functional correctness (hidden tests), implemented requirements, input validation/error handling, edge-case coverage, security hardening? | G2, G3 |
| **RQ3 Equity (scaffolding vs. rationing)** | When skill is low, is the change benign scaffolding (simpler style, *equal* functional completeness) or harmful rationing (fewer requirements met, dropped error handling/security, lower pass rate, more refusals/hedging/condescension)? | G4 |
| **RQ4 Channel** | Does it matter how the model learns the skill level — operator system-prompt profile (third person) vs. user self-disclosure in the message (first person)? | G5 |
| **RQ5 Shape of the response** | Is the effect monotonic in stated skill, or does the no-information control sit at an extreme (i.e., does the model default to "assume expert" or "assume novice")? | G1, G6 |

All RQs are asked of Claude specifically (G6); results are not to be generalized to other
models unless a comparison run is added.

## 2. Hypotheses

Directional and falsifiable. "Level" = stated skill (beginner < hobbyist < professional <
expert). Metrics are defined in §5 and finalised in stage 4.

- **H1 (style, complexity of explanation).** Comment/docstring density and the
  explanation-to-code ratio *decrease* monotonically with stated skill level
  (beginner > hobbyist > professional > expert). Test: mixed model on `comment_ratio`,
  `prose_words`; Jonckheere–Terpstra trend across levels within task. **Falsified if**
  the level coefficient CIs include 0 for both metrics.
- **H2 (code complexity).** Cyclomatic/cognitive complexity and use of advanced idioms
  (comprehensions, generics/type hints, decorators, functional style) *increase*
  with stated skill. Test: `cyclomatic`, `advanced_idiom_count` vs. level. **Falsified if**
  no monotone trend (or opposite sign).
- **H3 (correctness — the equity primary).** Hidden-test pass rate is **not** equal
  across levels: beginner < expert. Rationale: Gupta et al. (2024) persona-induced
  degradation, Poole-Dayan et al. (2024) lower-status → worse answers; competing
  prediction from Wang et al. (2025): expertise framing has negligible effect on
  substance (a null). **Both outcomes are informative**; the null is pre-registered as the
  outcome of "pass-rate gap |beginner − expert| < 3 percentage points with the 95% CI
  inside ±6 pp" (equivalence margin).
- **H4 (rationing of robustness).** Beginner-stated conditions receive *less* input
  validation / error handling / security hardening than expert-stated conditions
  (`validation_checks`, `security_flags`), independent of correctness. This is the
  "silent drop" hypothesis (G3); direction could also reverse (beginners get *more*
  defensive code) — a two-sided test is run and the directional prediction is as stated.
- **H5 (scaffolding, not rationing).** For beginner vs. expert, style shifts (H1) occur
  *without* a matching drop in implemented requirements: judge-rated
  `requirements_met` difference is within the equivalence margin (±0.5 of a 0–N
  requirement count normalised to proportion, ±5 pp) even where H3/H4 are significant
  on other axes. Failure of this hypothesis, with H1 true, is the paper's "rationing" finding.
- **H6 (channel).** Self-disclosure (first person) produces larger style shifts than the
  system-prompt profile: the `level × channel` interaction on H1/H2 metrics is positive
  in magnitude (|β_message| > |β_system|). Competing prediction (Reusens et al., 2024):
  explicit self-disclosure *anchors* and also yields larger *quality* penalties; this is
  tested separately on pass rate (H3 × channel).
- **H7 (default assumption).** The no-information control is closest to the
  professional/expert end on H1/H2 metrics (model assumes a competent user by default):
  on `comment_ratio` and `cyclomatic`, the control mean is closer to the professional condition than to the beginner condition. **Falsified** if control is closer to beginner/hobbyist.
- **H8 (condescension).** Blind-judge `condescension` rating is higher for beginner than
  expert and ≈ control for professional/expert. (Exploratory-confirmatory; judged blind.)

**Multiplicity plan.** Confirmatory family = {H1, H2, H3, H4, H5, H6, H7}, one primary
metric each (named in §5), Holm-adjusted at α = .05. H8 and all secondary metrics are
exploratory and labeled as such in the analysis.

## 3. Research-design choice

**Mixed methods, quantitative-dominant.** Primary evidence is a **within-task factorial
experiment** (carried from v1: each task is its own control), scored with *automated,
deterministic* measures (unit tests, static analysis) wherever possible, plus a *blind*
LLM judge for the constructs that can't be computed (requirements met, condescension).
A small **qualitative pass** (open coding of ~40 stratified response pairs, beginner vs.
expert on the same task) is added to interpret *why* the numbers move (tone, what was
omitted) and to audit the judge; it is interpretive only and never used for hypothesis tests.

Why not purely qualitative: the equity question (RQ3) turns on magnitudes of small
differences in pass rate and robustness features across many tasks; that needs counts and
effect sizes. Why not purely quantitative: skill-conditioned *tone* (patronising vs.
supportive) is not captured by static metrics.

### Factors

| Factor | Levels | Notes |
|---|---|---|
| Task | 24 | Random effect. Tagged by language, difficulty tier, type (§4). |
| Stated skill level | 4 | `beginner` (just starting; first programming course) · `hobbyist` (~1 year of casual coding) · `professional` (software engineer, ~5 years) · `expert` (staff-level engineer, 15+ years, maintainer of large systems). Exact wording frozen in `conditions_skill.json` by methods-engineer. |
| Disclosure channel | 2 | `system`: operator profile in the system prompt, third person ("The user is {level}."). `message`: self-disclosure prefix in the user turn, first person ("A bit about me: I am {level}."). |
| Control | 1 | No information: base system prompt, bare task. Reference level for contrasts. |
| Replicates | 3 | Independent samples per cell, default sampling temperature, to estimate within-cell variance (Zheng et al., 2024: persona effects are noisy). |

Cells per task: 4 × 2 + 1 = **9**. Matches v1 layout so `run_experiment.py` / `analyze.py`
patterns transfer.

### Optional extension (not in the confirmatory design)

A **second-turn self-repair** probe (Jain et al., 2024): for the beginner and expert
`message` cells only, feed back a failing test and measure repaired pass rate. Run **only
if** the main collection finishes under budget; reported as exploratory. Not counted in
the request total below.

## 4. Task battery specification

**Size:** 24 tasks (fixed by power analysis, §6). Final tasks are chosen in stage 4; this
section fixes *selection criteria*.

1. **Languages:** 16 Python, 4 JavaScript, 4 other (e.g., SQL, Java, Go, Bash) — Python
   dominates because the static-analysis toolbox is mature there; non-Python tasks are
   tested with language-appropriate runners and scored for correctness only, with static
   metrics restricted to a language-agnostic subset (comment ratio, line count, nesting).
2. **Difficulty tiers (balanced, 8 per tier):**
   - *Tier A* — MBPP-style entry-level (Austin et al., 2021): both a beginner and an expert
     could plausibly ask it.
   - *Tier B* — intermediate (parsing, simple data structures, string/date handling,
     API design with edge cases).
   - *Tier C* — hard/realistic (concurrency, algorithmic optimisation, small multi-function
     modules; SWE-bench-flavoured scope reduced to a single file).
   **Baseline-solvability target:** pilot Claude on the *control* condition; keep tasks whose
   control pass rate is between 0.3 and 0.9. Tasks at ceiling (1.0) or floor (0.0) cannot
   reveal a correctness effect (Liu et al., 2023 show weak tests mask differences) and are
   replaced.
3. **Automated correctness:** every task ships a hidden test suite with ≥ 15 cases
   (including edge/boundary/invalid inputs), strengthened EvalPlus-style by
   differential/mutation checks against a reference solution (Liu et al., 2023). Tests
   must run in a sandbox with a timeout.
4. **Room for style variation and robustness:** each prompt is underspecified enough that
   a response *may* add docstrings, type hints, input validation, error handling, or
   explanation — i.e., these are optional, not demanded. At least 8 tasks (the "hardening
   subset") are explicitly **security-/robustness-sensitive** (path handling, SQL
   construction, untrusted input parsing, resource limits), with CWE-tagged checks
   (Pearce et al., 2022) via static analysis plus targeted failing-input tests.
5. **Requirement counting:** each task lists 3–6 explicit atomic requirements in the
   *hidden* spec (not shown to the model, derivable from the prompt) so that
   `requirements_met` is judged against a fixed checklist, making H5 measurable.
6. **Contamination control:** prefer newly written tasks (not verbatim from
   HumanEval/MBPP); at least half written fresh and unseen online (Jain et al., 2024).
7. **Neutrality of wording:** task text must be plausible from a user of *any* skill level
   — no skill-signalling vocabulary, no "simple/advanced" words, neutral register.
8. **No skill leakage in tests:** tests must not reward style (e.g., comments) — they test
   behaviour only.

## 5. Measures (specification; implementation in stage 4)

| Construct | Metric (primary in **bold**) | Tool |
|---|---|---|
| Correctness (H3) | **`pass_rate`** (fraction of hidden tests passed); `all_pass` (binary); extraction failure | sandboxed test runner |
| Style (H1) | **`comment_ratio`** (comment+docstring lines / code lines); `prose_words` (words outside code fences); `identifier_len`; header/bullet counts | static / regex |
| Complexity (H2) | **`cyclomatic`** (mean per function; Radon for Python); `max_nesting`; `advanced_idiom_count` (comprehensions, type hints, decorators, lambdas, generators) | Radon / AST |
| Robustness (H4) | **`validation_checks`** (raise/guard/`isinstance`/try-except counts); handled-invalid-input rate from the hardening-subset tests | AST + tests |
| Security (H4) | `security_flags` (Bandit-class findings for Python; CWE-keyed tests for hardening subset) | Bandit / tests |
| Substance (H5) | **`requirements_met`** proportion against hidden checklist | blind judge + tests |
| Tone (H8) | `condescension` 1–5; `assumed_knowledge` 1–5; hedging/refusal flags | blind judge |
| Manipulation check | Leakage: response mentions the stated level / "beginner" etc. | regex |
| Length | output tokens; total lines | API usage |

**Blind judge:** sees only the bare task text and the response, never the persona (v1
protocol); instructed to ignore references to the audience; leakage flag used for a
robustness re-analysis excluding leaking responses. Judge model ≠ subject model where
possible; a 10% sample is hand-validated (agreement reported).

## 6. Confound control

- **Byte-identical task text** across all nine cells; only the credential sentence varies
  (matched-guise logic: Hofmann et al., 2024). We manipulate the **stated credential only,
  never the user's writing competence** — no typos, no informal register in any condition.
- **Base system prompt** held constant (`You are a helpful AI assistant.`), as in v1.
- **Length-matched credential sentences** across levels (within ±3 words) so prompt
  length is not a confound.
- **Neutral skill phrasing:** each level stated as a fact, no request for tone ("explain
  simply") — we measure the model's *unprompted* adaptation.
- **Randomised order** of request execution (shuffle seed recorded) to prevent drift/rate-limit
  patterns from aliasing with conditions.
- **Fixed sampling params**; model ID resolved against the live catalog and recorded per
  response (EXPERIMENT.md §10 lesson); checkpoint every response; errors retried on rerun;
  abort after 5 consecutive failures.
- **Code extraction rule** frozen in advance: the first fenced block of the target
  language; responses with no extractable code scored `extraction_fail` (a recorded outcome,
  not dropped). Chance that this interacts with level (beginners get prose-heavy
  answers) is itself an analyzable outcome and counted as a correctness failure in
  `all_pass`, with a sensitivity analysis that excludes extraction failures.
- **Stated-but-not-inferable skill:** the task text contains no skill-revealing cues
  (§4.7), so the only route from credential to output is the credential itself.

## 7. Power analysis

Unit of analysis for primary inference: task-level means (24 tasks), each averaged over
3 replicates (and, for pooled contrasts, over channels), analysed with a mixed model
`metric ~ level * channel + (1 | task)` and a paired nonparametric check.

- **Observations per cell:** 24 tasks × 3 reps = **72** responses per level × channel
  cell; **144 per level** pooled over channels; control = 72.
- **Pooled level contrast** (e.g., beginner vs. expert, paired by task, n = 24 task means,
  two-sided α = .05): power 0.80 for a standardised paired effect **d_z ≈ 0.60**. Averaging
  three replicates lowers the noise component, so the detectable effect in *raw* terms is
  smaller than the single-response SD would imply when between-replicate variance is
  large (typical for persona effects, Zheng et al., 2024).
- **Per-channel contrast:** n = 24 → same d_z ≈ 0.60 per channel; the interaction (H6)
  needs roughly twice the effect (d_z ≈ 0.8) — H6 is flagged *underpowered/exploratory-
  confirmatory* and reported with CIs rather than as a verdict.
- **Binary pass/fail (H3):** with ~72 binary observations per cell around a 0.6 base
  rate, a pooled beginner-vs-expert gap of ~12–15 pp is detectable at 0.80 (mixed
  logistic model with task random intercept; design effect reduces this somewhat, so
  treat ~15 pp as the realistic floor). Gaps below that will be reported as
  *inconclusive*, not as "no effect"; the equivalence test (±6 pp) in H3 requires the CI
  to be that tight and is expected to be met **only if** the true gap is ≈ 0 and variance
  is low — pre-stated as possibly unmet.
- **Sensitivity of the style metrics (H1/H2):** prior persona-steering work reports large
  style effects, so d_z ≈ 0.6 is conservative for these; correctness (H3) is where
  power is the binding constraint.
- **Replicates:** 3 per cell is a pragmatic floor; if the pilot shows between-replicate
  SD of `pass_rate` > the between-task SD, increase to 5 for the Tier-B/C subset.
- **Reporting:** report raw low-vs-high gaps (percentage points, comment-ratio points,
  complexity points) with 95% CIs, not only p-values (Hu & Collier, 2024; README
  equity constraint).

## 8. Threats to validity (design level)

- **Persona ≠ person.** Stated credential isolates the causal effect of the *label*, but
  real users also leak skill through phrasing; total real-world adaptation is understated.
- **Judge bias.** Judge sees no persona; leakage re-analysis; 10% human validation.
- **Ceiling/floor tasks.** Controlled by the 0.3–0.9 control-pass-rate screen; screen
  uses the control cell only, so it does not select on the outcome of interest.
- **Subject model identity.** Headline claims are about Claude only if the Anthropic
  backend is used; otherwise they are about the substitute model (README constraint) and
  the report must say so. **Human decision needed before stage 5.**
- **Single-turn only.** Multi-turn drift not covered; self-repair extension is the first step.
- **Overfitted "novice" wording.** Four phrasings fixed in advance; a wording-robustness
  check (swap in an alternate phrasing for 3 levels on 6 tasks, ~162 extra requests) is
  listed as optional.

## 9. Ethics

Synthetic personas, no human subjects, no IRB. Equity framing reported with magnitudes
and uncertainty; no claim that outputs are "biased against" a population — claims are
about a model's response to a *stated credential*.

## 10. Decision table (factors × levels) and request count

| Factor | Levels | Count |
|---|---|---|
| Tasks | 24 (16 Py / 4 JS / 4 other; tiers A/B/C 8 each) | 24 |
| Stated skill | beginner, hobbyist, professional, expert | 4 |
| Channel | system (3rd person), message (1st person) | 2 |
| Control | no information | 1 cell |
| Replicates | independent samples | 3 |
| **Cells per task** | 4 × 2 + 1 | **9** |

- **Subject requests:** 24 × 9 × 3 = **648**
- **Judge requests (blind, one per response):** 648
- **Pilot (control only, task screening):** ~40 tasks × 3 = 120 requests (calibrates the
  0.3–0.9 solvability screen; precedes the main run, not part of the 648)
- **Total planned model calls:** 648 + 648 + ~120 = **~1,416**
- Optional extras (excluded above): self-repair probe (~144 requests), wording-robustness
  check (~162 requests).

## 11. Open decisions for the human

1. **Subject model/backend:** Anthropic (needs credits; claims about Claude) vs. Groq
   free tier (`openai/gpt-oss-20b`; pipeline pilot only).
2. **Approve skill-level wording** (beginner / hobbyist / professional / expert definitions).
3. **Approve 24-task / 3-replicate scale** vs. a cheaper 12-task / 3-replicate pilot first.
4. Whether to include the optional self-repair probe.
