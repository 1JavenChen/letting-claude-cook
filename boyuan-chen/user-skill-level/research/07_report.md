> **Status banner (stage 8, 2026-10-28).** All 648 subject responses are collected and the
> automated metrics are analyzed. The blind-judge pass is **incomplete** (judge scores exist
> for 225/648 responses, 16 of 24 tasks), so the two judge-dependent hypotheses, H5 and H8,
> are reported provisionally and are marked **[PROVISIONAL]** throughout. They must be re-run
> after `experiment/judge.py` and `experiment/stats.py` finish locally
> (`research/05_judge_readiness.md`); `research/05_analysis.md` does not yet exist. The
> subject model is `openai/gpt-oss-20b`, **not Claude**. Every statistic below is copied from
> `research/05_analysis_automated_metrics.md`, `research/05b_ablations.md`, or
> `research/06_findings.md`.

---

# Stated Programming Skill and the Code an LLM Writes: A Pre-Registered Within-Task Experiment on Credential-Conditioned Generation

**Version:** 2026-10-28 (stage 8, complete draft; judge-dependent results provisional) · **Pipeline:** `experiment/` at
commit-time state · **Subject model for this run:** `openai/gpt-oss-20b` (Groq) ·
**Judge model:** `openai/gpt-oss-120b` (Groq)

---

## Abstract

Large language models demonstrably adapt their output to who they believe the user is,
and prior work shows that attributes signaling lower status or expertise can degrade not
just the style but the substance of what users receive (Gupta et al., 2024; Poole-Dayan
et al., 2024; Reusens et al., 2024). Whether a user's *stated programming skill level*
changes the *code* an LLM produces — its correctness, complexity, documentation, error
handling, and security hardening — has not been measured. We ran a pre-registered
within-task factorial experiment: 24 Python tasks × 4 stated skill levels (beginner,
hobbyist, professional, expert) × 2 disclosure channels (operator system prompt vs.
first-person self-disclosure) + a no-information control, with 3 replicates per cell
(648 responses), holding task text byte-identical across all conditions so that only the
credential sentence varies. Outputs were scored with sandboxed hidden test suites (414
tests), deterministic static metrics, and a condition-blind LLM judge; eight hypotheses
(H1–H8) with primary metrics, equivalence margins, and a Holm-corrected confirmatory
family were frozen in advance. Stated skill reshaped the *form* of the code but not its substance. Beginner-framed prompts
received more prose around the code (+23.4 words, 95% CI [+8.3, +43.2]) and code with fewer
advanced idioms (−1.06 per response, CI [−1.47, −0.66]), both exploratory secondary metrics.
Pre-registered primaries for documentation density (H1), cyclomatic complexity (H2),
validation checks (H4), the channel interaction (H6), and the default-user assumption (H7)
were not significant after Holm correction. Hidden-test pass rate differed by +1.4 pp
(beginner − expert, CI [−2.3, +5.6]), meeting the pre-registered equivalence criterion (H3),
though at a ceiling (mean pass rate 0.9645). On a partial judge sample, requirements met
were equivalent across levels (H5, provisional) and condescension was inconclusive (H8,
provisional). Findings concern `openai/gpt-oss-20b`; replication on Claude is outstanding.
In the present run, stated skill shifted form (about +23 words of prose and about −1.06 advanced idioms for beginners relative to experts) without a detectable correctness gap, which is more consistent with scaffolding than rationing on this model. As
motivating evidence, a v1 pilot on the adjacent education-level manipulation (191
responses, same pipeline, `openai/gpt-oss-20b`) found reading grade monotone in stated
education (≈7 → ≈12) with the no-information control at the PhD end. Findings from the
present run characterize `openai/gpt-oss-20b`, not Claude; the pipeline reruns on any
backend with one flag. Whether the same holds for Claude, and on harder tasks where correctness is not at ceiling, remains open.

**Keywords:** large language models, code generation, persona conditioning, sycophancy,
algorithmic equity, pre-registration

---

## 1. Introduction

Code assistants are most valuable to the programmers least equipped to audit them.
Novices gain the largest measured productivity benefits from LLM pair programmers (Peng
et al., 2023) while also being the least able to detect when generated code is wrong
(Moradi Dakhel et al., 2023; Nguyen et al., 2024). If a model, on learning that its user
is a beginner, silently changes *what it delivers* — fewer requirements implemented,
less input validation, missing security hardening, lower pass rates — the harm lands
precisely on the users least able to notice. The benign alternative is equally
plausible: a model may *scaffold* for novices (simpler style, more explanation, equal
functional substance), which the computing-education literature treats as desirable
adaptation (Prather et al., 2023; Kazemitabaar et al., 2023). Distinguishing scaffolding
from rationing requires measuring style and substance separately, on the same tasks,
with the user's stated credential as the only manipulated variable. That is what this
study does.

### 1.1 Why stated skill level, and why now

Two adjacent literatures make the question urgent but leave it unanswered. First,
persona and attribute conditioning measurably steers LLM output: impersonating an
expert improves task performance (Salewski et al., 2023), socio-demographic personas
silently degrade reasoning (Gupta et al., 2024), and models grow more sycophantic with
scale and preference training (Perez et al., 2022; Sharma et al., 2024). Second, the
equity literature shows that user attributes signaling lower status already produce
worse *substance*: less accurate and less truthful answers for users with lower
education (Poole-Dayan et al., 2024), degraded responses when a user discloses
non-native English (Reusens et al., 2024), and covert dialect prejudice that survives
alignment training (Hofmann et al., 2024). Yet the nearest code-domain study
manipulates how the prompt is *written*, not what the user *claims* (Rojpaisarnkit et
al., 2025), and the most direct mechanistic result reports that expert framing of the
user has a negligible effect on sycophancy in factual-agreement tasks (Wang et al.,
2025) — a null that has never been tested against the properties of generated code.

### 1.2 Gaps this study addresses

The literature review (`research/01_literature_review.md` §3) identified six specific
gaps, which this experiment is designed to close:

1. **G1 — No credential-only manipulation on code.** No study holds the prompt text
   byte-identical while varying a *stated* programming skill level and measuring the
   generated code. Nearest works confound credential with linguistic signal
   (Rojpaisarnkit et al., 2025; Reusens et al., 2024) or study education level on
   factual QA rather than code (Poole-Dayan et al., 2024).
2. **G2 — The expertise-framing null is untested for generation substance.** Wang et
   al. (2025) measured deference to user claims, not the correctness, error handling,
   or edge-case coverage of generated artifacts; the two behaviors plausibly
   dissociate.
3. **G3 — Non-correctness code axes have never been tested for persona sensitivity.**
   Models have measurable default styles in readability, maintainability, and
   efficiency (Zheng, Cao, et al., 2024) and a reusable security-evaluation
   methodology exists (Pearce et al., 2022), but no one has asked whether these axes
   shift with the user's stated skill.
4. **G4 — Scaffolding vs. rationing is unoperationalized.** The education literature
   reads simplified output as support (Prather et al., 2023; Kazemitabaar et al.,
   2023); the equity literature reads attribute-conditioned substance reduction as
   harm (Poole-Dayan et al., 2024). No study separates simpler *style* with equal
   *functional completeness* from genuinely reduced substance, for code.
5. **G5 — Disclosure-channel effects are untested for skill level.** First- vs.
   third-person framing changes sycophancy (Wang et al., 2025) and explicit
   disclosure can anchor degradation (Reusens et al., 2024), but no code-generation
   study compares an operator-set profile ("the user is a beginner") with
   self-disclosure ("I'm a beginner").
6. **G6 — Claude is under-measured.** Published persona/equity results cover
   GPT-class and Llama-class panels; none characterizes skill-level conditioning of
   Claude's code output. (This run, by backend necessity, measures
   `openai/gpt-oss-20b`; the Claude claim awaits an Anthropic-backend rerun — see
   §3.8.)

### 1.3 Preliminary evidence: the v1 education-level pilot

This study's design descends from a v1 experiment on the adjacent manipulation —
stated *education level* and general question answering (`EXPERIMENT.md`). A partial
v1 collection (191 of 810 planned responses, covering roughly the first 7 of 30
questions balanced across all nine conditions, on `openai/gpt-oss-20b`) yielded
descriptive pilot results that motivate the present hypotheses. Flesch-Kincaid reading
grade was monotone in stated education level — approximately grade 7.2–7.5 for the
"did not finish high school" persona rising through 8.2–8.7 for intermediate levels to
10.5–12.1 for the PhD persona, a swing of roughly four grade levels. The
no-information control sat at FK grade 10.3, at the PhD end of the range, suggesting
the model writes for an educated reader by default. In the system-prompt channel,
lower-education personas also received *shorter* answers (523 words vs. 750 for PhD),
a possible substance effect the v1 pilot could not separate from style. Persona
leakage ("since you have a PhD…") rose sharply with stated level. These are
descriptive numbers from a pilot on ~7 questions with no judge pass and no inferential
statistics, on gpt-oss-20b rather than a production assistant; we treat them strictly
as motivation. The present study carries the same within-item, byte-identical-prompt,
dual-channel architecture into the code domain, where substance is far more
objectively measurable: hidden test suites decide correctness, static analysis decides
complexity, and CWE-targeted checks decide hardening.

### 1.4 Contributions

(1) The first credential-only causal design for code generation: matched-guise logic
(Hofmann et al., 2024) applied to programming skill, with the task text byte-identical
across nine conditions. (2) A frozen, fully automated substance/style measurement
battery — 414 hidden tests across 24 tasks, 8 CWE-tagged hardening tasks,
deterministic static metrics, and a condition-blind judge — with all eight hypotheses,
primary metrics, equivalence margins, and the multiplicity plan pre-registered before
collection. (3) Evidence that, for one open model, stated skill level changes the presentation of code (prose and idiom choice) while hidden-test correctness stays equivalent within a pre-registered margin, with all effects reported as raw magnitudes and confidence intervals. (4) A reusable,
backend-agnostic pipeline enabling the same pre-registration to be rerun on Claude or
any OpenAI-compatible model.

---

## 2. Related Work

### 2.1 Persona conditioning and its limits

Telling a model who it is, or who the user is, changes behavior — but unevenly.
Salewski et al. (2023) showed that in-context impersonation of a domain expert
improves task performance relative to non-expert personas, while also surfacing age
and gender stereotypes. Gupta et al. (2024) found the darker complement: across 19
socio-demographic personas, 24 reasoning datasets, and 4 models, persona assignment
causes substantial performance drops — often via abstention — revealing stereotypes
the models disavow when asked directly, with de-biasing prompts barely helping.
Against these, Zheng, Pei, et al. (2024) tested 162 personas across 4 model families
on 2,410 factual questions and found persona system prompts do not reliably improve
accuracy; per-persona effects are largely random. Hu and Collier (2024) quantified the
ceiling, showing persona variables explain under 10% of annotation variance in
subjective tasks. Jointly, these studies establish that attribute-conditioning effects
are real but heterogeneous and noisy — motivating this study's within-task replicated
design, effect-size reporting, and pre-registered equivalence margins rather than bare
significance tests.

### 2.2 Sycophancy: what the model bends to

Perez et al. (2022) documented at scale that larger, more RLHF-trained models
increasingly echo the user's stated views, and Sharma et al. (2024) traced sycophancy
across five production assistants to human preference data that rewards agreement.
Most pertinent here, Wang et al. (2025) decomposed *which* prompt elements trigger
sycophancy: first-person opinion statements reliably induce it, while framing the user
as an expert has a negligible effect — models do not appear to internally encode user
authority. That result is the strongest competing prediction for the present study
(no substance effect of stated skill), but it was obtained on factual-agreement tasks,
not on generation. Bo et al. (2026) connect sycophancy to the novice setting:
novices paired with a highly sycophantic chatbot corrected fewer misconceptions,
over-relied on unhelpful responses, performed worse, and mostly failed to notice —
evidence that any skill-conditioned agreement behavior would disproportionately harm
the users least able to detect it.

### 2.3 Measuring LLM-generated code beyond pass@k

Functional correctness evaluation is mature: HumanEval's docstring-to-function
problems scored with hidden unit tests and pass@k (Chen et al., 2021), MBPP's 974
entry-level tasks (Austin et al., 2021), repository-scale SWE-bench (Jimenez et al.,
2024), and contamination-controlled, continuously refreshed LiveCodeBench (Jain et
al., 2024). Liu et al. (2023) showed that test adequacy is decisive — roughly 80×
more tests cut measured pass@k by up to 19–29% and reorder model rankings — so any
cross-condition correctness comparison needs strong suites; our battery ships 15–21
hidden tests per task, validated against reference solutions. Beyond correctness, the
RACE benchmark scores readability, maintainability, and efficiency and finds models
hold inherent stylistic preferences they struggle to override on request (Zheng, Cao,
et al., 2024) — supplying operational definitions for exactly the non-correctness axes
our hypotheses target. Security is separately measurable: Pearce et al. (2022) found
roughly 40% of 1,689 Copilot completions across 89 CWE-based scenarios vulnerable,
and their CWE-scenario-plus-static-analysis template underlies our eight hardening
tasks.

### 2.4 Expertise asymmetries in human studies of code assistants

Human-subject studies consistently find that expertise moderates both the value and
the risk of code assistants. Copilot did not improve completion time or success for 24
participants who nonetheless preferred it and struggled to understand and debug its
output (Vaithilingam et al., 2022). Interaction is bimodal — acceleration when the
programmer knows the next step, exploration when they do not (Barke et al., 2023) —
predicting that experts and novices want different output properties. Novices exhibit
metacognitive difficulties with suggestions (Prather et al., 2023) and fail at every
step of the text-to-code loop — writing prompts, judging output, repairing prompts —
even on level-appropriate problems (Nguyen et al., 2024). Yet generator access
*helped* 69 young novices without harming retention (Kazemitabaar et al., 2023), and
the largest productivity gains in a randomized trial accrued to less-experienced
developers (Peng et al., 2023). Moradi Dakhel et al. (2023) crystallize the asymmetry:
Copilot is an asset for experts and a liability for novices who cannot filter buggy
suggestions. If models additionally serve *disclosed* novices worse, identical tasks
would carry compounding risk for exactly the users with the most to gain.

### 2.5 Attribute-conditioned quality disparities

The strongest precedents for the equity framing come from outside the code domain.
Poole-Dayan et al. (2024) varied English proficiency, education level, and country of
origin and found state-of-the-art models produce less accurate, less truthful answers
and more refusals for lower-proficiency, lower-education, and non-US users. Reusens et
al. (2024) found quality gaps for non-native prompts plus an anchoring effect:
*explicitly telling* the model the user is a non-native speaker further degrades
quality — the single most important precedent for our disclosure manipulation.
Hofmann et al. (2024) showed via matched-guise probing that dialect alone triggers
covertly prejudiced downstream decisions while overt attitudes stay positive, and that
alignment training conceals rather than removes the effect — which is why our analysis
measures substance (tests passed, requirements implemented) rather than tone alone. In
the code domain, Rojpaisarnkit et al. (2025) showed that the English-proficiency level
of the prompt writing changes generated-code correctness; by holding writing constant
and varying only the stated credential, the present design isolates the complementary
causal path.

---

## 3. Method

The design was fixed in `research/02_hypotheses_and_design.md` (stages 2–3) and
implemented and frozen in `research/03_methods_plan.md` and the `experiment/` pipeline
(stage 4) before any subject API call. Deviations between the two documents are logged
in §3.9.

### 3.1 Design

A within-task factorial experiment in which every task appears in every condition,
so each task serves as its own control:

| Factor | Levels | Role |
|---|---|---|
| Task | 24 Python coding tasks | random effect |
| Stated skill level | beginner · hobbyist · professional · expert | fixed |
| Disclosure channel | `system` (third person) · `message` (first person) | fixed |
| Control | no skill information | reference level |
| Replicates | 3 independent samples per cell | within-cell variance |

Cells per task: 4 levels × 2 channels + 1 control = 9. Total subject requests:
24 × 9 × 3 = **648**, plus one blind-judge request per response (648) — ~1,300 model
calls in the confirmatory run.

### 3.2 Conditions: exact frozen wording

The base system prompt is constant in all conditions: `You are a helpful AI
assistant.` The four skill levels are stated as facts, length-matched at 11–12 words,
with no tone request (the model's *unprompted* adaptation is the measurand):

- **beginner** — "just starting to learn programming, currently taking a first
  programming course"
- **hobbyist** — "a hobbyist programmer who has been coding casually for about a year"
- **professional** — "a professional software engineer with about five years of
  industry experience"
- **expert** — "a staff-level software engineer with fifteen-plus years of experience
  maintaining large systems"

Channels: in `system`, the system prompt becomes "You are a helpful AI assistant. The
user is {level}." (third person, operator-set profile). In `message`, the user turn is
prefixed "A bit about me: I am {level}." followed by the unmodified task (first-person
self-disclosure). The control uses the base system prompt and the bare task.

**Byte-identity control.** The task text is byte-identical across all nine cells; only
the credential sentence differs (matched-guise logic; Hofmann et al., 2024). This is
asserted automatically by the runner's `--dry-run` check before collection. No
condition varies the user's writing competence — no typos, no informal register — so
the only route from condition to output is the stated credential itself.

### 3.3 Task battery

24 Python tasks (`experiment/tasks.json`), 8 per difficulty tier:

- **Tier A** — entry-level, MBPP-style (Austin et al., 2021): e.g., running totals,
  letter frequencies, clamping, snake_case→camelCase.
- **Tier B** — intermediate: duration-string parsing, anagram grouping, query-string
  parsing, business-day arithmetic, interval merging.
- **Tier C** — hard/realistic: lexicographic topological sort, sliding-window rate
  limiter, safe path resolution, parameterized SQL builder, template renderer,
  longest-increasing-subsequence under a performance budget, config parser, arithmetic
  expression evaluator without `eval`.

Every task ships a hidden suite of 15–21 assert-style tests — **414 tests in total**
including robustness cases — each validated against a reference solution both
in-process and through the real sandbox (`score.py --verify-tasks`: 24/24 pass).
Weak-test masking of condition differences (Liu et al., 2023) is mitigated by suite
size and edge/boundary/invalid coverage. Eight tasks form a **hardening subset** with
CWE tags and targeted failing-input tests, following the scenario methodology of
Pearce et al. (2022): CWE-20 (input validation: chunking, duration parsing, business
days, template rendering, config parsing), CWE-22 (path traversal: safe path
resolution), CWE-89 (SQL injection: parameterized query builder), and CWE-95 (code
injection: expression evaluator without `eval`). Task prompts are register-neutral —
no "simple"/"advanced" vocabulary, no skill-revealing cues — and leave docstrings,
type hints, validation, and error handling optional unless the behavior itself is the
task, so that adding or omitting them is a free choice the condition can move. Tests
check behavior only, never style. Most tasks were freshly written for this study; a
few reuse well-known task shapes (LIS, anagram grouping) with new wording, so
contamination (Jain et al., 2024) is mitigated rather than eliminated. Hidden per-task
checklists of 3–6 atomic requirements (never shown to the subject model) make
`requirements_met` judgeable against a fixed list.

### 3.4 Procedure

Requests are built as the full 648-cell matrix and executed in **shuffled order (seed
20261014, recorded in every response record)** so rate-limit pacing and temporal drift
cannot alias with condition. Sampling: temperature 0.7 and `max_tokens` 2,400,
recorded per response along with the resolved model ID and stop reason. The subject
backend for this run is the Groq free tier running `openai/gpt-oss-20b`
(reasoning effort low; `<think>` blocks stripped before measurement); the model ID is
resolved against the account's live catalog at startup. Every success is checkpointed
to `out/responses_skill.jsonl`; errors are logged and retried on rerun; five
consecutive failures abort the run. Scoring is local and deterministic: each response's
extracted code runs in an isolated interpreter (`python -I`, empty environment, closed
stdin, temp working directory, 5-second per-test alarm, overall subprocess timeout), so
model-generated code cannot reach the network or the collection machinery.

**Pre-registered extraction rule** (frozen before collection): the first fenced block
labeled `python`/`py`/`python3`; else the first unlabeled fenced block that compiles as
Python; else `extraction_fail` — a recorded, analyzed outcome (counted as a correctness
failure, with a sensitivity analysis excluding such rows), never silently dropped.

### 3.5 Measures

One row per response in `out/scores_skill.csv`; primary metric per hypothesis in bold.

| Construct (hypothesis) | Metrics | Instrument |
|---|---|---|
| Style (H1) | **`comment_ratio`** = (comment + docstring lines) / code lines; `prose_words`, `docstring_lines`, `md_headers`, `md_bullets`, `identifier_len_mean`, `response_chars`, `output_tokens` | static/regex |
| Complexity (H2) | **`cyclomatic_mean`** (stdlib-AST decision-point proxy per function); `cyclomatic_total`, `max_nesting`, `advanced_idioms` (comprehensions, lambdas, decorators, generators, walrus, type hints), `loc`, `n_functions` | AST |
| Correctness (H3) | **`pass_rate`** (fraction of hidden tests passed); `all_pass`, `extraction`, `exec_status` | sandboxed runner |
| Robustness (H4) | **`validation_checks`** (AST count of `try`/`raise`/`isinstance`/`issubclass`); `robust_rate` (hardening subset: clean `ValueError`/`TypeError` on invalid input); `security_flags` (stdlib "Bandit-lite" count of `eval`/`exec`/`__import__`/`os.system`/`pickle.load`/`shell=True`) | AST + targeted tests |
| Substance (H5) | **`requirements_met`** vs. hidden checklist | blind judge |
| Channel (H6) | `channel` × all of the above | design |
| Default (H7) | control vs. level cells on H1/H2 primaries | design |
| Tone (H8) | `condescension` (1–5), `assumed_knowledge` (1–5), hedging/refusal flags | blind judge |
| Manipulation check | `leakage` (response prose mentions the stated level) | regex |

The robustness criterion is deliberately strict and frozen: on a hardening task's
invalid input, the implementation must raise `ValueError` or `TypeError` as an
intentional validation signal; garbage returns, hangs, or incidental exceptions fail.
A known measurement noise — unguarded code that *incidentally* raises `ValueError`
(e.g., `range(..., 0)`, `date.fromisoformat`) — is documented in the methods plan and
is why AST-based `validation_checks` is H4's primary and `robust_rate` supporting.

### 3.6 Blind-judge protocol

One judge request per response (648) on `openai/gpt-oss-120b` (judge ≠ subject model).
The judge sees **only** the bare task text, the hidden requirements checklist, and the
response — never the persona or channel — and is instructed to ignore any audience
references in the response (the v1 protocol). It returns JSON: per-requirement
booleans (`requirements_met`), `condescension` 1–5, `assumed_knowledge` 1–5, and
hedging/refusal flags. Responses whose `leakage` flag fires get a pre-registered
robustness re-analysis excluding them; a 10% stratified sample is hand-validated, with
agreement reported.

### 3.7 Pre-registered hypotheses, tests, and margins

The confirmatory family is {H1…H7}, one primary metric each, Holm-adjusted at
α = .05; H8 and all secondary metrics are exploratory. Primary inference uses
task-level means (24 tasks, each averaging 3 replicates) in a mixed model
`metric ~ level × channel + (1 | task)` with paired nonparametric checks
(Jonckheere–Terpstra / Page trend tests within task; Wilcoxon pairwise vs. control).

- **H1 (style).** `comment_ratio` and explanation-to-code prose decrease monotonically
  with stated skill (beginner > hobbyist > professional > expert). Falsified if the
  level-coefficient CIs include 0 for both primary style metrics.
- **H2 (code complexity).** `cyclomatic_mean` and advanced-idiom use increase with
  stated skill. Falsified if no monotone trend or opposite sign.
- **H3 (correctness — the equity primary).** Hidden-test `pass_rate` is not equal
  across levels, directionally beginner < expert (per Gupta et al., 2024, and
  Poole-Dayan et al., 2024), against the competing null from Wang et al. (2025). Both
  outcomes are informative; the null verdict is pre-registered as |beginner − expert|
  gap < 3 percentage points with the 95% CI inside a **±6 pp equivalence margin**.
- **H4 (rationing of robustness).** Beginner-stated conditions receive less input
  validation / error handling / security hardening (`validation_checks`,
  `security_flags`) than expert-stated ones, independent of correctness; two-sided
  test with the stated directional prediction.
- **H5 (scaffolding, not rationing).** Style shifts occur without a matching drop in
  judge-rated `requirements_met`: the beginner-vs-expert difference lies within a
  **±5 pp equivalence margin**. Failure of H5 with H1 true is the paper's "rationing"
  finding.
- **H6 (channel).** Self-disclosure (first person) produces larger style shifts than
  the system-prompt profile (|β_message| > |β_system|); the competing anchoring
  prediction (Reusens et al., 2024) — that self-disclosure also carries a larger
  *quality* penalty — is tested separately as H3 × channel. H6 is flagged
  underpowered (interaction detectable only at roughly d_z ≈ 0.8) and is reported
  with CIs rather than a binary verdict.
- **H7 (default assumption).** On H1/H2 primaries, the no-information control lies
  closer to the professional condition than to the beginner condition (the model
  assumes a competent user by default — the pattern the v1 pilot showed for education).
  Falsified if control is closer to beginner/hobbyist.
- **H8 (condescension; exploratory).** Blind-judge condescension is higher for
  beginner than expert.

**Power.** With 24 task means per contrast, paired two-sided tests at α = .05 reach
0.80 power at d_z ≈ 0.60 — conservative for style effects, which prior persona work
finds large (Hu & Collier, 2024), but binding for correctness: with ~72 binary
observations per cell around a mid-range base rate, a pooled beginner-vs-expert gap of
roughly 12–15 pp is the realistic detection floor. Gaps below that are reported as
*inconclusive*, not "no effect"; the ±6 pp equivalence verdict requires a
correspondingly tight CI and was pre-stated as possibly unmet.

**Pipeline validation.** Before collection resumed, a full-scale synthetic dress
rehearsal (648 randomly generated score rows plus 648 synthetic judge entries, model
field `TOY-SYNTHETIC`) verified end-to-end that the scoring schema, the pre-registered
statistics script, and the findings template execute and correctly render all three
verdict types (significant, equivalence-met, and inconclusive); its numbers are
synthetic by construction and appear nowhere in this paper's results.

### 3.8 Backends and the Claude question

The pipeline is backend-agnostic (Groq, local OpenAI-compatible, Hugging Face,
Anthropic Message Batches). The study's motivating target is Claude (gap G6), but the
account held no Anthropic API credits at collection time, so the confirmatory run uses
`openai/gpt-oss-20b` on the Groq free tier. **Accordingly, every finding in this paper
characterizes `openai/gpt-oss-20b`, not Claude**; a Claude rerun is a single-flag
change (`--backend anthropic`) under the identical pre-registration, and the title
claim should be read with that substitution until such a rerun exists.

### 3.9 Deviations and decision log

Deviations from the stage 2–3 design, decided and logged at stage 4 before collection:

1. **All 24 tasks are Python**, deviating from the designed 16 Python / 4 JavaScript /
   4 other mix. Rationale: one sandbox covers correctness *and* the full static-metric
   battery for every task; polyglot runners would add runtime dependencies and give
   correctness-only scoring for a third of the battery. Cost: cross-language
   generalization is lost (see Limitations).
2. **The 0.3–0.9 control-pass-rate solvability screen moved from a pre-run pilot to a
   stage-5/6 sensitivity analysis** (primary models rerun excluding at-ceiling/at-floor
   tasks) because screening requires model calls and replacement loops conflict with a
   pre-registered battery on a free tier.
3. **Self-repair probe and wording-robustness check skipped** in this run (both were
   designed as optional, under-budget extras); future work.
4. **Cyclomatic complexity is a stdlib-AST proxy** (decision-point counting) rather
   than Radon, and `security_flags` is a stdlib "Bandit-lite" rather than Bandit,
   keeping the dependency footprint minimal; the proxies count the standard decision
   points and the Bandit-class calls relevant to these tasks.
5. **Judge input extended**: the design specified a judge seeing only task text and
   response; the implemented protocol additionally provides the hidden requirements
   checklist (necessary for checklist-based `requirements_met`). The judge remains
   blind to persona and channel.
6. **Test strengthening** is implemented as large hand-built suites validated against
   reference solutions (15–21 cases/task) rather than the design's "EvalPlus-style
   differential/mutation augmentation"; suite size and invalid-input coverage are the
   operative safeguards.
7. **Temperature fixed at 0.7** and recorded per response (the design said "default
   sampling temperature"; the pipeline default is now explicit and logged).
8. **Judge and subject share a model family** (gpt-oss-120b vs. gpt-oss-20b). The
   design asked for judge ≠ subject "where possible"; on the available free tier the
   different-scale sibling was the strongest available non-identical judge. Family-
   shared blind spots are acknowledged in Limitations.

---

## 4. Results

Contrast convention: beginner − expert, paired within task (n = 24 tasks unless noted),
bootstrap 95% CI (5,000 resamples). The confirmatory family (H1, H2, H3, H4, H6, H7) is
Holm-corrected; H5 is confirmatory but could not be run (judge scores incomplete).

### 4.1 Sample, manipulation checks, and data quality

All 648 planned responses were collected (2026-10-14 to 2026-10-16) from `openai/gpt-oss-20b`
(temperature 0.7, max_tokens 2,400, `reasoning_effort: low`): 72 per condition across the nine
conditions, 0 duplicate IDs, 0 errored requests, 0 truncated responses. Code was extracted
from explicitly `python`-labeled fences in 648/648 responses; 634 executed successfully and
14 raised a code error (scored as failing tests). Data collection was slowed by free-tier
quota pacing and two stalls later traced to host idle-sleep; neither affected the data
(`research/04_collection_log.md`). The overall mean pass rate was 0.9645, confirming the
ceiling risk flagged in the interim check. `security_flags` was zero for all responses; the
flag detector was verified on a synthetic positive. `comment_ratio` is (comment + docstring
lines) ÷ code lines, range [0, ∞), so values above 1 are expected for this model's
docstring-heavy style (condition means 3.9–5.2).

Persona leakage (a response mentioning the stated level) was 4.2% for beginner/system, 9.7%
for beginner/message, 9.7% for hobbyist/system, 1.4% for hobbyist/message, and 0% for the
control, professional, and expert cells. Judge input was condition-blind. Human validation
of a judge subsample has not been done.

### 4.2 Confirmatory hypothesis tests

Table 1 gives per-condition means; Table 2 the contrasts.

**Table 1.** Per-condition means of automated metrics (n = 72 per condition).

| condition | pass_rate | comment_ratio | prose_words | cyclomatic_mean | advanced_idioms | validation_checks |
|---|---:|---:|---:|---:|---:|---:|
| control/none | 0.975 | 4.439 | 125.264 | 4.786 | 3.306 | 1.153 |
| beginner/system | 0.986 | 4.894 | 134.778 | 4.470 | 2.722 | 1.014 |
| beginner/message | 0.944 | 3.888 | 157.167 | 4.621 | 2.493 | 0.831 |
| hobbyist/system | 0.985 | 4.510 | 129.236 | 4.514 | 3.264 | 1.000 |
| hobbyist/message | 0.963 | 4.512 | 123.500 | 4.561 | 3.111 | 1.167 |
| professional/system | 0.973 | 4.238 | 134.625 | 4.441 | 3.887 | 1.042 |
| professional/message | 0.953 | 4.066 | 120.472 | 4.525 | 3.500 | 1.056 |
| expert/system | 0.948 | 5.243 | 129.472 | 4.745 | 3.750 | 1.111 |
| expert/message | 0.954 | 4.567 | 115.694 | 4.581 | 3.694 | 1.181 |

- **H1 (style) — not supported.** `comment_ratio` b − e = −0.54 [−1.64, +0.33], exact
  Wilcoxon p = 0.584, Holm p = 1.0; Friedman p = 0.604; monotone-trend p = 0.839.
- **H2 (complexity) — not supported.** `cyclomatic_mean` b − e = −0.11 [−0.58, +0.26],
  Holm p = 1.0; Friedman p = 0.996; trend p = 0.444.
- **H3 (correctness, equity primary) — null, equivalence met.** `pass_rate` b − e = +0.014
  [−0.023, +0.056], p = 0.625, Holm p = 1.0. The pre-registered equivalence criterion
  (|gap| < 3 pp and 95% CI within ±6 pp) was **met**. Caveat: at a 0.9645 mean pass rate
  this establishes equivalence only where the model nearly always succeeds. In the three
  tasks with control pass rate below 0.95, the contrast is +0.047 [−0.18, +0.32], which is
  uninformative. Exploratory `all_pass`: +0.035 [−0.021, +0.097], p = 0.305.
- **H4 (robustness) — not supported (underpowered).** `validation_checks` b − e = −0.175
  [−0.404, +0.028], p = 0.139, Holm p = 0.696. The direction (beginner < expert) matches
  the rationing prediction, but the CI includes zero. Security flags: 0 in every condition.
  Handling of invalid inputs on the 8 hardening tasks: +0.094 [−0.017, +0.224], exact
  p = 0.156.
- **H5 (substance preserved) — [PROVISIONAL].** The confirmatory test was skipped by
  `stats.py`; on the 225 judged responses (16 tasks), `requirements_met` b − e = +0.0005
  [−0.024, +0.025], equivalence met (`research/05b_ablations.md`).
- **H6 (channel) — not supported.** Interaction on `comment_ratio` (message − system
  contrast of b − e) = −0.38 [−2.51, +1.57], p = 0.491, Holm p = 1.0. On `pass_rate`
  (exploratory): −0.048 [−0.130, +0.031], p = 0.374.
- **H7 (default assumption) — not supported after correction.**
  |control − professional| − |control − beginner| on `comment_ratio` = +0.69 [+0.16, +1.43],
  raw p = 0.027, Holm p = 0.162. On `cyclomatic_mean` (exploratory): +0.36 [+0.01, +0.81],
  p = 0.212.

### 4.3 Exploratory analyses

- **Prose words (H1s).** Beginner − expert = **+23.4 words** [+8.3, +43.2], exact
  p = 0.0035; Friedman p = 0.029; trend p = 0.017. The effect is concentrated in the
  self-disclosure channel (+41.5 [+15.9, +75.4], p = 0.0005) and absent under the system
  prompt (+5.3 [−16.5, +26.2], p = 0.577). It is strongest in Tier C (+50.0 [+10.8, +101.2],
  p = 0.039) and unchanged with persona-leaking responses excluded (+24.1, p = 0.006).
  It lost uncorrected significance in several subsets (Tiers A and B, the 3 below-ceiling
  tasks, replicate 3), so it is the less robust of the two form effects; its p-value is also
  borderline against a Bonferroni threshold of about 0.0036 over roughly 14 tests.
- **Advanced idioms (H2s).** Beginner − expert = **−1.06** [−1.47, −0.66], p = 0.0002;
  Friedman p = 0.0002; trend p = 3.8×10⁻⁵. The CI excludes zero in every ablation slice
  with n ≥ 8 tasks: each tier, both channels, all three replicates (−0.92 to −1.35), and
  with leaking responses excluded (−1.09). Experts averaged about 3.7 idioms per response
  versus 2.5–2.7 for beginners.
- **H8 (tone) — [PROVISIONAL].** Condescension (1–5) b − e = +0.13 [−0.06, +0.31],
  p = 0.251 on 16 tasks. Self-disclosure channel only: +0.46 [+0.19, +0.74], p = 0.055
  (uncorrected); system-prompt channel: 0.00. Inconclusive pending the full judge pass.
- **Ceiling sensitivity.** Pass-rate contrasts by tier: A −0.021, B +0.023, C +0.040; none
  meets the equivalence criterion on its own. The largest single-slice gap is +0.074
  (replicate 2, uncorrected, beginner higher).
- **Judge validation.** Human-judge agreement has not been measured.

### 4.4 Effect-size summary

**Table 2.** Beginner − expert contrasts (paired within task, bootstrap 95% CI).

| Metric | Hypothesis | n tasks | Effect | 95% CI | Status |
|---|---|---:|---:|---|---|
| comment_ratio | H1 | 24 | −0.538 | [−1.644, +0.328] | confirmatory, ns |
| cyclomatic_mean | H2 | 24 | −0.109 | [−0.582, +0.258] | confirmatory, ns |
| pass_rate | H3 | 24 | +0.014 | [−0.023, +0.056] | confirmatory, equivalence met |
| validation_checks | H4 | 24 | −0.175 | [−0.404, +0.028] | confirmatory, ns |
| requirements_met | H5 | 16 | +0.0005 | [−0.024, +0.025] | provisional |
| prose_words | H1s | 24 | +23.39 | [+8.32, +43.21] | exploratory |
| advanced_idioms | H2s | 24 | −1.058 | [−1.464, −0.656] | exploratory |
| condescension | H8 | 16 | +0.130 | [−0.057, +0.313] | provisional, exploratory |

---

## 5. Discussion

> **Placeholder section** except §5.2. Interpretation bullets below are structural
> stubs tied to each hypothesis, to be written from the completed analysis.

### 5.1 Interpretation

**Tailoring of form, not rationing of substance.** For this model, a stated skill level
changed how the answer was presented, with beginners getting more surrounding prose and
fewer advanced idioms, but not whether it worked. Correctness was equivalent within the
pre-registered margin (H3: +1.4 pp [−2.3, +5.6]) and, on partial data, so were judged
requirements (H5, provisional). Cyclomatic complexity was flat, so there is no sign of
experts receiving over-engineered code. Inline documentation density was flat too; the
beginner accommodation went into prose and idiom choice instead.

**Relation to prior work.** The form-level shifts are consistent with persona and
attribute effects that are real but modest (Salewski et al., 2023; Hu & Collier, 2024).
The study does not reproduce the correctness penalties for lower-status or lower-proficiency
users reported by Poole-Dayan et al. (2024) and Reusens et al. (2024). One reading is that
those penalties depend on weaker, implicit signals (language proficiency, schooling) and on
question answering, and do not appear for an explicit skill statement on code with
executable tests. This is a hypothesis, not something the data test. It is also consistent
with the null for expertise framing in Wang et al. (2025), though that comparison concerns a
different task family. The non-idiom metrics varied across tiers and replicates, echoing
the "largely random" persona effects of Zheng, Pei, et al. (2024).

**H4 and the possible silent drop.** Beginners received about 0.18 fewer validation checks
(about 15% of the control mean of 1.15), the direction rationing would predict, but the
interval includes zero and no security flags occurred anywhere. The study cannot say
whether hardening is dropped for beginners; it can only say a large drop is unlikely at this
battery's resolution.

**H6 and H7.** The channel interaction on the primary was null. Exploratory slices suggest
that the prose and tone shifts live in the self-disclosure channel, which would fit a model
reacting more to a user's first-person statement than to an operator description. This is a
hypothesis for an adequately powered follow-up. For H7, the control sat closer to the
professional end on `comment_ratio` (+0.69), but the result does not survive Holm correction,
so no claim is made about whom the model writes for by default.

**H8 and tone.** The self-disclosure condescension difference (+0.46 on a 1–5 scale) is
small-to-moderate and uncorrected, on 9 tasks. Whether polite rationing or overt
condescension is occurring, in the sense of Hofmann et al. (2024), awaits the full judge pass.

**Equity reading.** No group was served materially worse code. The largest pass-rate gap in
any ablation slice favored beginners (+7.4 pp, replicate 2, uncorrected). Two caveats
temper "no harm". First, beginners saw a different idiom mix; whether that aids readability
or reduces exposure to modern idioms is unmeasured. Second, equivalence was shown at the
ceiling, so the battery could not detect rationing on tasks the model always solves.
Given asymmetric risks for novices who cannot audit generated code (Moradi Dakhel et al.,
2023; Bo et al., 2026; Peng et al., 2023), a harder battery is the key next test.

### 5.2 Limitations

1. **Persona ≠ person.** The manipulation isolates the causal effect of a *stated
   credential* under matched-guise control; real users also leak skill through
   phrasing, vocabulary, and error patterns (Rojpaisarnkit et al., 2025), so the total
   real-world adaptation is understated by design.
2. **Single subject model — and not Claude.** All findings characterize
   `openai/gpt-oss-20b` (Groq free tier). The study's motivating target, Claude,
   remains unmeasured here; the identical pre-registration reruns with
   `--backend anthropic` once credits exist, and no claim in this paper transfers to
   Claude or to any other model without that rerun (persona effects are known to be
   model-heterogeneous; Zheng, Pei, et al., 2024).
3. **Python-only battery.** The implemented battery dropped the designed JS/SQL/other
   tasks (deviation 1), so cross-language generalization — plausibly different, since
   style norms and hardening idioms differ by ecosystem — is out of scope.
4. **Ceiling effects.** The full-sample mean pass rate (0.9645) sits above the design's
   intended 0.3–0.9 solvability band, so H3's equivalence verdict is informative mainly
   where the model nearly always succeeds; only 3 tasks fall below 0.95 in the control. Weak-test masking
   runs in the same direction (Liu et al., 2023).
5. **LLM judge.** `requirements_met` and all tone constructs (provisional here: 225/648 responses judged) come from an LLM judge
   that, while condition-blind, has not yet been human-validated and shares a model family with the
   subject (gpt-oss-120b judging gpt-oss-20b); family-shared blind spots and
   style-sensitivity biases cannot be fully excluded.
6. **Single-turn, single-attempt.** Real assistant use is conversational; multi-turn
   drift and the self-repair loop (Jain et al., 2024) are untested here (the designed
   self-repair probe was deferred).
7. **Measurement proxies.** Cyclomatic complexity and security flags are stdlib
   proxies rather than Radon/Bandit; the robustness criterion is known to be inflatable
   by incidental `ValueError`s on two tasks (documented pre-collection); the
   `comment_ratio` and `security_flags` audits were resolved (§4.1).
8. **Contamination mitigated, not eliminated.** Some task shapes are classical even
   where wording is fresh; memorized solutions would bias toward ceiling and against
   finding condition effects.

### 5.3 Future work

In priority order: (1) complete the judge pass and re-run H5 and H8; (2) rerun the frozen
pre-registration on Claude (`--backend anthropic`); (3) use a harder battery targeting
0.5–0.8 baseline pass rates so rationing can show up in correctness; (4) add a cross-family
judge and a human audit of about 50 responses; (5) run an adequately powered follow-up on
the self-disclosure versus system-prompt channel difference; (6) test naturally written
novice prompts against synthetic personas. The deferred self-repair probe, multi-turn
extension, wording-robustness check, and polyglot battery remain open.

---

## 6. Conclusion

For `openai/gpt-oss-20b`, a stated programming skill level changes how code is presented
but not whether it is correct. Beginner-framed prompts received about 23 more words of prose
and about one fewer advanced idiom per response than expert-framed prompts, while hidden-test
correctness was equivalent within the pre-registered margin (+1.4 pp, CI [−2.3, +5.6]),
albeit at a ceiling. The pre-registered primaries for documentation density, complexity,
robustness, channel interaction, and default-user position were not significant after
correction. On a partial judge sample, substance was preserved and tone effects were
inconclusive. These results should not be generalized to Claude or other models without the
planned replication and the completion of the judge pass.

---

## References

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai,
C., Terry, M., Le, Q., & Sutton, C. (2021). *Program synthesis with large language
models*. arXiv. https://arxiv.org/abs/2108.07732

Barke, S., James, M. B., & Polikarpova, N. (2023). Grounded Copilot: How programmers
interact with code-generating models. *Proceedings of the ACM on Programming
Languages, 7*(OOPSLA1). https://arxiv.org/abs/2206.15000

Bo, J. Y., Kazemitabaar, M., Deng, M., Inzlicht, M., & Anderson, A. (2026). Invisible
saboteurs: Sycophantic LLMs mislead novices in problem-solving tasks. *Proceedings of
the CHI Conference on Human Factors in Computing Systems (CHI 2026)*. ACM.
https://arxiv.org/abs/2510.03667

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H.,
Burda, Y., Joseph, N., Brockman, G., et al. (2021). *Evaluating large language models
trained on code*. arXiv. https://arxiv.org/abs/2107.03374

Gupta, S., Shrivastava, V., Deshpande, A., Kalyan, A., Clark, P., Sabharwal, A., &
Khot, T. (2024). Bias runs deep: Implicit reasoning biases in persona-assigned LLMs.
*International Conference on Learning Representations (ICLR 2024)*.
https://arxiv.org/abs/2311.04892

Hofmann, V., Kalluri, P. R., Jurafsky, D., & King, S. (2024). Dialect prejudice
predicts AI decisions about people's character, employability, and criminality.
*Nature*. https://arxiv.org/abs/2403.00742

Hu, T., & Collier, N. (2024). Quantifying the persona effect in LLM simulations.
*Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (ACL 2024)*. https://arxiv.org/abs/2402.10811

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A.,
Sen, K., & Stoica, I. (2024). *LiveCodeBench: Holistic and contamination free
evaluation of large language models for code*. arXiv. https://arxiv.org/abs/2403.07974

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K.
(2024). SWE-bench: Can language models resolve real-world GitHub issues?
*International Conference on Learning Representations (ICLR 2024)*.
https://arxiv.org/abs/2310.06770

Kazemitabaar, M., Chow, J., Ma, C. K. T., Ericson, B. J., Weintrop, D., & Grossman, T.
(2023). Studying the effect of AI code generators on supporting novice learners in
introductory programming. *Proceedings of the CHI Conference on Human Factors in
Computing Systems (CHI 2023)*. ACM. https://arxiv.org/abs/2302.07427

Liu, J., Xia, C. S., Wang, Y., & Zhang, L. (2023). Is your code generated by ChatGPT
really correct? Rigorous evaluation of large language models for code generation.
*Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.
https://arxiv.org/abs/2305.01210

Moradi Dakhel, A., Majdinasab, V., Nikanjam, A., Khomh, F., Desmarais, M. C., & Jiang,
Z. M. (2023). GitHub Copilot AI pair programmer: Asset or liability? *Journal of
Systems and Software*. https://arxiv.org/abs/2206.15331

Nguyen, S., Babe, H. M., Zi, Y., Guha, A., Anderson, C. J., & Feldman, M. Q. (2024).
How beginning programmers and code LLMs (mis)read each other. *Proceedings of the CHI
Conference on Human Factors in Computing Systems (CHI 2024)*. ACM.
https://arxiv.org/abs/2401.15232

Pearce, H., Ahmad, B., Tan, B., Dolan-Gavitt, B., & Karri, R. (2022). Asleep at the
keyboard? Assessing the security of GitHub Copilot's code contributions. *2022 IEEE
Symposium on Security and Privacy*. https://arxiv.org/abs/2108.09293

Peng, S., Kalliamvakou, E., Cihon, P., & Demirer, M. (2023). *The impact of AI on
developer productivity: Evidence from GitHub Copilot*. arXiv.
https://arxiv.org/abs/2302.06590

Perez, E., Ringer, S., Lukošiūtė, K., Nguyen, K., Chen, E., et al. (2022).
*Discovering language model behaviors with model-written evaluations*. arXiv.
https://arxiv.org/abs/2212.09251

Poole-Dayan, E., Roy, D., & Kabbara, J. (2024). *LLM targeted underperformance
disproportionately impacts vulnerable users*. arXiv. https://arxiv.org/abs/2406.17737

Prather, J., Reeves, B. N., Denny, P., Becker, B. A., Leinonen, J., Luxton-Reilly, A.,
Powell, G., Finnie-Ansley, J., & Santos, E. A. (2023). "It's weird that it knows what
I want": Usability and interactions with Copilot for novice programmers. *ACM
Transactions on Computer-Human Interaction*. https://arxiv.org/abs/2304.02491

Reusens, M., Borchert, P., De Weerdt, J., & Baesens, B. (2024). Native design bias:
Studying the impact of English nativeness on language model performance. *Findings of
IJCNLP–AACL*. https://arxiv.org/abs/2406.17385

Rojpaisarnkit, R., Fan, Y., Matsumoto, K., & Kula, R. G. (2025). How natural language
proficiency shapes GenAI code for software engineering tasks. *IEEE Software*.
https://doi.org/10.1109/MS.2025.3622690 (arXiv: https://arxiv.org/abs/2511.04115)

Salewski, L., Alaniz, S., Rio-Torto, I., Schulz, E., & Akata, Z. (2023). In-context
impersonation reveals large language models' strengths and biases. *Advances in Neural
Information Processing Systems 36 (NeurIPS 2023)*.
https://proceedings.neurips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract.html

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., et al.
(2024). Towards understanding sycophancy in language models. *International Conference
on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2310.13548

Vaithilingam, P., Zhang, T., & Glassman, E. L. (2022). Expectation vs. experience:
Evaluating the usability of code generation tools powered by large language models.
*CHI Conference on Human Factors in Computing Systems Extended Abstracts (CHI EA
'22)*. ACM. https://doi.org/10.1145/3491101.3519665

Wang, K., Li, J., Yang, S., Zhang, Z., & Wang, D. (2025). *When truth is overridden:
Uncovering the internal origins of sycophancy in large language models*. arXiv.
https://arxiv.org/abs/2508.02087

Zheng, J., Cao, B., Ma, Z., Pan, R., Lin, H., Lu, Y., Han, X., & Sun, L. (2024).
*Beyond correctness: Benchmarking multi-dimensional code generation for large language
models*. arXiv. https://arxiv.org/abs/2407.11470

Zheng, M., Pei, J., Logeswaran, L., Lee, M., & Jurgens, D. (2024). When "a helpful
assistant" is not really helpful: Personas in system prompts do not improve
performance of large language models. *Findings of the Association for Computational
Linguistics: EMNLP 2024*. https://arxiv.org/abs/2311.10054

---

*Provenance note: Introduction §1.3 pilot numbers are from `EXPERIMENT.md` §10 (2026-10-14
entry, 191/810 responses, `openai/gpt-oss-20b`). All Results and Discussion statistics are
from `research/05_analysis_automated_metrics.md`, `research/05b_ablations.md`, and
`research/06_findings.md`; H5 and H8 rest on 225/648 judged responses and must be refreshed
after the full judge pass. The synthetic dress rehearsal (`research/dryrun_conclusions_TOY.md`)
is referenced once, in Method §3.7, as pipeline validation only; none of its numbers appear
here.*
