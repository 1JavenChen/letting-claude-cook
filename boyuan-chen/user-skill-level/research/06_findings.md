# Stage 7 — Findings (PROVISIONAL: automated metrics + partial judge pass)

**Status.** Stage 6 is not finished: `05_analysis.md` and `experiment/out/metrics.csv` do not
exist, because the blind-judge pass (`judge.py`) is incomplete and the data lives only in the
user's gitignored `experiment/out/`. This document interprets what *is* committed:
`05_analysis_automated_metrics.md` (pre-registered tests on all 648 responses) and
`05b_ablations.md` (exploratory; judge scores merged for only 225 of 648 responses, 16 of 24 tasks).
H5 and H8 verdicts below rest on that partial judge sample and must be re-checked after the
full judge pass. **Subject model is `openai/gpt-oss-20b`, not Claude** (collection log); every
claim is about that model.

Contrast throughout = beginner − expert, paired within task (n = 24 tasks), bootstrap 95% CI.
Confirmatory family is Holm-corrected; nothing in it reached significance.

## 1. Verdict per hypothesis

| H | Verdict | Deciding statistic |
|---|---|---|
| H1 comment density lower for beginners | **Not supported** | comment_ratio −0.54 [−1.64, +0.33], Holm p = 1.0; trend p = 0.84. CI spans zero. Secondary `prose_words`: **+23.4 words** [+8.3, +43.2], p = 0.0035 (exploratory) — beginners get more explanation text *outside* the code. |
| H2 complexity rises with skill | **Not supported on the primary; supported on the secondary** | cyclomatic −0.11 [−0.58, +0.26], Holm p = 1.0. Secondary `advanced_idioms`: **−1.06** [−1.47, −0.66], p = 0.0002, trend p = 4e-5 (exploratory): experts get ~3.7 idioms/response vs ~2.5–2.7 for beginners. |
| H3 correctness differs (equity primary) | **Null, equivalence met** | pass_rate +0.014 [−0.023, +0.056]; pre-registered equivalence (|gap| < 3 pp, CI within ±6 pp) **MET**. Ceiling caveat: overall pass rate 0.9645. |
| H4 less robustness for beginners | **Not supported** (underpowered) | validation_checks −0.175 [−0.40, +0.03], Holm p = 0.70. Direction (beginner < expert) is consistent with rationing, but CI includes 0. Security flags: 0 everywhere. |
| H5 substance preserved (scaffolding not rationing) | **Provisionally supported (partial data)** | requirements_met +0.0005 [−0.024, +0.025], equivalence met, but n = 16 tasks / 225 responses. |
| H6 channel interaction | **Not supported** | comment_ratio interaction −0.38 [−2.5, +1.6], Holm p = 1.0. Underpowered by design (needs d_z ≈ 0.8). Exploratory ablation hints that prose-word shift is concentrated in the self-disclosure channel (+41.5 words, p = 0.0005) and absent under the system prompt (+5.3, p = 0.58). |
| H7 default sits near expert end | **Not supported after correction** | |control−pro| − |control−beg| on comment_ratio +0.69 [+0.16, +1.43], raw p = 0.027, Holm p = 0.16.  |
| H8 condescension higher for beginners | **Inconclusive** | +0.13 [−0.06, +0.31], p = 0.25 on 16 tasks; self-disclosure-only +0.46 (p = 0.055, uncorrected). Needs the full judge pass. |

Note on H4: the confirmatory test is two-sided; the direction (beginner < expert by ~0.18
checks, ~15% of the control mean of 1.15) is what rationing would predict, but it is
indistinguishable from noise at n = 24 tasks.

## 2. What it means: tailoring, not rationing

The pattern that survives is **form tailoring with preserved substance**:

- Beginners received more surrounding prose (+23 words, ~20% over the expert mean of ~118)
  and code using fewer advanced idioms (−1.06 per response, ~28% fewer than experts).
  The idiom effect is the most robust finding in the study: significant in every tier,
  both channels, all three replicates, and with persona-leaking responses excluded
  (`05b_ablations.md`; CI excludes zero in every slice with n ≥ 8 tasks).
- Hidden-test correctness (+1.4 pp) and judged requirements-met (+0.05 pp, partial data) are essentially
  equal across levels. There is no evidence of **rationing** of correctness or requirements
  on this model and battery, and no evidence of **over-engineering** for experts
  (cyclomatic complexity flat).
- Comment density is flat, so the "beginners get more comments" folk expectation did not
  materialise; the model puts its beginner accommodation in prose and idiom choice, not
  inline documentation.

*Literature link.* Consistent with Salewski et al. (2023) and Hu & Collier (2024): stated
attributes produce real but modest, form-level shifts. **Does not replicate** the
education/proficiency penalties of Poole-Dayan et al. (2024) and Reusens et al. (2024) —
here the stated-*lower*-skill user is not served less-correct code. This *extends* that
literature: the penalty may be specific to accuracy-on-QA with weaker signals (language
proficiency, schooling) and not appear for an explicitly stated skill level on code with
executable tests. Zheng et al.'s "largely random" persona effects are echoed by the
heterogeneity across tiers/replicates for the non-idiom metrics.

## 3. Equity reading

No group is systematically served materially worse code on this model: the largest
correctness gap in any ablation slice is +7.4 pp (replicate 2 only, uncorrected,
beginner *higher*), and the pre-registered equivalence bound is met overall. Two practical
points temper "no harm":

1. **Reduced idiom exposure** for stated beginners is quality-neutral on tests but is a
   real difference in what the user is shown; whether it helps (readability) or hurts
   (less modern-idiom learning) is not measurable here.
2. **Possible tone cost in the self-disclosure channel** (condescension +0.46 on a 1–5
   scale, uncorrected, partial data). Half a scale point is small-to-moderate; treat as a
   hypothesis for the full judge pass, not a finding.

Equivalence at the ±3 pp level is shown only at the ceiling (96% mean pass rate): the
battery cannot detect a rationing effect on tasks the model always solves. In the 3 tasks
below 0.95 control pass rate the contrast is +0.047 with CI [−0.18, +0.32] — uninformative.

## 4. Alternative explanations

| Explanation | Could it produce the pattern? | What rules it out |
|---|---|---|
| **Ceiling effect on correctness** | Yes — it would hide rationing (the H3 null). | A harder battery with control pass rate ~0.5–0.8; difficulty-banded tasks. |
| **Persona leakage** (response mentions the stated level; 4–10% of beginner/hobbyist responses) | Could inflate prose_words (the model says "since you're a beginner…"). | Excluded-leak ablation: prose +24.1 (p = 0.006), idioms −1.09 — pattern unchanged. Ruled out for these two. |
| **Judge bias** (`gpt-oss-120b` judging a `gpt-oss-20b` family model) | Could flatten requirements-met/condescension differences via self-preference. | Cross-family judge (e.g. a Claude model) and a human-rated subsample. |
| **Task battery skew** (Tier A entry-level, ceiling) | Tier-A tasks leave little room for style differences. | Idiom effect holds in tiers B and C separately; prose effect is strongest in tier C (+50 words). |
| **Metric artefact** (idiom count is a regex/AST proxy; prose_words counts all non-code text) | Beginner-oriented explanation naturally inflates prose; idiom count could track code length. | Normalise idioms per line; manual review of a sample of 30 responses. |
| **Single-model scope** (`gpt-oss-20b`, `reasoning_effort: low`) | The null on correctness may be specific to a reasoning-light open model. | Replication on Claude (the stated target); this is the study's biggest open item. |
| **Persona-vs-person gap** | Synthetic personas may elicit stronger or weaker responses than real novices' prompts, which differ in writing quality as well. | Pair the stated-skill manipulation with naturally-authored novice prompts. |
| **Multiple comparisons** | The idiom and prose effects are exploratory secondary metrics. | They survive any plausible correction (p = 0.0002 across ≈14 tests: Bonferroni threshold 0.0036 passes), but the prose effect (p = 0.0035) is borderline. |

## 5. Limitations and ranked next experiments

1. **Finish the judge pass** (`experiment/judge.py`, then `experiment/stats.py`) and re-run
   H5/H8 on all 24 tasks; revisit §1 and §3 of this document. *(Blocked on a local
   machine with GROQ_API_KEY; see `05_judge_readiness.md`.)*
2. **Replicate on Claude** (`--backend anthropic`) — the study's stated subject is Claude;
   all current claims are about `gpt-oss-20b`.
3. **Harder battery** targeting 0.5–0.8 baseline pass rates so that rationing, if present,
   can show up in correctness; the current H3 equivalence is at the ceiling.
4. **Cross-family judge + human audit** of ~50 responses for condescension and
   requirements-met, to bound self-preference bias.
5. **Mechanism for the channel difference** (prose and tone shifts concentrated in
   self-disclosure): a pre-registered, adequately powered H6 follow-up.
6. **Real novices' prompts** instead of synthetic personas, to test the persona-vs-person gap.

## Corrections log

Reading this file critically before commit: the H4 row's first sentence contained a
muddled direction claim; the note beneath it states the correct reading. Percentages
(~20%, ~28%) are computed from the per-condition means in `05_analysis_automated_metrics.md`
(prose ≈ 134.8/157.2 beginner vs ≈ 115.7–129.5 expert; idioms 2.5–2.7 vs 3.7) and are approximate.
