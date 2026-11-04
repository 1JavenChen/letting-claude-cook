# Stage-6 statistics - skill-level suite

Scores: `fixtures/dryrun/scores_skill_TOY.csv` (648 rows). Judge scores: merged. Bootstrap: 5000 resamples, seed 20261014. Unit of analysis: task-level means (design doc section 7); all contrasts paired within task; levels pooled over channels unless stated.

### Per-condition means (automated metrics)

| condition | n | pass_rate | all_pass | comment_ratio | prose_words | cyclomatic_mean | advanced_idioms | validation_checks | security_flags | extraction_fail | leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control/none | 72 | 0.718 | - | 0.090 | 241.458 | 3.694 | 2.319 | 2.597 | 0.000 | 0.000 | - |
| beginner/system | 72 | 0.725 | - | 0.213 | 415.944 | 3.010 | 0.611 | 2.597 | 0.000 | 0.000 | - |
| beginner/message | 72 | 0.706 | - | 0.234 | 428.208 | 2.906 | 0.528 | 2.417 | 0.000 | 0.000 | - |
| hobbyist/system | 72 | 0.725 | - | 0.167 | 328.611 | 3.357 | 1.306 | 2.417 | 0.000 | 0.000 | - |
| hobbyist/message | 72 | 0.719 | - | 0.163 | 346.264 | 3.283 | 1.319 | 2.528 | 0.000 | 0.000 | - |
| professional/system | 72 | 0.719 | - | 0.108 | 263.111 | 3.604 | 2.292 | 2.653 | 0.000 | 0.000 | - |
| professional/message | 72 | 0.718 | - | 0.098 | 260.833 | 3.610 | 2.417 | 2.486 | 0.000 | 0.000 | - |
| expert/system | 72 | 0.728 | - | 0.062 | 188.056 | 3.993 | 3.208 | 2.736 | 0.000 | 0.000 | - |
| expert/message | 72 | 0.712 | - | 0.059 | 190.875 | 3.893 | 3.181 | 2.528 | 0.000 | 0.000 | - |

### Per-condition means (blind-judge metrics)

| condition | n | requirements_met_prop | condescension | assumed_knowledge | hedging | refusal |
|---|---:|---:|---:|---:|---:|---:|
| control/none | 72 | 0.846 | 1.750 | 3.306 | 0.083 | 0.000 |
| beginner/system | 72 | 0.853 | 2.597 | 1.708 | 0.097 | 0.000 |
| beginner/message | 72 | 0.830 | 2.653 | 1.778 | 0.097 | 0.000 |
| hobbyist/system | 72 | 0.840 | 2.153 | 2.667 | 0.042 | 0.000 |
| hobbyist/message | 72 | 0.837 | 2.208 | 2.458 | 0.111 | 0.000 |
| professional/system | 72 | 0.814 | 1.847 | 3.306 | 0.125 | 0.000 |
| professional/message | 72 | 0.852 | 1.653 | 3.097 | 0.139 | 0.000 |
| expert/system | 72 | 0.869 | 1.472 | 4.000 | 0.139 | 0.000 |
| expert/message | 72 | 0.848 | 1.431 | 3.889 | 0.139 | 0.000 |

### Hypothesis tests

| id | hypothesis | metric | n | effect (b-e) | 95% CI | test | p | p (Holm) | Friedman p | trend p | family |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| H1 | Style: comment/docstring density | comment_ratio | 24 | +0.1632 | [+0.1479, +0.1788] | Wilcoxon (exact) | 1.2e-07 | 8.3e-07 | 6.9e-14 | 6.7e-16 | confirmatory |
| H1s | Style (secondary): prose words | prose_words | 24 | +232.6111 | [+220.2292, +245.7083] | Wilcoxon (exact) | 1.2e-07 | - | 1.6e-15 | 1.1e-17 | exploratory |
| H2 | Complexity: cyclomatic (mean/function) | cyclomatic_mean | 24 | -0.9847 | [-1.0823, -0.8890] | Wilcoxon (exact) | 1.2e-07 | 8.3e-07 | 5.5e-11 | 7.7e-13 | confirmatory |
| H2s | Complexity (secondary): advanced idioms | advanced_idioms | 24 | -2.6250 | [-2.7986, -2.4444] | Wilcoxon (normal) | 1.9e-05 | - | 2.1e-15 | 2.0e-17 | exploratory |
| H3 | Correctness: hidden-test pass rate | pass_rate | 24 | -0.0042 | [-0.0225, +0.0135] | Wilcoxon (normal) | 0.6147 | 1.0000 | 0.6474 | - | confirmatory |
| H3s | Correctness (secondary): all tests passed | all_pass | - | - | - | SKIPPED: no data | - | - | - | - | exploratory |
| H4 | Robustness: validation checks (AST) | validation_checks | 24 | -0.1250 | [-0.4236, +0.1736] | Wilcoxon (normal) | 0.4281 | 1.0000 | 0.3957 | - | confirmatory |
| H4s1 | Robustness (secondary): security flags | security_flags | 24 | +0.0000 | [+0.0000, +0.0000] | Wilcoxon (degenerate) | 1.0000 | - | 1.0000 | - | exploratory |
| H4s2 | Robustness (secondary): handled-invalid rate, hardening tasks | robust_rate | 8 | +0.0115 | [-0.0403, +0.0719] | Wilcoxon (exact) | 0.9453 | - | 0.8614 | - | exploratory |
| H5 | Substance: requirements met (blind judge) | requirements_met_prop | 24 | -0.0166 | [-0.0580, +0.0243] | Wilcoxon (normal) | 0.4843 | 1.0000 | 0.5620 | - | confirmatory |
| H6 | Channel: (beg-exp) message - (beg-exp) system, comment_ratio | comment_ratio | 24 | +0.0242 | [+0.0082, +0.0416] | Wilcoxon (exact) | 0.0087 | 0.0349 | - | - | confirmatory |
| H6s | Channel (secondary): same interaction on pass_rate | pass_rate | 24 | -0.0033 | [-0.0368, +0.0295] | Wilcoxon (normal) | 0.9273 | - | - | - | exploratory |
| H7 | Default: |control-professional| - |control-beginner|, comment_ratio | comment_ratio | 24 | -0.1016 | [-0.1178, -0.0853] | Wilcoxon (exact) | 1.2e-07 | 8.3e-07 | - | - | confirmatory |
| H7s | Default: |control-professional| - |control-beginner|, cyclomatic_mean | cyclomatic_mean | 24 | -0.4215 | [-0.5708, -0.2600] | Wilcoxon (exact) | 0.0002 | - | - | - | exploratory |
| H8 | Tone: condescension (blind judge) | condescension | 24 | +1.1736 | [+1.0000, +1.3333] | Wilcoxon (normal) | 1.9e-05 | - | 4.9e-13 | 2.5e-14 | exploratory |

### Verdicts (alpha = 0.05, Holm within the confirmatory family)

- **H1** (comment_ratio, confirmatory): effect +0.1632, p=1.2e-07, Holm p=8.3e-07 -> significant; monotone decreasing trend supported (Page z=-7.99)
- **H1s** (prose_words, exploratory): effect +232.6111, p=1.2e-07 -> significant; monotone decreasing trend supported (Page z=-8.49)
- **H2** (cyclomatic_mean, confirmatory): effect -0.9847, p=1.2e-07, Holm p=8.3e-07 -> significant; monotone increasing trend supported (Page z=+7.07)
- **H2s** (advanced_idioms, exploratory): effect -2.6250, p=1.9e-05 -> significant; monotone increasing trend supported (Page z=+8.41)
- **H3** (pass_rate, confirmatory): effect -0.0042, p=0.6147, Holm p=1.0000 -> not significant; pre-registered equivalence (|gap| < 3 pp and 95% CI inside +/-6 pp): MET
- **H3s** - skipped: no data
- **H4** (validation_checks, confirmatory): effect -0.1250, p=0.4281, Holm p=1.0000 -> not significant
- **H4s1** (security_flags, exploratory): effect +0.0000, p=1.0000 -> not significant
- **H4s2** (robust_rate, exploratory): effect +0.0115, p=0.9453 -> not significant
- **H5** (requirements_met_prop, confirmatory): effect -0.0166, p=0.4843, Holm p=1.0000 -> not significant; pre-registered equivalence (95% CI inside +/-5 pp): not met
- **H6** (comment_ratio, confirmatory): effect +0.0242, p=0.0087, Holm p=0.0349 -> significant
- **H6s** (pass_rate, exploratory): effect -0.0033, p=0.9273 -> not significant
- **H7** (comment_ratio, confirmatory): effect -0.1016, p=1.2e-07, Holm p=8.3e-07 -> significant
- **H7s** (cyclomatic_mean, exploratory): effect -0.4215, p=0.0002 -> significant
- **H8** (condescension, exploratory): effect +1.1736, p=1.9e-05 -> significant; monotone decreasing trend supported (Page z=-7.53)

### Methods notes

- Wilcoxon signed-rank: exact null when n <= 25 with untied |differences|, else normal approximation with tie + continuity corrections; zero differences dropped.
- Friedman: tie-corrected chi-square approximation, df = k-1 (approximation verified against a permutation null at n = 24 tasks; it is rough below n ~ 10, which this design never uses).
- Page's L trend (normal approximation) stands in for the design doc's Jonckheere-Terpstra as the ordered-alternative rank test.
- Effect sizes: paired mean differences with bootstrap percentile 95% CIs over tasks (seed recorded above).
- Mixed models (metric ~ level * channel + (1|task)) require statsmodels (not installed) and are an optional extra; the pre-registered verdicts rest on the nonparametric tests above.
