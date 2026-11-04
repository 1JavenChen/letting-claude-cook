# Stage-6 statistics - skill-level suite

Scores: `/Users/boyuanchen/user-model/.claude/worktrees/education-level-genai-experiment-27e674/experiment/out/scores_skill.csv` (648 rows). Judge scores: not available. Bootstrap: 5000 resamples, seed 20261014. Unit of analysis: task-level means (design doc section 7); all contrasts paired within task; levels pooled over channels unless stated.

### Per-condition means (automated metrics)

| condition | n | pass_rate | all_pass | comment_ratio | prose_words | cyclomatic_mean | advanced_idioms | validation_checks | security_flags | extraction_fail | leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control/none | 72 | 0.975 | 0.917 | 4.439 | 125.264 | 4.786 | 3.306 | 1.153 | 0.000 | 0.000 | 0.000 |
| beginner/system | 72 | 0.986 | 0.931 | 4.894 | 134.778 | 4.470 | 2.722 | 1.014 | 0.000 | 0.000 | 0.042 |
| beginner/message | 72 | 0.944 | 0.903 | 3.888 | 157.167 | 4.621 | 2.493 | 0.831 | 0.000 | 0.000 | 0.097 |
| hobbyist/system | 72 | 0.985 | 0.931 | 4.510 | 129.236 | 4.514 | 3.264 | 1.000 | 0.000 | 0.000 | 0.097 |
| hobbyist/message | 72 | 0.963 | 0.917 | 4.512 | 123.500 | 4.561 | 3.111 | 1.167 | 0.000 | 0.000 | 0.014 |
| professional/system | 72 | 0.973 | 0.917 | 4.238 | 134.625 | 4.441 | 3.887 | 1.042 | 0.000 | 0.000 | 0.000 |
| professional/message | 72 | 0.953 | 0.931 | 4.066 | 120.472 | 4.525 | 3.500 | 1.056 | 0.000 | 0.000 | 0.000 |
| expert/system | 72 | 0.948 | 0.875 | 5.243 | 129.472 | 4.745 | 3.750 | 1.111 | 0.000 | 0.000 | 0.000 |
| expert/message | 72 | 0.954 | 0.889 | 4.567 | 115.694 | 4.581 | 3.694 | 1.181 | 0.000 | 0.000 | 0.000 |

### Hypothesis tests

| id | hypothesis | metric | n | effect (b-e) | 95% CI | test | p | p (Holm) | Friedman p | trend p | family |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| H1 | Style: comment/docstring density | comment_ratio | 24 | -0.5382 | [-1.6440, +0.3278] | Wilcoxon (exact) | 0.5838 | 1.0000 | 0.6041 | 0.8389 | confirmatory |
| H1s | Style (secondary): prose words | prose_words | 24 | +23.3889 | [+8.3194, +43.2083] | Wilcoxon (exact) | 0.0035 | - | 0.0293 | 0.0169 | exploratory |
| H2 | Complexity: cyclomatic (mean/function) | cyclomatic_mean | 24 | -0.1091 | [-0.5816, +0.2581] | Wilcoxon (normal) | 1.0000 | 1.0000 | 0.9964 | 0.4438 | confirmatory |
| H2s | Complexity (secondary): advanced idioms | advanced_idioms | 24 | -1.0583 | [-1.4639, -0.6556] | Wilcoxon (normal) | 0.0002 | - | 0.0002 | 3.8e-05 | exploratory |
| H3 | Correctness: hidden-test pass rate | pass_rate | 24 | +0.0141 | [-0.0227, +0.0560] | Wilcoxon (exact) | 0.6250 | 1.0000 | 0.3811 | - | confirmatory |
| H3s | Correctness (secondary): all tests passed | all_pass | 24 | +0.0347 | [-0.0208, +0.0972] | Wilcoxon (normal) | 0.3053 | - | 0.2920 | - | exploratory |
| H4 | Robustness: validation checks (AST) | validation_checks | 24 | -0.1750 | [-0.4042, +0.0278] | Wilcoxon (normal) | 0.1392 | 0.6958 | 0.2240 | - | confirmatory |
| H4s1 | Robustness (secondary): security flags | security_flags | 24 | +0.0000 | [+0.0000, +0.0000] | Wilcoxon (degenerate) | 1.0000 | - | 1.0000 | - | exploratory |
| H4s2 | Robustness (secondary): handled-invalid rate, hardening tasks | robust_rate | 8 | +0.0944 | [-0.0167, +0.2236] | Wilcoxon (exact) | 0.1562 | - | 0.1466 | - | exploratory |
| H5 | Substance: requirements met (blind judge) | requirements_met_prop | - | - | - | SKIPPED: judge scores not available - run judge.py first | - | - | - | - | confirmatory |
| H6 | Channel: (beg-exp) message - (beg-exp) system, comment_ratio | comment_ratio | 24 | -0.3755 | [-2.5120, +1.5736] | Wilcoxon (exact) | 0.4908 | 1.0000 | - | - | confirmatory |
| H6s | Channel (secondary): same interaction on pass_rate | pass_rate | 24 | -0.0476 | [-0.1298, +0.0306] | Wilcoxon (normal) | 0.3737 | - | - | - | exploratory |
| H7 | Default: |control-professional| - |control-beginner|, comment_ratio | comment_ratio | 24 | +0.6867 | [+0.1634, +1.4330] | Wilcoxon (exact) | 0.0269 | 0.1616 | - | - | confirmatory |
| H7s | Default: |control-professional| - |control-beginner|, cyclomatic_mean | cyclomatic_mean | 24 | +0.3634 | [+0.0114, +0.8148] | Wilcoxon (normal) | 0.2119 | - | - | - | exploratory |
| H8 | Tone: condescension (blind judge) | condescension | - | - | - | SKIPPED: judge scores not available | - | - | - | - | exploratory |

### Verdicts (alpha = 0.05, Holm within the confirmatory family)

- **H1** (comment_ratio, confirmatory): effect -0.5382, p=0.5838, Holm p=1.0000 -> not significant; monotone decreasing trend not supported (Page z=+0.99)
- **H1s** (prose_words, exploratory): effect +23.3889, p=0.0035 -> significant; monotone decreasing trend supported (Page z=-2.12)
- **H2** (cyclomatic_mean, confirmatory): effect -0.1091, p=1.0000, Holm p=1.0000 -> not significant; monotone increasing trend not supported (Page z=+0.14)
- **H2s** (advanced_idioms, exploratory): effect -1.0583, p=0.0002 -> significant; monotone increasing trend supported (Page z=+3.96)
- **H3** (pass_rate, confirmatory): effect +0.0141, p=0.6250, Holm p=1.0000 -> not significant; pre-registered equivalence (|gap| < 3 pp and 95% CI inside +/-6 pp): MET
- **H3s** (all_pass, exploratory): effect +0.0347, p=0.3053 -> not significant
- **H4** (validation_checks, confirmatory): effect -0.1750, p=0.1392, Holm p=0.6958 -> not significant
- **H4s1** (security_flags, exploratory): effect +0.0000, p=1.0000 -> not significant
- **H4s2** (robust_rate, exploratory): effect +0.0944, p=0.1562 -> not significant
- **H5** - skipped: judge scores not available - run judge.py first
- **H6** (comment_ratio, confirmatory): effect -0.3755, p=0.4908, Holm p=1.0000 -> not significant
- **H6s** (pass_rate, exploratory): effect -0.0476, p=0.3737 -> not significant
- **H7** (comment_ratio, confirmatory): effect +0.6867, p=0.0269, Holm p=0.1616 -> not significant
- **H7s** (cyclomatic_mean, exploratory): effect +0.3634, p=0.2119 -> not significant
- **H8** - skipped: judge scores not available

### Notes

- Confirmatory primaries skipped (excluded from the Holm family of 6): H5.

### Methods notes

- Wilcoxon signed-rank: exact null when n <= 25 with untied |differences|, else normal approximation with tie + continuity corrections; zero differences dropped.
- Friedman: tie-corrected chi-square approximation, df = k-1 (approximation verified against a permutation null at n = 24 tasks; it is rough below n ~ 10, which this design never uses).
- Page's L trend (normal approximation) stands in for the design doc's Jonckheere-Terpstra as the ordered-alternative rank test.
- Effect sizes: paired mean differences with bootstrap percentile 95% CIs over tasks (seed recorded above).
- Mixed models (metric ~ level * channel + (1|task)) require statsmodels (not installed) and are an optional extra; the pre-registered verdicts rest on the nonparametric tests above.
