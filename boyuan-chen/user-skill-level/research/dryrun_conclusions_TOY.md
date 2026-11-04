# ⚠️ SYNTHETIC DRY-RUN — TOY DATA — NOT RESEARCH FINDINGS ⚠️

**Every number in this document comes from randomly generated data**
(`experiment/fixtures/make_dryrun_data.py`, seed 424242, model field = `TOY-SYNTHETIC`).
It exists only to validate the stage 6→7 pipeline (scoring schema → `stats.py` →
findings format) before real data arrives. It supports **no conclusions about any
model**. The real collection was paused at 100/648 responses to run this rehearsal;
real findings will replace this document's role at `research/06_findings.md`.

---

## What the dress rehearsal demonstrates (pipeline facts, real)

1. `stats.py` consumed a full-scale 648-row scores file + 648 judge entries and
   produced verdicts for all pre-registered hypotheses with Holm correction,
   trend tests, equivalence margins, and bootstrap CIs (`experiment/fixtures/dryrun/stats_skill_TOY.md`).
2. All three verdict types are exercised and render correctly: significant
   (planted effects), equivalence-met (planted null), and **inconclusive** (H5: not
   significant *and* equivalence not met — the CI was too wide). That third outcome
   is a live risk for the real run and is worth anticipating in the write-up.
3. The findings template below is the format stage 7 will use.

## Toy verdicts, interpreted AS IF real (format demonstration only)

*(Planted pattern: "tailoring, not rationing." The generator was told to make style
adapt and substance stay flat — the pipeline merely recovered what was planted.)*

- **H1 — style adapts (TOY: supported).** Comment ratio falls monotonically with
  stated skill (+0.163 beginner−expert, Holm p < .001; Page trend z = −8.0); prose
  words fall ~233 words from beginner to expert.
- **H2 — complexity rises with stated skill (TOY: supported).** Cyclomatic mean
  −0.98 beginner−expert (Holm p < .001); advanced idioms likewise.
- **H3 — equity/correctness (TOY: null, equivalence MET).** Beginner−expert pass-rate
  gap −0.4 pp, 95% CI inside ±6 pp: the pre-registered "no rationing of correctness"
  verdict. *This is the headline cell for the real study.*
- **H4 — hardening (TOY: no effect detected).**
- **H5 — substance via blind judge (TOY: INCONCLUSIVE — the instructive case).**
  Not significant, but the CI exceeded the ±5 pp equivalence margin, so neither
  "differs" nor "equivalent" can be claimed. If the real run lands here, the honest
  report says exactly that.
- **H6 — channel interaction (TOY: supported).** Self-disclosure moved comment ratio
  more than the system-prompt profile (+0.024, Holm p = .035).
- **H7 — default-user assumption (TOY: supported).** Control sits nearer the
  professional condition than the beginner one — echoing the *real* v1 pilot pattern
  (control FK grade 10.3, at the PhD end).
- **H8 — tone (TOY: supported).** Judge-rated condescension +1.17 points higher for
  beginners.

## Carry-forward checklist for the real analysis

- [ ] Watch H5's CI width — 24 tasks may be underpowered for the ±5 pp margin.
- [ ] `security_flags` was all-zero in toy data; confirm the real scorer populates it.
- [ ] Robust-rate tests only cover the 8 hardening tasks (n = 8 pairs — exact
      Wilcoxon, low power); frame H4 accordingly.
- [ ] Resume collection: same command, checkpoint continues from 100/648.
