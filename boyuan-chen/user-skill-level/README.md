# Does Telling an AI Your Skill Level Change the Code It Writes?

*Boyuan Chen · Harvard University · Letting Claude Cook first-year seminar, Fall 2026*

## About this assignment

This is my project for *Letting Claude Cook*, a Harvard first-year seminar taught by
Douglas Finkbeiner on how far AI agents can go in doing science on their own. Using AI
agents to do the work is the point: an AI coding agent (Claude Code) wrote most of the
code, ran the data collection, and drafted the report. I chose the question, approved
each design decision, checked the agent's work, and decide what the results mean. The
experiment is also *about* AI: it asks whether a code-writing model treats people
differently depending on how skilled they say they are.

## The question

People often tell a chatbot who they are: "I'm a beginner, go easy on me" or "I'm a
senior engineer, skip the basics." Models are known to change their *tone* in response.
The harder question is whether they change the *substance*. If someone who presents as
less skilled quietly gets code that is more likely to be wrong, that is a fairness
problem, and it is the kind of problem the person is least able to notice.

Code is a good place to test this because it can be checked objectively: the program
either passes the hidden tests or it does not, however the answer reads.

## How the experiment works

1. **Same task, different user.** Each programming task is sent to the model under nine
   conditions: four stated skill levels (beginner, hobbyist, professional, expert) told
   to the model in two ways (a system prompt set by the operator, or the user saying it
   themselves in the message), plus a control with no skill statement. The task text is
   byte-identical across conditions. Each condition is repeated three times.
2. **Objective scoring.** The model's code is extracted and run in a sandbox against
   hidden tests. We also measure the prose around the code (word count, apologies,
   give-ups), the code's style (comments, advanced idioms, complexity), and a second,
   larger model blindly judges quality.
3. **Three batteries.** Experiment 1 was a pilot on everyday questions with stated
   education level. Experiment 2 used 24 ordinary programming tasks (HumanEval-style).
   Experiment 3, in progress, uses 24 much harder competitive-programming problems from
   Codeforces (ratings 1000 to 2000), chosen so the model fails often enough for a
   difference to be visible.

Subject model: `openai/gpt-oss-20b` on Groq's free tier. Judge: `gpt-oss-120b`. A Colab
notebook (`experiment/colab_gemma4_cf.ipynb`) reruns Experiment 3 on Gemma 4 12B.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Pilot: everyday questions, stated education level | Done |
| 2 | Main study: 24 programming tasks x 9 conditions x 3 reps = 648 answers, sandbox-scored | Done (blind-judge pass 74% complete) |
| 3 | Hard study: 24 Codeforces problems, same design | In progress: 303 of 648 answers collected; first replicate scored |
| 4 | Fresh control answers for the hard study (unbiased baseline) | Queued |
| 5 | Positive control: can the model sandbag on beginners when told to? | Planned |

## Findings so far

**Experiment 2 (ordinary tasks).** Stated skill level did **not** change correctness.
Beginners' code passed hidden tests at the same rate as experts' (difference +1.4
percentage points, well inside the pre-registered equivalence band). What changed was
the *form*: beginners got about 20% more explanatory prose and code with about 28% fewer
advanced idioms, the most robust effect in the study. Self-disclosure in the message
moved the prose more than the operator's system prompt did. A caveat: the model solved
96% of these tasks, so there was little room for correctness to drop.

**Experiment 3 (hard tasks, provisional, first replicate).** With the ceiling removed
(the model now passes about half the tests), the picture holds: pass rates for beginner,
hobbyist, professional and expert are within a few points of each other, and the
task-paired confidence intervals include zero. Stating any level roughly halves the
explanatory prose. The control condition looks a few points better than every stated
level, but the 24 tasks were selected using those same control answers, so that gap is
expected from regression to the mean; stage 4 collects fresh control answers to remove it.

The plain-language write-up with all numbers, figures and references is
[`research/report.pdf`](research/report.pdf). The provisional findings memo is
[`research/06_findings.md`](research/06_findings.md).

## What is in this folder

- `experiment/`: the runner (`run_experiment.py`), sandbox scorers (`score.py`,
  `score_cf.py`), statistics (`stats.py`), ablations and figures, the Codeforces task
  builder (`build_tasks_cf.py`), the pipeline scripts, and the Colab notebook.
- `results/`: scored answers and summaries (CSV and text).
- `research/`: literature review, pre-registered hypotheses and design, methods plan,
  collection log, analysis memos, and the LaTeX report.
- `EXPERIMENT.md`: the original design document.

The 50 MB Codeforces task file and the raw model answers are not included; the task
file is rebuilt by `experiment/build_tasks_cf.py` from the `open-r1/codeforces` dataset.

## References

See the bibliography in `research/report.pdf`. Closest prior work: Poole-Dayan, Roy &
Kabbara (2024) on targeted underperformance for vulnerable users; Salewski et al. (2023)
and Hu & Collier (2024) on persona effects; van der Weij et al. (2025) on sandbagging.
