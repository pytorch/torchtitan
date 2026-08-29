# Q&A Correctness Grading Protocol (ATE-Bench, Appendix B.1.2)

The effort metrics (turns, per-turn context, output tokens) only count if the
attempt is **correct**. Q&A correctness is graded by humans, not programmatically,
because the answer is a set of code citations whose truth must be checked against
the code on disk.

## What the agent returns
Each Q&A answer is wrapped in `<final_answer>...</final_answer>` and cites file
paths, function names, and line numbers in the framework codebase that implement
the behavior the question asks about, with **one citation per claim** and a
step-by-step trace of the execution path.

## Grading procedure (from the paper)
1. **Two independent graders.** Each grader opens every cited code location and
   verifies that the code there actually does what the agent claims in the answer.
2. **Satisfied** = both graders confirm **every** citation against the code on
   disk. A single wrong/uncheckable citation fails the attempt.
3. **Disagreements** are resolved by a third reader who looks **only** at the
   cited evidence (not the agent's prose).
4. **Negative answers count.** If a feature is genuinely absent, a correct
   "absent, verified" answer (citing the grep/pattern that returned zero matches,
   or the file inspected that lacks it) is accepted and preferred over a
   fabricated one.

In the paper, all 108 Q&A attempts (12 questions x 3 frameworks x 3 attempts) were
judged satisfied by both graders — i.e. the comparison is purely about *effort*,
since correctness saturates.

## Recording results
For each `run<N>.jsonl` transcript, record a verdict in
`results/<label>/<task_id>/run<N>.grade.json`:

```json
{ "satisfied": true, "grader_notes": "all 3 citations verified", "graders": ["A", "B"] }
```

Only attempts with `"satisfied": true` should be included when comparing effort
across frameworks. `aggregate.py` currently medians over runs that the agent
itself did not error on; once grades exist, filter to satisfied attempts before
reporting (an optional `--grades` mode can be added).

## Tip: bootstrap grading with an LLM judge
The paper grades Q&A by hand but judges the *new-feature* tasks with an
independent `claude-opus-4-7` session at xhigh effort against fixed rules. You can
mirror that for a first pass on Q&A — have a fresh agent re-open each cited
`file.py:LINE` and confirm it supports the claim — then spot-check by hand. Keep
the judge independent from the agent that produced the answer.
