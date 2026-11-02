---
name: pr-description
description: Write a torchtitan PR description that convinces a human reviewer -- concrete problem with a simple example, what changed, user-facing before/after, numerics, testing. Use when opening or updating a PR, or when asked to write or improve a PR summary.
---

# Writing a torchtitan PR Description

A maintainer reads the description first and must be convinced by it. They don't
want a narration of the diff; they want to know what problem exists, why this is
the right fix, and what evidence shows it works. Write for a reviewer who knows
torchtitan but not your context.

## Rules

- Never write the "Summary (human-written)" section. Leave its placeholder for
  the author, and tell the user to fill it in before review. Don't draft text
  for them to paste. With ghstack the PR body is the commit message, so keep the
  placeholder there too.
- Lead with the problem, in concrete terms. If you can't state a concrete input
  that misbehaves on `main` (or a concrete thing that is hard to do today), stop
  and reconsider whether the PR should exist.
- Include a **simple example**: a few lines showing the scenario before and after
  (a config snippet, a tiny tensor/sequence example, an ASCII diagram of data or
  control flow). This is the part reviewers ask for most often.
- Say why this is the right layer and the smallest fix, and name the alternatives
  you rejected in one line each. If the fix works around an upstream gap, link the
  upstream issue.
- Separate mechanical changes (moves, renames) from logic changes, and say which
  files reviewers can skim. For large changes, split commits into moves-only and
  logic commits, or stack PRs ("Stacked on #N").
- State evidence, not claims: numerics table, perf numbers end to end, the exact
  command. List failures that also happen on `main` separately.
- Keep it scannable: short sections, tables over prose. Don't paste a long agent
  transcript; if an agent wrote the summary, a human must have checked it says
  what concretely changes.
- Don't include sections with nothing to say.

## Template

```markdown
## Summary (human-written)
<!-- Write this yourself, not with an AI tool. 2-5 sentences: the problem, why
     this approach, and what you checked yourself. -->

## Why
<The concrete problem: what fails or is hard today, for whom. 2-5 sentences.>

**Example.**
<5-15 lines: minimal scenario, before vs. after.>

## What changed
- <grouped by concern; mark mechanical parts as such>

## Alternatives considered
- <one line each, why not>

## User-facing changes
| Before | After |
| --- | --- |
| `old.config.path` | `new.config.path` |

## Numerics
<For non-computation changes: bitwise-identical loss and grad_norm vs. base
commit <sha> (seed 42, deterministic, `scripts/loss_compare.py --assert-equal`),
per parallelism layout touched. For computation changes: convergence evidence
and an explanation of expected golden-loss changes.>

| Config | GPUs | Result |
| --- | --- | --- |

## Performance
<Only if relevant: TPS/MFU and memory before vs. after, end to end, same config.>

## Testing
<What you ran, on what hardware, and which existing tests cover the change. For
each new test, the regression it catches.>

## Known issues / follow-ups
<Failures that reproduce on main; TODOs left in code; next PR in the stack.>
```
