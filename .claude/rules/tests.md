---
description: Rules for tests and CI
globs: tests/**, **/tests/**, .github/workflows/**
---

# Test and CI Rules

CI capacity (especially GPU) is shared and limited. A test is code that must be
maintained; it has to earn its cost.

## When to Add a Test
- Add a test when it guards an invariant that would otherwise break silently
  (numerics, gradient placement, checkpoint round-trip, a composability
  combination) and no existing test covers it. State which regression it catches.
- Bug fixes: add a regression test only if the bug could silently come back and
  existing tests didn't catch it; otherwise say why none is needed.
- Don't add tests that only re-read a config they just set, assert constants,
  grep docs or READMEs, or duplicate an existing test for one more model.

## Prefer Extending Existing Coverage
- Enable a new feature in an existing integration configuration (e.g. change an
  existing "FSDP 4" entry to "FSDP 2 + CP 2") instead of adding a new entry.
- Cover the harder configuration: interleaved PP schedules over 1F1B, EP x SP
  with TP > 1, CP, gradient accumulation. One test that composes several features
  beats several trivial ones.
- CI should exercise the recommended defaults (e.g. the attention backend and
  SPMD settings recipes actually use), with spmd_types type checking on.
- Use the cheapest hardware and smallest debug model that exercises the code.
  Debug models keep small vocab and sequence length.

## Placement
- Unit tests go in `tests/unit_tests/{cpu,gpu}/` mirroring the source tree; RL
  tests in `tests/rl/`; an experiment's tests in that experiment's folder. Never
  under `torchtitan/`.
- Experiment and RL jobs don't go into core CI workflow files.
- Don't put GPU-only (`skipUnless(cuda)`) tests in CPU-only suites: they never run.
- Don't guard tests on packages that are already in `requirements-dev.txt`.

## Golden Numerics
- Golden loss files capture any numerics change; never loosen tolerances to make
  a change pass. An expected change must be explained in the PR and regenerated
  for every affected backend and suite (e.g. both real-PG and fake-PG goldens).
- Verify that the intended test actually ran (not skipped, right config).
