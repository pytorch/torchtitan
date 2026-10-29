# torchtitan Review Checklist

Each item is phrased as the question a maintainer would ask. Only report items
that are actually violated in the diff. Items marked (diff check) are mechanical:
look for them in added lines with a quick search of the diff.

## 1. Necessity and Design

- [ ] **Concrete problem.** "Without this PR, what error would you hit, and
  when?" The description or your own analysis must give a minimal failing
  example. "Do such issues happen on main today?"
- [ ] **Root cause.** Is this a workaround? "Why doesn't the other call site need
  this treatment -- is there a bug there too?" "What regressed recently?"
- [ ] **Right layer.** "Is this a model problem rather than a state-dict-loading
  problem?" Would the fix be simpler in the dataloader / config / model than in
  the trainer? Would a different ordering (e.g. init after parallelize) make the
  problem disappear?
- [ ] **Upstream.** "Why do we have to call private APIs? Who should expose this
  option, in principle?" Workarounds for PyTorch/vLLM/Monarch/DCP gaps need a TODO
  with an issue link. Techniques of moderate-to-large complexity (new
  parallelisms, optimizers, profiler tooling) belong upstream.
- [ ] **Ownership.** Should torchtitan own this at all? (Security hardening of
  generic utilities, broad env-var surfaces, vendor-specific tuning, features for
  one team's fork.) "Can this be hosted in your fork / your own CI?"
- [ ] **Direction.** Does it build on a path `direction.md` says the codebase is
  migrating away from? Is it superseded by an open PR ("this issue will be gone
  after #N")?
- [ ] **Value evidence.** New feature or hyperparameter: users, papers, tech
  reports, or measured gains? Perf claims quantified end-to-end ("how much e2e,
  from X% MFU?"), not only microbenchmarks? An extension point with no in-repo
  consumer needs a named user ("mind providing motivation / use cases?").
- [ ] **Alternatives.** Is there a simpler design? Write it out with its approximate
  size. Prefer composing features independently over coupling them.

## 2. Complexity and Minimalism

- [ ] Net LoC growth is justified; large additions with no deletions get "where
  does the added complexity mainly come from?"
- [ ] Each new flag / config field / env var: "Why do we need this? Would it ever
  be set differently? Why don't we always enable it?" If always constant,
  hardcode; if a case isn't needed, ban it.
- [ ] Each new function / class / file: "Is this used more than once? Can we
  inline it?" No forwarding-only wrappers, no base class with one subclass, no
  registry where nothing registers. A method that doesn't use `self` is a function.
- [ ] Optional/`None` handling: "Would it ever be None?" If not, remove the
  `Optional`, the default, and the fallback branch.
- [ ] No speculative checks, casts, try/except fallbacks, or version shims.
- [ ] Intrusiveness: does `trainer.py` / `training_engine.py` / a common module
  now know about one model or feature? "Can we achieve this in the model folder /
  pipelining function / via config at build time?"
- [ ] Duplicated code vs a new hook: for a single consumer, duplicating a little
  code in the consumer beats adding a hook or branch to shared code.
- [ ] Hacky integration (monkeypatch, `install_*` / `patch_*` functions, thread
  tricks, editing args in place to pass information) signals the wrong design.

## 3. Placement, Ownership, Config

- [ ] Code is in the right place (`CLAUDE.md` Code Placement): model-specific code
  in the model folder; generic helpers in `distributed/` or `models/common/`
  only when more than one consumer exists; thin wrappers in `nn_modules.py` only.
- [ ] Framework/engine code (vLLM, Monarch) stays in its subpackage; core is
  framework-free; imports have no side effects. (diff check: new top-level imports
  of vllm/monarch outside their packages)
- [ ] Core does not import `torchtitan/experiments/` (diff check).
- [ ] Component takes only what it needs: no whole-trainer or root-config
  parameters; no vendor terms leaking up the call chain.
- [ ] Config (`rules/config.md`): `X.Config | None` instead of `enable_*`
  (diff check: `enable_` / `enabled` fields); implementations selected by config
  type, not strings/modes; `Literal` for fixed sets; no default on required
  fields; justified defaults; validation in `__post_init__`; no runtime mutation of
  the config tree; no new CLI flags; no pass-through `**kwargs`.
- [ ] Public vs private: no cross-module imports of `_private` names (diff check);
  module-local helpers have a leading `_`; public methods have external callers
  ("who calls this?").
- [ ] Docs next to code: new `docs/*.md` pages, or docs far from their component,
  should move; new docs outside `.claude/` are named `README.md` or
  `UPPERCASE_WITH_UNDERSCORES.md` (diff check).

## 4. Correctness: Distributed, Numerics, Performance, Hardware

- [ ] Every parallelism the change touches: FSDP/HSDP, TP/SP, PP (including
  interleaved schedules and multiple stages per rank), CP, EP, and their
  combinations; plus gradient accumulation, remat, CUDA graphs, checkpoint resume.
  "What if PP > 1? What if EP degree is 1 but TP > 1? What if a PP stage has none
  of these layers?"
- [ ] Mesh assumptions: no assumed 1D mesh or axis order; local vs global values
  correct on every rank; collectives over several axes leave ranks consistent;
  no rank-dependent control flow that can hang a collective.
- [ ] SPMD: new code uses spmd_types / `ShardingConfig`; no new
  `isinstance(..., DTensor)`, `.to_local()`, `DTensor.from_local` in model forward
  (diff check); `local_map` regions declare `in_types`; where DTensor conversions
  remain, `grad_placements` are explicit.
- [ ] Numerics: does the change alter loss/grad_norm? If it claims not to, is
  that plausible (refactors must be bitwise identical)? If it does, is the change
  explained and are golden files updated for every affected suite/backend?
  "Where does the numerics change come from, exactly?" Be suspicious both when
  numbers move unexpectedly and when they fail to move.
- [ ] Precision: no silent dtype downgrades in accumulation; trainer and RL
  generator compute consistent numerics.
- [ ] Performance: no hidden D2H syncs in the step (diff check: `.item()`,
  `.tolist()`, `.cpu()`, `repeat_interleave` without `output_size`); still
  CUDA-graph capturable and compile friendly; nothing per layer that can be done
  once; no memory/throughput regression on the main path. Fast paths should be
  the defaults in recipes.
- [ ] Hardware: no `torch.cuda.*` in generic code (use `torch.accelerator` / device
  module); no NCCL- or vendor-specific knobs in core config; unsupported hardware
  fails clearly at config validation (diff check: `torch.cuda.`, `"cuda"` string
  literals, `NCCL_` env vars in core).
- [ ] Errors: unsupported configurations raise `ValueError` early; no
  warn-and-continue or silent fallback to a slower path.

## 5. Naming

- [ ] Accurate and unambiguous; avoids overloaded terms ("batch", "step",
  "stream", "group", "replica", "engine", "weight sync"); established terms rather
  than invented ones; one name per concept; consistent with sibling files and
  upstream (DTensor/spmd_types language); symmetric names for counterparts.
- [ ] Names don't mislead: soft bounds aren't named `target_`; a function named
  `shard_x` always shards; "full" doesn't include exceptions.
- [ ] Shape suffixes: one letter per dim after the last `_`; logical (global)
  dims, not sharded/local ones; one legend per file; suffixes not dropped in
  rewritten code.
- [ ] Unexplained renames of existing names in the diff ("why changing the name?").

## 6. Tests and CI (`rules/tests.md`)

- [ ] Each new test: what regression does it catch that existing tests don't? Flag
  tests that re-read configs, assert constants, grep docs, or duplicate existing
  coverage for one more model.
- [ ] Could it be folded into an existing integration config, or cover a harder
  configuration instead?
- [ ] Placement: `tests/unit_tests/{cpu,gpu}/`, `tests/rl/`, or the experiment's
  own folder; never under `torchtitan/`; experiment/RL jobs not added to core CI
  files; no GPU-only tests in CPU suites; no skip guards for dev requirements.
- [ ] Debug models stay small; the cheapest hardware that exercises the code.
- [ ] Missing test only if a silent-correctness invariant is unguarded (e.g.
  gradient scaling under a parallelism combination).

## 7. Scope, Description, Docs

- [ ] One concern per PR: unrelated renames, reformatting, comment rewrites, or
  drive-by fixes are flagged ("accidental?", "does this belong to this PR?").
- [ ] Description: concrete problem and example, before/after for user-facing
  changes, numerics evidence, what was tested. Claims match the code (defaults,
  behavior). AI-generated summaries that don't say what concretely changes are a
  finding.
- [ ] Temporary workarounds carry a TODO (and an issue link when they wait on
  upstream); magic numbers have a comment explaining the value.
- [ ] Non-trivial features have a doc next to the code with an example or ASCII
  diagram; docs describe current state, not history.
- [ ] Dependencies: new ones pinned, placed at the right level (dev-only,
  example-only), and justified.
