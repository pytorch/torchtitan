# TorchTitan Development Guide

## Build & Test

```bash
# Install dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Lint and format (required before any PR)
pre-commit run --all-files

# Run unit tests
pytest tests/ -x
```

### Run GPU integration tests (requires GPUs)
Integration tests override default config for Llama 3 debug model.
See tests/integration_tests/ for `OverrideDefinitions`.

### Validating Numerics
Non-computation changes (e.g. activation checkpointing, refactoring) must produce
**identical loss** before vs. after with `--debug.seed=42` and `--debug.deterministic`.
Computation changes require loss convergence on representative datasets (e.g. C4).

With the same parallelisms, GPU settings, and the debug options, two runs should produce
bit-wise identical loss and grad_norm. Note that stdout only prints the most
significant five digits, which may not be enough. Follow `scripts/loss_compare.py` to
enable profiling and check loss and grad_norm from the TensorBoard results.

You should NEVER use `--debug.deterministic_warn_only`.

## Core Principles

1. **PyTorch-native training techniques.** Core torchtitan's training infrastructure
   and parallelism code must not depend on non-PyTorch libraries. Techniques with
   moderate-to-large complexity belong in their proper upstream repo (pytorch/pytorch
   for parallelisms, pytorch/data for data loaders, etc.).

2. **Investigate root cause before patching.** Don't land band-aid fixes. Understand
   *why* something fails before proposing a solution. If a change seems to help but
   you can't explain why, dig deeper.

3. **Reuse over duplication.** Before writing new code, check if existing implementations
   already handle the case. Unify similar code paths across models rather than creating
   per-model wrappers. If upstream (torchao, PyTorch) already provides functionality,
   use it.

4. **Don't leak experiments into core.** The `torchtitan/experiments/` folder exists for
   a reason. Don't modify core torchtitan code to accommodate experiment-specific needs
   (e.g. don't add `if experiment_x:` branches to core files). Deprecated files should
   be removed, not updated.

5. **Protect battle-tested code paths.** Be cautious changing converged behavior. Flag
   potential silent breakage of existing user code or checkpoints. When in doubt, ask.

6. **Audit all callsites.** When changing shared code (common model components, config
   fields, distributed utilities), check and update every callsite. This includes all
   model variants: llama3, llama4, qwen3, deepseek_v3, gpt_oss, flux, etc.

## Code Style

### Naming
- Names must be **accurate, descriptive, and reflect actual scope**. Don't use
  "toy/test/temp" in production names — put that context in docstrings instead.
- Follow upstream conventions: match torchao and PyTorch naming where applicable.
  E.g. if torchao calls it `Float8Linear`, use `Float8Linear` not `Float8Config`.
- Use `num_` prefix for counts (e.g. `num_expert_groups` not `n_expert_groups`)
  when not directly matching an upstream API.
- **`axis` for `DeviceMesh`, `dim` for tensors.** In any name we own — variables,
  parameters, attributes, helpers, comments, docstrings, error messages — use
  ``axis``/``axes`` for a ``DeviceMesh`` axis and reserve ``dim``/``dimension``
  for tensor dimensions. The lone exception is when calling into PyTorch
  upstream API (``DeviceMesh.mesh_dim_names``, ``DataParallelMeshDims``, etc.):
  match upstream spelling exactly at the call site, then assign into a locally
  named ``mesh_axis_names`` if the value flows through our code.

### Code Placement
- Put code in the **most general applicable location**:
  - Model-agnostic parallelism helpers → `torchtitan/distributed/`
  - Shared model components (attention, MoE, embeddings) → `torchtitan/models/common/`
  - Model-specific code → the specific model folder
- Don't put model-agnostic functionality in model-specific files just because
  that's where you first needed it.

### Assertions and Error Handling
- **`ValueError`** for user-facing errors (bad config, invalid input).
- **`assert`** only for internal invariants that indicate programmer error.
- Always validate mesh axes, tensor placements, and config values explicitly
  in distributed code — don't assume a 1-axis mesh or specific placements.
- When a code path silently skips user configuration, **emit a warning**.

### Parameters and Config
- Important parameters first; less important ones later.
- Prefer keyword-only arguments after the first positional arg.
- No `None` defaults for required config fields.
- `dataclasses.replace()` is a shallow copy: nested dataclasses and list/dict
  fields are shared by reference. Be explicit when deep copies are needed.

### Comments and Documentation
- Add comments only for genuinely non-obvious things: dimension semantics,
  parallelism gradient placements, why a workaround exists.
- Use TODO comments for known limitations with a brief explanation.
- Put descriptions in docstrings, not in names.

## PR Expectations

1. **Lint first.** Run `pre-commit run --all-files` and fix all issues before
   requesting review. CI linting failures waste everyone's time.
2. **Show numerical proof.** Include loss comparison for any non-trivial change.
3. **Explain "why" not just "what"** in the PR description.
4. **Add tests.** New features need CPU unit tests at minimum; GPU integration
   tests when involving parallelism. Verify CI actually runs the intended test
   config (check `--model.name` and other flags).
5. **Keep model code minimal.** After model changes, ensure original checkpoints
   still load correctly. Document reasons for model changes.
