# NVFP4 Local SPMD Cleanup

## Goal

Remove NVFP4 DTensor support and unrelated branch changes so NVFP4 uses the
`spmd_types` local-tensor path exclusively.

## Current State

- NVFP4 forwards local weight, bias, and seed tensors directly to TorchAO.
- `Config.build` inlines colwise/rowwise local-map sharding and declares
  `_sr_seed` varying across DP, CP, and TP.
- All four Llama NVFP4 recipes default to `spmd_types` while allowing CLI
  overrides.
- Both Blackwell-only NVFP4 integration variants and their skip gate are removed.
- `torchtitan/overrides/README.md` matches `origin/main`.

## What Changed

- Removed NVFP4 DTensor imports, unwrapping, type detection, helper APIs, and
  forward-time seed assertion.
- Inlined colwise/rowwise sharding construction into `NVFP4Linear.Config.build`.
- Added `_sr_seed` to module state sharding as varying across DP, CP, and TP.
- Set all four Llama NVFP4 recipes to default to `spmd_types`.
- Removed both NVFP4 integration variants and `skip_if_no_blackwell`.
- Restored `torchtitan/overrides/README.md` from `origin/main`.
- Replaced helper-oriented unit coverage with direct colwise/rowwise build,
  seed-layout, recipe-default, and CLI-override assertions.

## Validation

- PASS: Direct ufmt check over all six changed Python files; all already formatted.
- PASS: Direct flake8 with `.flake8` over all six changed Python files.
- PASS: `pytest -q tests/unit_tests/test_quantization.py` -- 18 passed.
- PASS: `git diff --check`.
- PASS: `git diff origin/main -- torchtitan/overrides/README.md` is empty.
- PASS: Searches confirm the removed NVFP4 DTensor/helper symbols, integration
  variants, and Blackwell gate are absent.
- SKIPPED: Pre-commit and Pyrefly, by explicit user direction.
- NOT RUN: Paired 10-step Blackwell numerical comparison. SM100 hardware is
  available, but the resumed validation scope was explicitly limited.

## Preflight

- Contract: Make only the planned NVFP4, recipe, integration-test, unit-test,
  upstream README restoration, and required state-file changes.
- Next action: Run the configured hooks through `uvx pre-commit run --all-files`.
- Expected outcome: All configured repository checks pass without source changes;
  if a hook changes or rejects source, enter the bounded debug loop before editing.
- Risk: `uvx` will populate an external package cache and pre-commit will create
  its normal hook environments outside the repository.
- Confidence: HIGH.

## Surgical Simplicity

The unit-test edits replace tests for removed helpers with direct colwise/rowwise
build assertions and recipe-default coverage. `debug-session.md` is required by
the failed-validation recovery loop; this file is required because the change
touches more than three files and validation required an environment checkpoint.
No new production file, abstraction, parameter, or standalone test file was
introduced.
