---
description: Current migrations -- what new code should build on
globs: torchtitan/**, torchtitan_recipes/**, tests/**
---

# Direction: Where the Codebase Is Going

Maintained by the torchtitan maintainers. Reviewers use this to flag PRs that
build on a path the codebase is moving away from; authors use it to pick the
path that will last.
Update this file in the same PR that starts, finishes, or reverses a migration.

## Build On / Migrate Toward
- **SPMD**: spmd_types plus declarative `ShardingConfig`s
  (`torchtitan/models/base/sharding.py`; example `torchtitan/models/llama3/flavors.py`).
  Don't add DTensor-specific code in model forward or new features.
- **Model config**: `build_model_config(flavor, *, parallelism, ...)` builds a
  complete config, sharding included (`torchtitan/models/README.md`).
- **Config transforms**: `ModelConfigTransform` (`torchtitan/config/transform/`).
  `ModelConfigConverter` is legacy -- don't add new converters.
- **Optional components**: `X.Config | None`. Remaining `enable_*` booleans are
  legacy; don't add new ones.
- **Data**: the Grain-based pipeline (`torchtitan/components/data/`); HF datasets
  only as a source. Microbatches are packed 1-D token sequences sized by
  `num_tokens_per_microbatch_per_dp_rank`; padding and document boundaries are
  passed explicitly from the data side, not inferred later.
- **Activation recompute**: torch_remat via `torchtitan/distributed/activation_remat.py`
  (`torchtitan/distributed/REMAT.md`).
- **MoE**: token dispatch backends (all-to-all, DeepEP, HybridEP under
  `torchtitan/distributed/deepep/`) and Dist-MoE (`torchtitan/models/common/dist_moe/`)
  are all maintained. Build on them rather than adding new dispatch paths.
- **Docs**: next to the code. `docs/` keeps only benchmarks and a few legacy
  pages; don't add new pages there.
- **Recipes**: `torchtitan_recipes/`. Overrides there are opt-in examples
  (`torchtitan/config/OVERRIDE.md`).
