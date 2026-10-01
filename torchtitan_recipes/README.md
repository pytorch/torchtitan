# TorchTitan recipes

This package contains complete, runnable training configurations. Core model
packages under `torchtitan.models` contain architecture definitions and model
flavors, but do not select datasets, optimizers, parallelism, or cluster paths.

Public recipes are limited to configurations that have been exercised by the
project in training, convergence, performance, or maintained feature workflows.
Configurations used only for CI live under `torchtitan_recipes.tests` and are
not recommendations for real training.

The package is organized by ownership:

- `models/` contains verified recipes for core models.
- `graph_trainer/`, `torchft/`, and `rl/` mirror their owning experimental or
  training subsystem.
- `overrides/` contains opt-in component implementations.
- `tests/` contains CI-only model fixtures and composed test suites.

| Module | Scope | Validation reference |
| --- | --- | --- |
| `torchtitan_recipes.models.llama3` | Llama 3 pretraining | `docs/converging.md`, `benchmarks/` |
| `torchtitan_recipes.models.deepseek_v3` | DeepSeek V3 pretraining | `torchtitan/models/deepseek_v3/README.md` |
| `torchtitan_recipes.models.qwen3` | Qwen3 pretraining and SFT | `torchtitan/models/qwen3/README.md` |
| `torchtitan_recipes.models.flux` | Flux pretraining | `torchtitan/models/flux/README.md` |
| `torchtitan_recipes.models.muse_glimmer` | Muse Glimmer pretraining | `torchtitan/models/muse_glimmer/README.md` |
| `torchtitan_recipes.graph_trainer` | GraphTrainer training | `torchtitan/experiments/graph_trainer/README.md` |
| `torchtitan_recipes.rl` | RL examples | `torchtitan/rl/examples/` |
| `torchtitan_recipes.torchft.llama3` | TorchFT replicated training | `torchtitan/experiments/torchft/README.md` |

Select a recipe by its exact import path:

```bash
MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh
```

Bundled opt-in component overrides live under `torchtitan_recipes.overrides`.
The generic override mechanism is documented in `torchtitan/config/OVERRIDE.md`
and accepts external modules as well.
