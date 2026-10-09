# Test recipes

These recipes exist to exercise TorchTitan in CI. They are not recommendations
for production training.

- `models/` contains reusable debug configurations for individual models.
- `suites/` composes model configurations into the cross-model feature and
  hardware integration suites.
- `graph_trainer/` and `torchft/` contain subsystem-specific test recipes.
- `rl/` contains debug-model and CI recipes for RL examples, one module per
  example (verified recipes live in `torchtitan_recipes/rl/`).
- `transformers_modeling_backend.py` contains the test recipes for that
  standalone test surface.

Keep a configuration with its model when it only changes model-local test
setup. Put it in `suites/` when it represents a cross-model or hardware test
matrix. Derive feature variants in the same module as their base recipe unless
the module is shared by several suites.
