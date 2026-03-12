---
description: Rules for model implementations
globs: torchtitan/models/**
---

# Model Code Rules

## Keep Models Minimal and Readable
- Model files should contain model architecture, not training infrastructure.
- Weight initialization belongs in the config or a dedicated init function, not
  scattered through `Module.__init__`.
- After any model change, ensure original checkpoints still load correctly.

## Audit All Variants
When changing shared components (attention, normalization, MoE routing), check and
update **all** model variants: llama3, llama4, qwen3, deepseek_v3, gpt_oss, flux,
and any others present. Don't leave stale patterns in sibling models.

## Unify Across Models
- Don't create per-model wrappers for the same functionality. Aim for at most one
  general wrapper shared by all models.
- If multiple models have near-identical code (e.g. `apply_fsdp`, `apply_ac`,
  `apply_compile`), consolidate into `torchtitan/models/common/` or
  `torchtitan/distributed/`.
- Before adding a new rotary embedding, MoE router, or similar component, check if
  an existing implementation already supports the use case.

## Standard Model Folder Structure
Each model folder follows a consistent pattern:
- `config_registry.py` — registers model configs (sizes, hyperparameters)
- `parallelize.py` — defines parallelism strategy for the model
- Model definition files (architecture, layers)

## Shared Components
Components used by multiple models belong in `torchtitan/models/common/`:
- Attention mechanisms (`attention.py`)
- Feed-forward layers (`feed_forward.py`)
- Decoder blocks (`decoder.py`)
- MoE routing and expert implementations (`moe/`)
- Rotary/positional embeddings (`rope.py`)
- Normalization layers (`rmsnorm.py`)

## Don't Over-Specialize
If a feature is only needed by one model (e.g. HFTransformerModel), implement it
in that model's folder. Don't modify shared infrastructure or base classes to
accommodate a single model's needs.

## Control Flow in Forward
Keep control flow (routing decisions, conditional logic) in the `forward` method.
Don't bury important branching logic inside helper methods where it's hard to trace.
