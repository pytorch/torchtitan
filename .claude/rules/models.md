---
description: Rules for model implementations
globs: torchtitan/models/**
---

# Model Code Rules

## Keep Models Minimal and Readable
- Model files should contain model architecture, not training infrastructure.
  The exception is a workload with its own training loop (Flux diffusion), which
  keeps its `Trainer`/`Validator` subclass, data and inference next to the model.
- Weight initialization belongs in the config or a dedicated init function, not
  scattered through `Module.__init__`.
- After any model change, ensure original checkpoints still load correctly.

## Audit All Variants
When changing shared components (attention, normalization, MoE routing), check and
update **all** model variants under `torchtitan/models/`: llama3, qwen3,
qwen3_5 (and its qwen3_6/qwen3_8 flavors), deepseek_v3, deepseek_v4, gpt_oss,
kimi_k2_7, kimi_k3, muse_glimmer, flux, and any others present. Don't leave
stale patterns in sibling models.

## Unify Across Models
- Don't create per-model wrappers for the same functionality. Aim for at most one
  general wrapper shared by all models.
- If multiple models have near-identical code (e.g. sharding helpers or
  `_apply_fsdp` overrides), consolidate into `torchtitan/models/common/` or
  `torchtitan/distributed/`.
- Before adding a new rotary embedding, MoE router, or similar component, check if
  an existing implementation already supports the use case.

## Standard Model Folder Structure
Each model folder follows a consistent pattern (see `torchtitan/models/README.md`):
- `flavors.py` -- `MODEL_FLAVORS` and `build_model_config(flavor, parallelism=...)`
  (sizes, hyperparameters, and each component's sharding)
- `model.py` -- model definition; extra architecture files (e.g. `moe.py`) as needed
- `sharding.py` -- model-specific `ShardingConfig` constructors (TP/SP/EP placements),
  used by `flavors.py`; omitted when the shared ones in `models/common/` suffice
- `state_dict_adapter.py` -- HF checkpoint mapping
- `__init__.py` -- re-exports the model class, `MODEL_FLAVORS`, `build_model_config`

Training recipes live in `torchtitan_recipes/`, not in the model folder.

## Base Classes vs Shared Components
Abstract contracts live in `torchtitan/models/base/`: `BaseModel` (model
lifecycle), `Module` and its containers, `ShardingConfig`, and the state dict
adapter bases. Concrete components used by multiple models belong in
`torchtitan/models/common/`, including base architectures such as `Decoder`:
- Attention mechanisms (`attention/`)
- Feed-forward layers (`feed_forward.py`)
- Decoder blocks (`decoder.py`, `decoder_sharding.py`)
- MoE routing and expert implementations (`moe.py`, `moe_sharding.py`,
  `token_dispatcher.py`, `dist_moe/`)
- Rotary/positional embeddings (`rope.py`)
- Normalization layers (`norm.py`)
- HF state dict adapter for MoE models (`moe_state_dict_adapter.py`)
- Parameter counting and FLOP formulas (`flops.py`)

## Don't Over-Specialize
If a feature is only needed by one model (e.g. HFTransformerModel), implement it
in that model's folder. Don't modify shared infrastructure or base classes to
accommodate a single model's needs.

## Control Flow in Forward
Keep control flow (routing decisions, conditional logic) in the `forward` method.
Don't bury important branching logic inside helper methods where it's hard to trace.
