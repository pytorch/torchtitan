# Module Override Registry — Design Document

**Status:** Proposal (seeking feedback)
**Authors:** Tianyu Liu

## Motivation

Torchtitan hosts models with a config-driven build system where each module defines a
nested `Config` dataclass and `config.build()` constructs the owning module. Module
swapping (e.g., Float8 quantization, LoRA) currently works via `ModelConfigConverter`,
which traverses the config tree and replaces Config nodes at config construction time.

This approach requires modifying `config_registry.py` functions — meaning every
alternative implementation must be committed to the repo. This creates friction for:

1. **Hardware-specific kernels.** Efficient Triton kernels for rotary embeddings, fused
   attention, or custom norms that only work on specific hardware. These are valuable for
   performance but (a) hardware-specific, (b) harder to maintain, (c) held to a lower
   code quality bar than core. We don't want to gate landing these on the same review bar
   as core modules, and we don't want to say "no" to vendors when we can instead provide
   a clean extension mechanism.

2. **Fused module implementations.** An efficient MoE implementation may fuse the
   dispatcher-compute-combine pipeline with custom routing, communication, and
   quantization. These span multiple Config nodes and are best expressed as a single
   replacement at a higher scope (e.g., replace `MoE.Config` entirely).

3. **Vendor contributions.** Hardware vendors should be able to ship optimized
   implementations as separate packages, without requiring changes to the main branch.

## Design Overview

A **module override registry** where implementations register as alternatives to
existing `Configurable.Config` types. Overrides are applied post-config-construction,
pre-model-build.

### Core Principles

- **Explicit opt-in.** No auto-detection. The user specifies which override modules to
  import via `--override.modules`. If you don't ask for it, nothing changes.
- **Error on conflict.** If two imported modules register overrides for the same Config
  class, fail loudly. No priority system — the user resolves conflicts by importing
  fewer modules.
- **Module-level scope.** Overrides replace `Configurable.Config` nodes in the config
  tree. For op-level replacement (e.g., a single function), either wrap it in a Module
  first, or use PyTorch's `TORCH_LIBRARY` mechanism.
- **Minimal infrastructure.** The registry is ~100 lines. It extends naturally from the
  existing `traverse()` + replace pattern used by Float8 and LoRA converters.

## How It Works

### Registration

Override authors write a Python module that imports `register` and decorates a factory
function:

```python
# torchtitan/registry/triton_rope.py (in-repo example)
from torchtitan.registry import register
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

class TritonRoPE(Module):
    """RoPE implementation using Triton kernels."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        block_size: int = 128

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # ... Triton kernel setup ...

    def forward(self, xq, xk, positions=None):
        # Triton kernel application
        ...

@register("triton_rope", target=RoPE.Config,
          description="Triton-based rotary embedding, ~2x faster on A100+")
def triton_rope_override(cfg: RoPE.Config) -> TritonRoPE.Config:
    return TritonRoPE.Config(
        dim=cfg.dim,
        max_seq_len=cfg.max_seq_len,
        theta=cfg.theta,
        backend=cfg.backend,
        scaling=cfg.scaling,
    )
```

The `@register` decorator stores metadata + factory in a global dict. The factory
function takes the original Config and returns a replacement Config.

### Activation

The user specifies override modules via config (CLI or config file):

```bash
torchtitan_train --module llama3 --config llama3_8b \
    --override.modules='["torchtitan.registry.triton_rope"]'
```

### Application

In `Trainer.__init__`, after the model config is constructed and updated, but before
`model_config.build()`:

1. Import all listed modules (triggers `@register` decorators).
2. Check for conflicts (two overrides targeting same Config class → error).
3. For each registered override, `model_config.traverse(target_cls)` finds all
   matching Config nodes and replaces them with the factory output.
4. Log every replacement with FQN and type change.

```
INFO: [Override] triton_rope: layers.0.rope RoPE.Config -> TritonRoPE.Config
INFO: [Override] triton_rope: layers.1.rope RoPE.Config -> TritonRoPE.Config
...
INFO: Applied 32 module overrides
```

### External Packages

A vendor ships a pip package:

```python
# In package "torchtitan-vendor-x", file: vendor_x/overrides.py
from torchtitan.registry import register
from torchtitan.models.common.moe import MoE

class VendorXFusedMoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        top_k: int
        # vendor-specific fields...

    def forward(self, x):
        # Vendor's fused dispatch-compute-combine
        ...

@register("vendor_x_fused_moe", target=MoE.Config,
          description="Vendor X fused MoE for XPU")
def vendor_x_moe(cfg: MoE.Config) -> VendorXFusedMoE.Config:
    return VendorXFusedMoE.Config(
        num_experts=cfg.num_experts,
        top_k=cfg.router.top_k,
        ...
    )
```

```bash
pip install torchtitan-vendor-x
torchtitan_train --module deepseek_v3 --config dsv3_671B \
    --override.modules='["vendor_x.overrides"]'
```

## Override-Specific Configuration

Overrides may need their own tuning parameters (e.g., `block_size` for a Triton
kernel). Since overrides are applied after CLI parsing, extra Config fields can't be
set via CLI flags.

**Approach: the override module IS the configuration.**

In-repo example overrides ship with sensible defaults — most users import them
directly. For custom parameters, the user writes a small module file:

```python
# my_overrides.py
from torchtitan.registry.triton_rope import TritonRoPE
from torchtitan.models.common.rope import RoPE
from torchtitan.registry import register

@register("triton_rope", target=RoPE.Config,
          description="Triton RoPE with block_size=256")
def custom_triton_rope(cfg: RoPE.Config) -> TritonRoPE.Config:
    return TritonRoPE.Config(
        dim=cfg.dim, max_seq_len=cfg.max_seq_len,
        theta=cfg.theta, backend=cfg.backend, scaling=cfg.scaling,
        block_size=256,  # customized
    )
```

```bash
torchtitan_train --override.modules='["my_overrides"]' ...
```

**Why this over other approaches:**

| Approach | Typed? | CLI? | Extra infra? | Tradeoff |
|----------|--------|------|--------------|----------|
| Custom module file (recommended) | Yes | No (edit file) | None | Simple, fully typed, no infra. Changing params requires file edit. |
| Untyped kwargs dict | No | Yes | ~50 lines | CLI-friendly but error-prone, awkward nested dict syntax. |
| Typed override Config (like ModelConfigConverter) | Yes | Yes | ~100 lines | Couples config_registry to override package, defeating the "no-touch" goal. |

The custom module approach requires zero additional infrastructure and handles all
use cases. Most overrides have few tuning knobs with sensible defaults.

## Interaction with Existing Converters

Existing converters (Float8, LoRA) run inside `model_registry()` during config
construction. Overrides run later in `Trainer.__init__`. The override sees the
post-converter config tree.

| Override Target | Converter Conflict? | Notes |
|-----------------|---------------------|-------|
| RoPE.Config | None | Converters don't touch RoPE |
| FeedForward.Config | None | Converters don't touch FFN |
| MoE.Config | None | Converters target inner GroupedExperts |
| RMSNorm.Config | None | Converters don't touch norm |
| Inner attention | None | Converters don't touch attention |
| GroupedExperts.Config | Maybe | Float8GroupedExpertsConverter touches this |
| Linear.Config | Yes | Float8/LoRA replace these |

The 90% use case (RoPE, attention, MoE, norm, FFN) has zero conflicts. For the
edge cases (Linear, GroupedExperts), the factory can check `type(cfg)` and return
`None` to skip already-converted instances.

## Prerequisite: RoPE Refactor

Currently `RoPE.forward()` only returns a precomputed cache tensor. The actual
embedding application is done by free functions (`apply_rotary_emb_complex`, etc.)
called inside `GQAttention.forward()`. To make RoPE replaceable at the Module level,
the application step must be part of the RoPE Module.

**Plan:** For v1, pass the RoPE Module (not just its cache tensor) through the forward
chain. `GQAttention.forward()` calls `rope(xq, xk, positions)` instead of
`apply_fn(xq, xk, cache, positions)`. In a follow-up, each attention layer will own
its own RoPE module as a submodule, making it naturally replaceable via the config tree.

## Parallelism Compatibility

Replacement modules must satisfy the `Module` protocol (`ShardingConfig`, `init_states`,
etc.) for TP/FSDP to work. Model-specific `parallelize_fn` functions may reference
specific module types.

**v1:** Document that overrides must be drop-in compatible (same interface, same
submodule structure).

**Long-term:** Push parallelism fully into the Module protocol so `parallelize_fn`
doesn't need to know module types. This aligns with torchtitan's ongoing direction
and will naturally make overrides fully composable with parallelism.

## Scope of Changes

| File | Change |
|------|--------|
| `torchtitan/registry/__init__.py` | **New.** Registry mechanism (~100 lines) |
| `torchtitan/config/configs.py` | Add `OverrideConfig` dataclass (~10 lines) |
| `torchtitan/trainer.py` | Add `override` field, apply overrides before build (~15 lines) |
| `torchtitan/models/common/rope.py` | Prerequisite: make forward() include application |
| `torchtitan/models/common/attention.py` | Update to use RoPE Module instead of free functions |
| `torchtitan/registry/triton_rope.py` | **New.** Example override for demonstration |

**~200 lines of new infrastructure + ~100 lines changed in rope/attention refactor.**

No changes needed to any model's `config_registry.py` or `__init__.py` — that's the
point of the design.

## Open Questions for Feedback

1. **Naming:** "Module Override Registry", "Kernel Registry", "Implementation Registry"?
   Current proposal uses "override" throughout.

2. **Checkpoint compatibility:** Override authors are responsible for state_dict key
   compatibility. Should the registry enforce or validate this?

3. **Scope of traverse:** Currently limited to `model_config` (the model's Config tree).
   Should it extend to the full `Trainer.Config` tree (optimizer, dataloader, etc.)?

4. **Filter functions:** v1 replaces ALL instances of a target Config class. Should we
   support per-instance filtering (e.g., "only replace RoPE in layers 0-10") in the
   `@register` API, or is that a v2 feature?
