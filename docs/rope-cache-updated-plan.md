# Updated RoPE sharing plan for PR #4376

## Context

This plan follows the review discussion on
[PR #4376](https://github.com/pytorch/torchtitan/pull/4376), especially the
proposal to make Decoder-owned RoPE modules the explicit dependencies of the
attention layers:

- [latest maintainer proposal](https://github.com/pytorch/torchtitan/pull/4376#issuecomment-5465560593)
- [analysis of the single-module design](https://github.com/pytorch/torchtitan/pull/4376#issuecomment-5465562358)
- [registry-scoping review](https://github.com/pytorch/torchtitan/pull/4376#discussion_r3887739753)
- [duplicate-compute review](https://github.com/pytorch/torchtitan/pull/4376#discussion_r3887741233)
- [Helion review](https://github.com/pytorch/torchtitan/pull/4376#discussion_r3887743886)
- [MTP context review](https://github.com/pytorch/torchtitan/pull/4376#discussion_r3887745493)

The earlier attempt in
[PR #4111](https://github.com/pytorch/torchtitan/pull/4111) cached the built
module on a shared config object. We should not adopt that mechanism because it
makes `Config.build()` stateful and weakens the rule that a config build returns
a fresh owner. The updated design makes sharing explicit at the Decoder boundary
instead.

The newest reply identifies an important limitation in the single-module
version: newer architectures can intentionally use more than one effective RoPE
configuration. For example, DeepSeek-V4 selects `compress_rope_theta` for
compressed layers and `rope_theta` for pure sliding-window layers in its
`freqs_cis` construction ([reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py#L439-L445)).
The design therefore needs a model-local collection of canonical RoPE modules,
not a homogeneity assertion.

This remains a proposed direction, not a settled review outcome. The PR author
has [asked the reviewer to confirm the revised
design](https://github.com/pytorch/torchtitan/pull/4376#issuecomment-5465642070),
and the latest reply proposes a collection of Decoder-owned RoPE modules
([comment](https://github.com/pytorch/torchtitan/pull/4376#issuecomment-5488760743)).
Implementation should keep the ownership and selection rules isolated and easy
to review before the distributed-path changes are stacked on top.

The current branch is also behind `origin/main`; rebase it before implementation
and re-check all touched call sites against the rebased tree.

## Decision

Replace the registry/reader implementation with a Decoder-owned `ModuleDict` of
canonical RoPE modules. The Decoder registers each distinct RoPE config under a
descriptive string key. Pass the collection explicitly through the existing
layer construction loops; each attention builder derives the key from its own
RoPE config and fetches the matching canonical module.

```text
Decoder
|-- rope_modules["ComplexRoPE_d128_ctx1048576_theta10000_scaling_none"]
|                               registered canonical RoPE owner
|-- rope_modules["CosSinRoPE_d128_ctx1048576_theta10000_scaling_none"]
|                               registered canonical RoPE owner
`-- layers
    `-- attention
        `-- rope -------------- non-registered reference to one owner
```

Each PP stage owns its own copy of the complete `Decoder.rope_modules` collection.
Duplicate caches across multiple virtual stages on one rank are accepted. We do
not attempt cross-stage or cross-model sharing.

The collection is deduplicated by exact effective RoPE configuration. A model
may therefore have zero, one, or several canonical RoPE modules. A layer must
never silently receive a cache for a different configuration.

`rope_modules` is the preferred name: it describes the registered objects and
does not imply that the Decoder owns raw cache tensors or a global registry. Use
`ModuleDict` because the key is now the explicit routing contract. The old
generated numeric key style (`rope_0`, `rope_1`, ...) is removed. Keys are
descriptive, deterministic, and contain only characters accepted by PyTorch
module names; they are internal module names, not config or checkpoint APIs.

## Why this supersedes the current branch

The current registry solves steady-state storage duplication, but introduces
four costs that the updated design removes:

1. `contextvars` implicitly select a model registry and require MTP to re-enter
   a context manually.
2. Every layer still computes a full candidate cache before duplicates are
   discarded, and every layer recomputes it during `init_states()`.
3. `RoPECacheReader` creates a strong path from each layer back through the
   registry to the Decoder.
4. Moving the real buffer to `_rope_cache_N` on Decoder leaves the existing
   `state_shardings={"cache": ...}` declarations attached to RoPE configs but
   disconnected from the module that owns the actual buffer.

With canonical modules, each module identity survives `.to()`, `to_empty()`,
`init_states()`, and buffer replacement. Only that module's `cache` changes;
every attention keeps resolving the selected cache through its stable module
reference. Therefore the reader, registry, context, and cache property are
unnecessary. The model-local collection replaces the old process of selecting a
cache slot through an implicit context.

## Detailed implementation plan

### 1. Restore RoPE to a normal module

Revert the cache implementation to the ordinary upstream RoPE module, then
add only the shared naming helper needed by the Decoder-owned collection:

- Delete `RoPECacheReader`, `_RoPECacheRegistry`, the context variable/context
  manager, and `register_rope_cache()`.
- Replace the old cache-slot `_cache_key()` with deterministic
  `RoPE.Config.rope_key()` formatting shared by Decoder registration and layer
  lookup.
- Keep the direct non-persistent `cache` buffer and ordinary
  `_init_self_buffers()` behavior.

### 2. Add explicit runtime dependency injection

Use the existing `Config.build(**kwargs)` support for runtime objects. Do not
add module objects or a runtime collection to config dataclasses.

The Decoder creates one registered `rope_modules` collection before its existing
layer loop, then passes that same collection to each RoPE-bearing layer build:

```python
layer_config.build(rope_modules=self.rope_modules)
```

The `rope_modules` argument is construction-only. Each transformer-block
builder forwards it to its attention build when that layer has a RoPE config.
The attention constructor derives a key from its own existing `config.rope` and
fetches the canonical module:

```python
rope_key = config.rope.rope_key()
self.rope = rope_modules[rope_key]
```

The Decoder retains explicit `rope_config is None` handling for heterogeneous
architectures: layers without full attention (for example, Qwen 3.5 GDN
layers) use their normal no-keyword build path, while RoPE-bearing layers must
be built with the Decoder-owned collection. This prevents a RoPE-backed layer
from silently constructing a private duplicate cache. Direct
`RoPE.Config.build()` remains available for the RoPE primitive itself, but
attention construction resolves RoPE through the supplied collection.

This preserves each layer's existing config as the source of truth while keeping
module selection explicit and model-scoped. The key is derived at construction
time and is not stored in model config or threaded through forward calls.

Update the Decoder transformer-block constructors that can contain RoPE-backed
attention so they accept the required keyword-only
`rope_modules: ModuleDict` and forward it to their attention build:

- Llama 3
- Qwen 3
- Muse Glimmer
- DeepSeek V3
- DeepSeek V3 MTP
- GPT-OSS
- Qwen 3.5

Models with no RoPE module, such as Kimi K3, continue using the no-keyword build
path and require no synthetic dependency.

Update the RoPE-backed attention constructors to accept the same keyword-only
argument:

- common `GQAttention`
- DeepSeek V3 `Attention`
- GPT-OSS `Attention`
- Qwen 3.5 `Qwen35Attention`
- `FusedMLAAttention`, which must forward the argument to DeepSeek attention

Muse Glimmer inherits the common GQA constructor. Helion overrides replace the
RoPE config/module type and therefore require no separate cache-sharing path.

This explicit path avoids post-build injection. Post-build injection would still
construct and discard one RoPE cache per layer, leaving the duplicate-compute
review unresolved.

### 3. Keep Decoder as the sole module owner

Build `self.rope_modules = ModuleDict()` before building `self.layers`, after
runtime config updates and overrides have been applied. Register the distinct
RoPE configs needed by the main layers, then keep the existing layer
construction loop and pass the same collection into every RoPE-bearing layer:

```python
for layer_config in config.layers:
    attention_config = getattr(layer_config, "attention", None)
    rope_config = getattr(attention_config, "rope", None)
    if rope_config is None:
        continue
    rope_key = rope_config.rope_key()
    if rope_key not in self.rope_modules:
        self.rope_modules[rope_key] = rope_config.build()

for i, layer_config in enumerate(config.layers):
    attention_config = getattr(layer_config, "attention", None)
    rope_config = getattr(attention_config, "rope", None)
    if rope_config is None:
        layer = layer_config.build()
    else:
        layer = layer_config.build(rope_modules=self.rope_modules)
    self.layers[str(i)] = layer
```

The registration and layer-build decisions are intentionally inlined in the
Decoder and MTPDecoder construction paths. The Decoder owns key formatting and
module insertion:

```python
for layer_config in config.layers:
    attention_config = getattr(layer_config, "attention", None)
    rope_config = getattr(attention_config, "rope", None)
    if rope_config is None:
        continue
    rope_key = rope_config.rope_key()
    if rope_key not in self.rope_modules:
        self.rope_modules[rope_key] = rope_config.build()
```

The collection has no hand-generated `rope_0`, `rope_1`, ... keys. The
descriptive key is the module registration name and the routing contract. The
layer's existing RoPE config remains the source of truth for deriving it.

Variant identity has three separate concepts:

| Concern | Representation | Contract |
| --- | --- | --- |
| Registered owner | `self.rope_modules[rope_key]` | Decoder-created canonical module |
| Deduplication | Equal `rope_config.rope_key()` values | Key covers the effective module contract |
| Layer routing | `rope_modules[config.rope.rope_key()]` | No key is stored in model config |

Do not derive the key from tensor metadata or object identity. Use deterministic
field formatting from the RoPE config, with safe characters for PyTorch module
names. The key should include the concrete implementation class, cache alignment
(`ComplexRoPE` versus `CosSinRoPE`), dimension, context length, theta, scaling
mode, and any active subclass/scaling fields.

Each RoPE-bearing attention constructor accepts the required `rope_modules`
collection and directly obtains its already-registered module with
`rope_modules[config.rope.rope_key()]`. The lookup never builds a RoPE. A
missing key is a model-construction bug and fails immediately. The attention
stores the returned module without registering it again under the attention. Use
`object.__setattr__(self, "rope", rope)` with one comment explaining that
bypassing `nn.Module.__setattr__` keeps Decoder as the sole registered owner.

There is no process-global registry, context variable, list wrapper, namespace,
weak reference, proxy tensor, or special cache property. `ModuleDict` plus
deterministic config keys is the entire sharing mechanism. The Decoder still
handles `rope_config is None` for hybrid layers that do not use full attention;
those layers take their normal no-keyword build path.

### 4. Use the same canonical path for main and MTP layers

Keep the current construction shape for every model layer, including MTP:

- `Decoder.__init__` creates `self.rope_modules`, inlines registration of the
  distinct main-layer RoPE configs, and passes the collection to each
  RoPE-bearing main-layer `build()` call;
- `MTPDecoder.__init__` inlines registration for `config.mtp_layers`, then
  passes the same collection to each MTP-layer `build()` call;
- every RoPE-bearing attention derives `config.rope.rope_key()` and performs a
  strict `rope_modules[key]` lookup;
- hybrid layers with no full attention, such as Qwen 3.5 GDN layers, do not
  receive or use a RoPE module;
- models with no RoPE module at all retain their current construction path.

There is no protected Decoder hook, MTP context, lazy construction in the
attention builder, or configuration-mismatch validation. Heterogeneous models
use one canonical module per distinct effective RoPE key, and a missing
pre-registered module fails at construction. The `None` handling is limited to
layers whose architecture genuinely has no RoPE config.

### 6. Preserve sharding through the real owner

Each canonical `Decoder.rope_modules[...]` module is built from its nested RoPE config,
so its existing `sharding_config` must be carried onto the real owner by
`Module.Config.build()`.

Audit and update the RoPE sharding setup in:

- `torchtitan/models/common/decoder_sharding.py`
- `torchtitan/models/deepseek_v3/sharding.py`
- `torchtitan/models/gpt_oss/sharding.py`
- `torchtitan/models/qwen3_5/sharding.py`

The final contract is one registered `cache` buffer per canonical variant,
distributed once as Replicate on TP, not one DTensor per layer. Remove stale
"per-layer cache" comments. Do not add a RoPE-specific `parallelize()` override;
ordinary recursive `Module.parallelize()` must visit each child of
`Decoder.rope_modules` once.

The key must include any sharding distinction that changes the module contract,
or sharding must be normalized before registration. Two modules with the same
key must be safe to traverse and distribute as one owner.

### 7. Replicate the owner per pipeline stage

Pipeline splitting must preserve the registered `rope_modules` child on every stage,
including custom `module_fqns_per_model_part` and GraphPP paths.

The current minimal rule is in the common split path: skip the top-level
`rope_modules` child during pruning so it remains intact in every
`_split_module()` result. This avoids duplicating the rule across automatic,
custom, VLM, Muse Glimmer, eager PP, and GraphPP stage-list generation. The
registry is not added to each stage's FQN list because its children are keyed by
RoPE configuration, not by layer index.
The rule is implementation-agnostic and therefore covers stock RoPE,
HelionCosSinRoPE, and HelionComplexRoPE equally: all are registered children
under the same `rope_modules` root.

The longer-term question is whether this should become model-owned metadata,
for example a `Decoder` class attribute such as
`modules_to_keep_on_all_model_parts = ("rope_modules",)`, consumed generically
by `_split_module()`. That would let models declare other shared root modules
without hard-coding their names in pipeline code. This is intentionally left as
an open reviewer discussion rather than adding a new protocol prematurely.

`copy.deepcopy()` must produce one stage-local copy of the complete RoPE
collection and preserve every remaining attention's non-registered reference to
the corresponding copied module. Multiple virtual stages on the same rank
intentionally receive independent collections.

### 8. Document the structural compatibility boundary

Configuration structure remains unchanged: every attention config still owns
its own RoPE config copy, so config overrides and checkpoint validation continue
to use the existing paths.

Runtime module ownership changes deliberately:

- `attention.rope` remains a usable attribute and points to its selected canonical
  module;
- the registered module FQNs become `rope_modules.N` on Decoder;
- `layers.N.attention.rope` no longer appears as a registered child in
  `named_modules()` or `named_buffers()`;
- the cache remains non-persistent, so model state-dict keys do not change.

Before implementation is considered complete, audit all in-tree FQN-based
module replacement, compile, FSDP, and diagnostics paths. If an in-tree consumer
requires each per-layer RoPE to be a registered child, stop and revisit the
design rather than adding an alias registration that would reintroduce duplicate
lifecycle traversal.

## Validation

Tests, distributed runs, and numerical comparisons are intentionally deferred
until the functional construction and ownership path is settled. Do not add
test-specific compatibility branches while the object model is still changing.

## Implementation sequence

1. Rebase the draft branch onto current `origin/main` while preserving the local
   investigation documents.
2. Add the Decoder-owned `rope_modules` collection and the strict config lookup
   path to every RoPE-bearing main and MTP layer constructor.
3. Delete registry/reader/context/key code and restore ordinary RoPE buffers.
4. Update Helion, fused MLA, and every affected transformer block/attention
   constructor to use the canonical collection.
5. Reconnect and verify sharding on each registered canonical RoPE owner.
6. Make PP/GraphPP preserve `Decoder.rope_modules` on every stage.
7. Revisit tests and deterministic numerical validation after functionality is
   stable.

## Non-goals

- No process-global cache or module singleton.
- No process-global keyed multi-cache registry; the only collection is owned by
  one Decoder instance.
- No proxy tensor, reader, or custom cache property.
- No post-build cache tying or re-alias pass.
- No RoPE-specific parallelization override.
- No cache sharing across independent models or PP stages.
- No factory API for lazy cache creation; canonical modules are registered before
  their consuming layers are built.

## Exit criteria

The updated functionality is ready for review when it has one obvious
model-local owner collection, zero duplicate cache construction per effective
variant within a Decoder, no implicit construction context, no dead sharding
declarations, explicit routing for heterogeneous RoPE configs, stage-local PP
ownership, and no private-cache fallback in RoPE-backed attention construction.
