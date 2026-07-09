# Configurable Override Mechanism

Swap any `Configurable` in a torchtitan training config — a model component,
optimizer, loss, dataloader, and so on — for an alternative implementation
without editing in-repo code, by importing a module that registers an override.
This is the reference for how the mechanism works and how to write overrides.

## Motivation

Torchtitan has a config-driven build system: every component defines a nested
`Config` dataclass and `config.build()` constructs the owning object. Swapping a
component (e.g. Float8 quantization, LoRA) currently works via
`ModelConfigConverter`, which traverses the model config tree and replaces
`Config` nodes during config construction, inside `config_registry.py` functions.

That works well for in-repo, first-class features, but it requires editing the
repo for *every* alternative implementation. The override mechanism removes that
requirement for three classes of work:

1. **Hardware-specific kernels.** Efficient Triton/CUDA kernels for rotary
   embeddings, fused attention, or custom norms that target specific hardware.
   These are valuable for performance but are (a) hardware-specific, (b) harder
   to maintain, and (c) reasonably held to a lower code-quality bar than core.
   We want to offer vendors a clean extension point rather than either lowering
   the core bar or saying "no".

2. **Larger fused regions.** An efficient MoE may fuse dispatch–compute–combine
   with custom routing, communication, and quantization. This spans several
   `Config` nodes and is best expressed as a single replacement at a higher
   scope (e.g. replace `MoE.Config` wholesale).

3. **Team experimentation.** Trying a non-trivial implementation without
   threading intrusive changes through core, then keeping or discarding it
   cheaply.

### In-repo vs. external

The same mechanism serves both in-repo examples (`torchtitan/overrides/`) and
external packages, and the decision of where an implementation lives maps onto
the goals above:

- **In-repo (`torchtitan/overrides/`)** — illustrative examples and broadly
  useful kernels we are willing to maintain at a lower bar than core. They are
  opt-in and never run unless requested, so they cannot regress default
  behavior.
- **External package** — vendor- or hardware-specific code, or anything we do
  not want to maintain in-tree. It lives in the vendor's namespace and is
  activated by listing its module path.

**Non-goals.** An operator-override API in TorchTitan. To replace or add an op,
use PyTorch's custom-operator path and wrap the op in a `Module` — see
[Custom kernels and `torch.compile`](#custom-kernels-and-torchcompile).
TorchTitan does not provide an op-level override surface.

## Design Overview

Override authors register a factory against a `Configurable.Config` class. After
config construction and before any `build()`, the mechanism traverses the config
tree and replaces matching nodes with the factory's output.

### Core principles

- **Explicit opt-in.** Nothing happens unless the user lists modules in
  `override.imports`. No auto-detection from hardware or installed packages.
- **Strictly scoped to the request.** Only overrides *registered by the listed
  modules* (or their submodules) are applied — not the whole global table. This
  is enforced by provenance (the module a factory is defined in).
- **Per-instance targeting, per-node conflicts.** An override selects *which*
  matched nodes it claims via `fqns` (FQN globs). Two overrides conflict only
  when they claim the *same node* (or one claims an ancestor of the other's
  node) — not merely the same Config class. So two vendors' norm kernels on
  different layers, or A/B-ing a kernel per layer, compose cleanly; genuine
  collisions error out.
- **Any `Configurable`.** Overrides apply across the entire `Trainer.Config`
  tree — model components, optimizer, loss, dataloader, etc. — not just the
  model.
- **Minimal surface.** The whole mechanism lives in
  `torchtitan/config/override.py`, reusing the `Configurable.Config.traverse()` +
  replace pattern that the Float8/LoRA converters already use.

## How It Works

### Registration

An override module defines a replacement and registers a factory with
`@override`:

`TritonRoPE` below is **illustrative** (a compact subclass-adds-one-field
example); no `triton_rope.py` is shipped. It is a valid override: each attention
module owns a `rope` submodule (a `RoPE.Config`), so swapping in a custom RoPE is
an ordinary component override.

```python
from dataclasses import dataclass

from torchtitan.config import derive, override
from torchtitan.models.common.rope import RoPE


class TritonRoPE(RoPE):
    """RoPE applied with a Triton kernel (illustrative)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        block_size: int = 128

    def __init__(self, config: Config): ...
    def forward(self, query, key, positions=None): ...


@override("triton_rope", target=RoPE.Config,
          description="Triton rotary embedding, ~2x faster on A100+")
def triton_rope(cfg: RoPE.Config, *, block_size: int = 128) -> TritonRoPE.Config:
    # `derive` copies every field shared with RoPE.Config; the factory states
    # only its deltas. See "Deriving the replacement config" below. `block_size`
    # defaults to 128 but can be tuned per config tree via a per-entry kwarg
    # (see "Optional per-entry kwargs" below).
    return derive(cfg, TritonRoPE.Config, block_size=block_size)
```

The factory receives the matched config as its first positional argument and
returns its replacement. It may declare keyword parameters (like `block_size`
above) that callers fill via a `(module_path, kwargs)` import entry; a bare
module-path entry uses their defaults. *Which* instances the factory applies to
is controlled declaratively by `fqns` (see
[per-instance targeting](#per-instance-targeting)), not by inspecting fields
inside the factory.

### Deriving the replacement config

Prefer `derive(cfg, NewConfig, **deltas)` over hand-copying each field. It builds
`NewConfig` by copying fields shared by name with `cfg`, applying `deltas` on
top; the factory states only what it actually changes.

This matters for version-robustness. Hand-copying couples a factory to the
target's field set at the time it was written:

```python
# Fragile: RoPE.Config has many fields (theta, scaling, rope_factor, beta_fast,
# ...); hand-copying only some silently drops the rest to their defaults — and a
# field added later would be dropped too, with no error.
return TritonRoPE.Config(dim=cfg.dim, max_seq_len=cfg.max_seq_len,
                         theta=cfg.theta, scaling=cfg.scaling)

# Robust: every RoPE.Config field (including ones added later, inherited by the
# subclass TritonRoPE.Config) is copied automatically; only block_size is a delta.
return derive(cfg, TritonRoPE.Config, block_size=128)
```

Semantics: fields on both `cfg` and `NewConfig` are copied from `cfg`; fields
only on `NewConfig` come from `deltas` (else their declared default); fields only
on `cfg` are dropped; a delta naming a field that doesn't exist on `NewConfig`
raises. Copying is shallow (like `dataclasses.replace`), which is fine because
the matched node is being replaced and detached. `derive` is most valuable when
the replacement subclasses the target (the common drop-in case), where new core
fields are inherited and thus carried over for free.

**Configuration is code.** By default the override module *is* its
configuration: the factory bakes in sensible defaults (above, `block_size=128`),
so a bare `override.imports` entry needs no other input.

**Optional per-entry kwargs.** When two config trees need the *same* override
module configured *differently*, a factory can expose keyword parameters and each
`override.imports` entry can supply them as a `(module_path, kwargs)` tuple
instead of a bare string; the `kwargs` are forwarded to that module's factories.
For `triton_rope` above, two trees can pick different block sizes without a
second module:

```python
# one tree tunes block_size=128 (the default), another 256:
OverrideConfig(imports=["my_pkg.triton_rope"])                      # -> 128
OverrideConfig(imports=[("my_pkg.triton_rope", {"block_size": 256})])
```

(The RL trainer and generator use this to activate one HybridEP dispatch module
with opposite `capacity_factor` values — blocking `None` for the trainer, a float
for the cudagraph-capturing generator — instead of two modules or a hardcoded
per-actor branch.)

The factory declares the keyword parameters it accepts (or `**kwargs`); a kwarg
it does not accept raises rather than silently no-op'ing. On the CLI, attach
kwargs to the module name as `module=<json-object>` (quote it as a single shell
token; `my_pkg.triton_rope` is a placeholder for your own module):

```bash
torchtitan_train --module llama3 --config llama3_8b \
    --override.imports 'my_pkg.triton_rope={"block_size": 256}'
```

JSON keeps the values typed (numbers, `null`, strings, nesting), so the same
entry works from Python or the CLI. We still avoid a typed override-`Config`
parsed at config-construction time, which would re-couple the config registry to
the override package and defeat the no-touch goal.

### Activation

```bash
# One or more modules, space- or comma-separated:
torchtitan_train --module llama3 --config llama3_8b \
    --override.imports torchtitan.overrides.fused_swiglu

# A module with per-entry kwargs -- attached to the name as module=<json>,
# quoted as one shell token (my_pkg.triton_rope is a placeholder for your module):
torchtitan_train --module llama3 --config llama3_8b \
    --override.imports 'my_pkg.triton_rope={"block_size": 256}'
```

### Application

In `Trainer.__init__`, after `model_config.update_from_config()` (which sets
sharding config on the pre-override modules) and before any component is built:

1. Import the listed modules — this triggers their `@override` decorators.
2. Resolve the *active* set: overrides whose defining module is one of the
   listed imports (or a submodule of one).
3. Collect claims: traverse the original tree and record every `(override,
   node)` pair, filtered by each override's `fqns` selector.
4. Check for per-node conflicts (two overrides claiming the same node, or one
   claiming an ancestor of another's node).
5. Apply: build each replacement and swap it in. Because claims are gathered
   before any mutation, application is order-independent (replacements are never
   re-traversed).
6. Log every replacement.

```
INFO: [Override] fused_swiglu: model_spec.model.layers.0.feed_forward FeedForward.Config -> FusedSwiGLU.Config
INFO: [Override] fused_swiglu: model_spec.model.layers.1.feed_forward FeedForward.Config -> FusedSwiGLU.Config
...
INFO: Applied 32 override(s)
```

The model config is reached even though it is nested under a non-`Configurable`
`ModelSpec`: `ModelSpec.traverse` exposes its `model` entry to the traversal.
FQNs are kept as full paths from the `Trainer.Config` root — the model config is
`model_spec.model` and a component is `model_spec.model.layers.0.feed_forward`.
Preserving the full path (rather than resetting to bare model names) is what lets
per-node conflict detection recognize a whole-model override as an ancestor of a
component override. (This differs from converter `filter_fqns`, which are bare
because converters traverse the model config directly.)

### External packages

```python
# vendor_x/overrides.py
from torchtitan.config import override
from torchtitan.models.common.moe import MoE


@override("vendor_x_fused_moe", target=MoE.Config,
          description="Vendor X fused MoE for XPU")
def vendor_x_moe(cfg: MoE.Config) -> "VendorXFusedMoE.Config":
    return VendorXFusedMoE.Config(num_experts=cfg.num_experts, ...)
```

```bash
pip install torchtitan-vendor-x
torchtitan_train --module deepseek_v3 --config dsv3_671B \
    --override.imports vendor_x.overrides
```

### Version compatibility

An override depends on the internal `Config` shapes it reads and the modules it
imports, so external packages should:

- **Pin a torchtitan version range** they have validated against. The override
  contract is the public `Configurable.Config` fields of the targets; these can
  change across releases.
- **Align their pytorch-nightly expectation** with the torchtitan version they
  pin, since hardware kernels are typically nightly-sensitive. Document the
  tested torchtitan × pytorch pair in the package.

## Per-instance Targeting

Targeting is first-class, not an afterthought: many motivating cases collide by
design (two vendors with norm kernels, or A/B-ing a kernel per layer). The
`fqns` selector on `@override` says *which* matched nodes an override claims,
declaratively — no field-sniffing inside the factory.

`fqns` is one of:

- `None` (default) — every instance of `target`.
- a list of FQN globs — `fnmatch` style, where `*` also crosses `.`; a node is
  claimed if any glob matches its FQN.

By default, `target` matches instances of the target config class and its
subclasses. If a replacement only implements the concrete target contract, pass
`exact=True` to `@override`; subclass configs are then not claimed, so they are
not logged as replacements and can still be handled by a subclass-specific
override.

The FQN is the full path from the `Trainer.Config` root, e.g. a model component
is `model_spec.model.layers.0.feed_forward` and the optimizer is `optimizer`.
Globs with `*` (which crosses `.`) keep selectors readable.

```python
# NVFP4 on every in-layer linear, leaving the top-level LM head ("...output")
# untouched (required for NVFP4; useful for MXFP8)
@override("nvfp4_linear", target=Linear.Config,
          fqns=["*.layers.*.attention.*", "*.layers.*.feed_forward.*"])
def nvfp4_linear(cfg: Linear.Config) -> "NVFP4Linear.Config":
    return NVFP4Linear.Config(in_features=cfg.in_features, ...)

# Fused MoE only on the later layers
@override("vendor_moe", target=MoE.Config, fqns=["*.layers.1[0-9].moe"])
def vendor_moe(cfg: MoE.Config) -> "VendorMoE.Config":
    return VendorMoE.Config(num_experts=cfg.num_experts, ...)
```

`fqns` accepts globs only. A general predicate selector (`(fqn, cfg) -> bool`),
which would also enable field/type-based selection (e.g. "skip already-quantized
linears"), is listed under [Future work](#future-work).

### Per-node conflict detection

Conflicts are detected per *node*, not per *class*. Two overrides may target the
same Config class freely as long as their claimed nodes are disjoint — so one
norm kernel on `*.layers.0.*` and another on `*.layers.1.*` coexist, and A/B-ing
a kernel across layers works. An error is raised only when two different overrides
claim the **same node**, or when one claims an **ancestor** of another's node
(nested replacement, which would be order-dependent). The fix in both cases is
to narrow the `fqns` selectors so the claims are disjoint.

Claims are gathered from the **original** tree before any mutation, and a
replacement is never re-traversed, so application is order-independent — one
override can never silently affect what another matches.

#### Parent- and child-class overrides

This same rule answers the parent/child question directly. Say override A
targets a parent Config (e.g. `MoE.Config`) and override B targets a Config
nested inside it (e.g. `GroupedExperts.Config`):

- **Disjoint subtrees** (A on `...layers.0.moe`, B on
  `...layers.1.moe.routed_experts.inner_experts`) — the claimed nodes are
  unrelated, so both apply.
- **Overlapping** (A on `...layers.0.moe`, B on
  `...layers.0.moe.routed_experts.inner_experts`) — B's node is inside A's, the
  ancestor case above, so we **error**.

We error rather than pick one of the two plausible behaviors implicitly:

1. *Apply the parent, then override a child in the new parent.* Only meaningful
   if A's replacement still exposes a matching child, and it reintroduces
   ordering (parent-before-child) and re-traversal — which we deliberately avoid.
2. *Silently drop the child override* (when A's replacement has a different
   structure with no such child). A silent no-op is exactly the kind of
   surprise that hides bugs.

Because claims are collected on the original tree, we can't even tell at
check time which case a given pair would fall into — so instead of guessing, we
ask the author to disambiguate: narrow `fqns` so the claims are disjoint, or drop
one override (e.g. let the parent override subsume the child).

## Scope: any `Configurable`

Overrides traverse the whole `Trainer.Config`, so the optimizer, loss,
dataloader, and validator configs are overridable too — not only model
components. For example, an emerging or mixed-precision optimizer can be swapped
in by targeting `OptimizersContainer.Config` without editing a config registry
function. The model config is reached via `ModelSpec.traverse` (above); the model
config itself is a valid target (whole-model swap), while `ModelSpec` is not — a
`target` must be a `Configurable.Config` subclass, so a plain class like
`ModelSpec` is rejected at registration.

## Interaction with Converters

In-repo converters (Float8, LoRA via `ModelConfigConverter`) run *first*, inside
`model_registry()` during config construction; overrides run later in
`Trainer.__init__` and see the post-converter tree. The order is deliberate: an
in-repo converter cannot be expected to understand arbitrary external overrides,
so it runs against the known core configs, and overrides layer on top.

Conversely, converter-style transforms *can* be expressed as overrides (a
factory that rewrites a `Config`), so external code does not need the converter
machinery.

| Override Target | Conflicts with a converter? | Notes |
|-----------------|------------------------------|-------|
| RoPE / FeedForward / MoE / RMSNorm / inner attention | No | Converters don't touch these |
| GroupedExperts.Config | Possibly | `Float8GroupedExpertsConverter` rewrites this |
| Linear.Config | Yes | Float8/LoRA replace these |

Where a converter already rewrote a node, target that node by location with
`fqns` so the override only claims the instances you intend (e.g. specific
layers the converter left as plain `Linear.Config`). Field/type-based exclusion
— "skip instances already turned into `Float8Linear.Config`" — is not expressible
with FQN globs alone and motivates the future predicate selector.

### Override vs. converter: which to use

Converters are the right home for **core, broadly-applicable transforms that
preserve a property we guarantee** (e.g. batch-invariance, which is largely
op-level and must hold across the model). Overrides exist for an *unblocking*
purpose: swapping in an alternative implementation. By design we make **no
promise that any core property survives an arbitrary override** — that is the
override author's responsibility. So features like batch-invariant mode stay
converters, not overrides.

## Checkpoint Compatibility

An override that changes a module's parameter layout changes its checkpoint
FQNs. By default an override checkpoints whatever real parameters it defines, so
a fused module that stores a single `w13` parameter would save/load
`...feed_forward.w13` rather than the stock `w1.weight` / `w3.weight`.

**Bridge layout differences with module-level `state_dict` hooks.** A replacement
module can present its weights in the *stock* layout by registering two hooks:

- `register_state_dict_post_hook` to split/rename its real parameters into the
  stock FQNs on save, and
- `register_load_state_dict_pre_hook` to recombine them before the default load,
  so the real parameter is loaded with normal DTensor/`strict` handling.

The fused example (`fused_swiglu.py`) does exactly this: it stores `w13` but
checkpoints `w1.weight` / `w3.weight`, so its checkpoints are a drop-in for the
stock `FeedForward` (and for the HF adapter, which targets the stock layout),
while still accepting a native `w13` key for back-compat. This is the symmetric
use of the same hook mechanism the activation-checkpoint wrapper uses to strip
its `_checkpoint_wrapped_module` prefix.

For mappings too complex for module hooks, a model-level `BaseStateDictAdapter`
(the mechanism used for HF conversion, e.g. `Llama3StateDictAdapter`) remains an
option; it transforms the flat key->tensor dict from `state_dict()`.

## Parallelism

Parallelism is expressed entirely through the `Module` protocol: a replacement
satisfies `init_states` and declares a `ShardingConfig` for the states and
activations it wants sharded. `Module.parallelize()` reads that `ShardingConfig`
exactly as it does for core modules, so an override composes with TP/FSDP by
declaring its own sharding — nothing model-specific is required, and the
mechanism deliberately stays config-driven rather than depending on imperative
per-model parallelization code.

One thing worth stating plainly:

- **Fusion under TP.** Fusing weights can interact subtly with tensor
  parallelism — the fused tensor's layout must admit a correct shard. A *flat*
  fused gate+up weight `(2*hidden, dim)` has none (`spmd.S(0)` would hand one rank
  all of `w1` and another all of `w3`). The fix is a layout whose TP-sharded axis
  gives each rank matching slices: `fused_swiglu` stores `w13` as
  `(hidden, 2, dim)` and shards `spmd.S(0)` on `hidden`, so each rank holds
  `(hidden/tp, 2, dim)` — a slice of both gate and up (the Megatron
  column-parallel layout). `hidden` is also dim 0, so FSDP shards it cleanly at
  any degree. The single GEMM is an `einsum` that keeps `hidden` sharded, so it
  never reshapes across the sharded axis. This composes with FSDP and TP through
  the ordinary `ShardingConfig`; no model-specific code.

## Custom kernels and `torch.compile`

An override that wraps a custom CUDA or Triton kernel must stay compatible with
`torch.compile`, because torchtitan compiles the transformer blocks by default
(`compile.components` includes `"model"`). A raw kernel call is opaque to Dynamo
and will graph-break or fail to trace. Making it compose is the override
author's responsibility — the mechanism deliberately adds **no** TorchTitan
operator-override API. Instead, register the kernel as a first-class PyTorch
custom operator, which gives `torch.compile` (and export) a concrete contract:

- **`torch.library.custom_op`** — wrap a Python/CUDA-extension kernel as an op
  with a stable schema and a clear functional/mutation contract.
- **`torch.library.triton_op`** — register a Triton-kernel-backed op so the
  kernel is visible to (and traceable by) `torch.compile` rather than a black
  box.
- **fake / meta kernel** (`register_fake`) — a shape/dtype-only implementation
  so tracing, `torch.compile`, and export can propagate metadata without running
  the kernel.
- **autograd registration** (`register_autograd`) — supply the backward so the
  op works in training, not only inference.
- **`torch.library.opcheck`** — test schema / fake / autograd consistency; a
  good unit test to ship in the override package.

The override module then wraps the registered op in its replacement `Module`,
and TorchTitan's mechanism keeps operating purely at the `Configurable` / Module
level. See PyTorch's "Custom Operators" landing page and the Triton-op tutorial
for the full recipe.

## Composability Notes

- **Context parallelism** is applied on the Attention module. An override of the
  attention block (or its inner attention) must remain compatible with that
  application point; this is a composability surface to respect.
- **Agentic / automated search.** Making model impls, optimizers, and kernels
  uniformly swappable at config time is a building block for automated
  hill-climbing over implementations. This mechanism, the GraphTrainer registries,
  and kernel-fusion prototypes are complementary; keeping the override surface
  small and declarative is what makes programmatic swapping tractable.

## Worked Examples

- `torchtitan/overrides/fused_swiglu.py` — **the parametrization example.** A
  fused SwiGLU feed-forward demonstrating custom `__init__` parametrization (one
  fused `(hidden, 2, dim)` `w13` weight, one GEMM), `param_init`, and a
  `sharding_config` that composes with both FSDP and TP (see "Fusion under TP"
  above). Needs no prerequisite. See "Checkpoint Compatibility" for why it is not
  interoperable with stock checkpoints.
- `torchtitan/overrides/helion_rope.py` — **the custom-kernel example.** Swaps
  `CosSinRoPE` for a fused Helion kernel (forward + backward) wrapped in a
  `torch.library.custom_op` (with `register_fake` / `register_autograd`), the
  recipe from "Custom kernels and `torch.compile`". `helion` is an optional
  dependency, so the module imports without it and falls back to the PyTorch RoPE
  when it (or CUDA) is unavailable; it is checkpoint-compatible with stock.
The `TritonRoPE` snippets above are illustrative — no `triton_rope.py` is
shipped — but RoPE is a fully valid override target (`helion_rope.py` is a real
one): each attention module owns a `rope` submodule (`RoPE.Config`), so a custom
RoPE is an ordinary component override.

## Code map

| File | Role |
|------|------|
| `torchtitan/config/override.py` | The mechanism: `OverrideConfig`, `Override`, `override`, `derive`, `apply_overrides`, `clear_overrides`. |
| `torchtitan/config/__init__.py` | Re-exports the override API. |
| `torchtitan/protocols/model_spec.py` | `ModelSpec.traverse` exposes the nested model config to the traversal. |
| `torchtitan/trainer.py` | Holds the `override` config field; applies overrides after `update_from_config`, before builds. |
| `torchtitan/overrides/` | In-repo example implementations (`fused_swiglu.py`, `helion_rope.py`). |
| `tests/unit_tests/test_override.py` | Unit tests: registration, provenance, FQN / exact targeting, per-node conflicts, per-entry kwargs, `derive`. |

Overriding a component requires no changes to any model's `config_registry.py`
or `__init__.py` — that is the point of the design.

## At a Glance

- **Activation.** Explicit opt-in via `override.imports`; only overrides
  registered by the listed modules (provenance-checked) are applied. An entry
  may carry kwargs (a `(module_path, kwargs)` tuple in Python, or
  `module=<json>` on the CLI) to configure the same module differently per
  config tree.
- **Conflicts.** Per-node, not per-class: overrides may share a target class as
  long as their claimed nodes are disjoint; same-node or ancestor/descendant
  claims error.
- **Per-instance targeting.** Via `fqns` (FQN globs) on `@override`; the factory
  is pure construction.
- **Subclass matching.** Targets match subclasses by default; use `exact=True`
  when a replacement only supports the concrete target config.
- **Scope.** Any `Configurable` in the `Trainer.Config` tree.
- **Config construction.** Use `derive(cfg, NewConfig, **deltas)` so factories
  don't silently drop fields added to the target later.
- **Override-specific config.** Via a small custom module; no new config surface.
- **Checkpoint.** Layout-changing overrides can stay stock-interoperable by
  bridging FQNs with `register_state_dict_post_hook` /
  `register_load_state_dict_pre_hook` (see Checkpoint Compatibility).

## Future Work

- **Predicate `fqns` selector.** A `(fqn, cfg) -> bool` selector to complement
  the glob strings, enabling field/type-based selection (e.g. skipping linears a
  converter already turned into `Float8Linear.Config`).
