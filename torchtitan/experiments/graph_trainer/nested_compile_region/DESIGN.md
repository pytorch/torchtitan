# Dynamo-Free `nested_compile_region` Capture in TorchTitan

Status: Proposed

## Summary

TorchTitan's `minimal_fx_tracer` should be able to preserve functions marked with
`torch.compiler.nested_compile_region` as `invoke_subgraph` higher-order operator
(HOP) calls without using Dynamo for frame capture, guard generation, or module
state lifting.

The proposed implementation has two parts:

1. PyTorch provides a context-local handler for
   `invoke_subgraph_placeholder`. Outside Dynamo or export, the existing eager
   fallback consults this handler before calling the marked function inline.
2. TorchTitan installs a per-trace handler around `make_fx`. The handler lifts a
   module's parameters and buffers into explicit HOP operands, reparameterizes a
   representative module inside the nested body, and caches bodies using the
   marked function's `reuse_hash_fn`.

For the TorchTitan Llama debug model, marking
`Llama3TransformerBlock.forward` once applies to all six block instances. With
`reuse_hash_fn=lambda block, *args: 0`, the expected graph contains six forward
calls to one shared forward body and six backward calls to one shared backward
body. Each call supplies that block's distinct parameters and RoPE buffer.

## Motivation

`nested_compile_region` currently relies on Dynamo to provide two important
services:

1. Determine when marked calls may reuse a previously captured graph.
2. Lift module parameters and buffers out of the nested graph.

TorchTitan's graph trainer starts from `make_fx` through `minimal_fx_tracer`, not
from Dynamo. Today, the eager implementation of `invoke_subgraph_placeholder`
therefore calls the marked function normally and `make_fx` inlines it into the
outer graph.

This loses the compile-time benefit of representing repeated transformer blocks
once. It also makes the graph much larger before downstream compilation.

TorchTitan already has the machinery needed for state lifting at the root model:

- `extract_module_state` returns parameters and buffers with
  `remove_duplicate=False`.
- `_reparametrize_train_state` installs fake, explicit state while tracing.
- `minimal_fx_tracer` exposes all model state as leading graph inputs.

The nested-region implementation should reuse the same model rather than create
a second state-management system.

## Goals

- Preserve marked regions as `invoke_subgraph` nodes under
  `minimal_fx_tracer` without Dynamo frame tracing.
- Lift parameters and buffers from module methods into explicit HOP operands.
- Reuse one nested body across structurally equivalent module instances with
  distinct state.
- Support training graphs, including lazy `invoke_subgraph` backward capture.
- Preserve positional arguments, keyword arguments, and pytree outputs.
- Make reuse decisions deterministic within one `minimal_fx_tracer` invocation.
- Fail clearly when a region cannot be represented safely.
- Keep eager behavior unchanged outside the explicit tracing context.

## Non-goals

- Reimplement Dynamo's general Python guard system.
- Detect arbitrary semantic equivalence between Python functions or modules.
- Make a traced graph valid after users mutate module topology or static config.
- Initially support mutation, input/output aliasing, or arbitrary Python side
  effects inside a region.
- Initially make every existing TorchTitan graph pass recurse into HOP bodies.
- Initially support non-`None` `NestedCompileRegionOptions` or regional Inductor
  configuration.

## User Experience

### Marking all instances of a block

The marker belongs on the shared class method:

```python
class Llama3TransformerBlock(TransformerBlock):
    @torch.compiler.nested_compile_region(
        reuse_hash_fn=lambda block, x, attention_masks, positions=None: 0,
    )
    def forward(self, x, attention_masks, positions=None):
        ...
```

All six `Llama3TransformerBlock` instances use this method, so all six calls are
marked. Users do not decorate six bound methods separately.

The equivalent dynamic application is:

```python
Llama3TransformerBlock.forward = torch.compiler.nested_compile_region(
    reuse_hash_fn=lambda block, *args, **kwargs: 0,
)(Llama3TransformerBlock.forward)
```

Assigning a wrapper only to `model.layers["0"].forward` marks only that instance.
It does not imply discovery of sibling modules.

### Enabling capture in TorchTitan

The initial integration should be explicit:

```python
traced = minimal_fx_tracer(
    train_step,
    module=model,
    capture_nested_compile_regions=True,
)(tokens, labels)
```

An opt-in limits the compatibility surface while graph passes learn to handle
opaque region bodies. Once those passes are compatible, the option can default
to true because an unmarked model has no behavioral change.

## Reuse Semantics

### Use the existing integer key

The no-Dynamo path should use the existing `reuse_hash_fn` API instead of adding
a boolean module predicate.

An integer key expresses more than a boolean:

```python
# All blocks are equivalent.
reuse_hash_fn=lambda block, *args: 0

# Two alternating block families.
reuse_hash_fn=lambda block, *args: block.variant_id
```

Calls to the same raw function with the same key are candidates to share a
body. Calls with different keys always get different bodies.

A predicate such as `lambda module: bool` only creates two groups and does not
state what a candidate is being compared with. A two-module predicate would
require an order-dependent, linear search through representatives. The existing
integer key is simpler and already part of the public decorator contract.

### No cross-execution guard cache is required

`minimal_fx_tracer` captures one known model execution. Reuse is selected among
call sites observed during that trace. At runtime, each call site is already
wired to a specific nested body.

This differs from Dynamo, which caches compiled frames across independent Python
executions and therefore needs runtime guards. A `make_fx` graph is already
invalid if users change the model topology or static Python configuration after
capture.

The no-Dynamo implementation still needs trace-time compatibility checks so a
bad key does not silently connect incompatible call sites.

### Compatibility signature

Each cache entry is indexed by:

```text
(
    raw_function_identity,
    reuse_key,
    module_type,
    module_state_schema,
    input_pytree_spec,
    dynamic_operand_metadata,
    captured_static_values,
)
```

`module_state_schema` includes:

- Ordered parameter and buffer names.
- Parameter versus buffer kind.
- Tensor shape, stride, dtype, device type, layout, and `requires_grad`.
- State alias groups for tied or repeated state.

`dynamic_operand_metadata` includes the same tensor metadata for non-state
operands. Symbolic shapes should compare symbolically within the active
`ShapeEnv`, rather than by converting symbols to concrete integers.

Static leaves that cannot be HOP operands are captured in the body and included
in the compatibility signature.

If two calls have the same user key but incompatible signatures, raise a reuse
key collision error. A user-provided key is an assertion that calls are
equivalent; silently creating a variant hides an incorrect assertion and
diverges from the public contract that equal keys reuse one entry.

`max_reuse_entries` limits the number of distinct keys for a raw function.
Hitting the limit raises an error that lists existing keys and the new key.

The user key remains responsible for static module attributes that cannot be
derived from tensor state, such as attention mode, head count, or Python control
flow flags. At minimum, the implementation validates identical module types.

When `reuse_hash_fn` is absent, the initial implementation should use module
identity and should not reuse across module instances. Automatic semantic reuse
without a user key would require either Dynamo-like guards or tracing and
comparing every candidate graph.

## Proposed Architecture

```text
minimal_fx_tracer
  |
  | installs context-local marked-region handler
  v
make_fx(fn_with_subclass_handling)
  |
  | calls a @nested_compile_region method
  v
invoke_subgraph_placeholder(raw_fn, module, *args, **kwargs)
  |
  | handler extracts current module state
  | handler selects or creates RegionEntry
  v
invoke_subgraph(region_body, stable_identifier, *state, *operands)
  |
  | ProxyTensor re-enters make_fx once per identifier
  v
outer FX graph with repeated invoke_subgraph call sites
```

### 1. Context-local PyTorch hook

The prototype replaces the module-global `invoke_subgraph_placeholder` during
the trace. That proves the data flow, but it is process-global and unsafe when
two threads trace concurrently.

The production implementation should add a private `ContextVar` handler in
`torch/_higher_order_ops/invoke_subgraph.py`:

```python
_nested_region_handler = ContextVar("nested_region_handler", default=None)

@contextlib.contextmanager
def _set_nested_region_handler(handler):
    token = _nested_region_handler.set(handler)
    try:
        yield
    finally:
        _nested_region_handler.reset(token)
```

`invoke_subgraph_placeholder` keeps its existing Dynamo and export behavior,
then checks the handler before its eager fallback:

```python
handler = _nested_region_handler.get()
if handler is not None:
    return handler(func, args, kwargs)
return func(*args, **kwargs)
```

This makes the interception scoped, nestable, and thread-local. TorchTitan
already disables autograd multithreading during the relevant `make_fx` trace,
which avoids losing the context during backward capture.

This does not make all of `minimal_fx_tracer` concurrently safe. FX traceback
metadata and live-module reparameterization have process-global or mutating
behavior today. TorchTitan should either retain its existing single-trace
assumption or place trace preparation, reparameterization, make_fx, backward,
and cleanup under a reentrant trace lock. Runtime `run_traced` does not need the
lock.

If a PyTorch change is not available, TorchTitan can temporarily use the scoped
global replacement from the proof, but that path must remain experimental and
must be serialized with a process-wide lock.

### 2. Per-trace registry

Create a new `NestedRegionRegistry` inside `_trace_with_args`, not in the outer
`minimal_fx_tracer` factory. Its lifetime must exactly match one `make_fx` trace.

This provides:

- Deterministic identifiers starting at zero for each trace.
- No references to stale FakeTensors or ShapeEnvs.
- No representatives retained after capture.
- No cache interaction between unrelated models.

A `RegionEntry` contains:

```python
@dataclass
class RegionEntry:
    identifier: str
    representative: nn.Module
    body: Callable
    input_spec: pytree.TreeSpec
    output_spec: pytree.TreeSpec | None
    state_schema: ModuleStateSchema
    signature: RegionSignature
```

### 3. Module state lifting

For the first supported form, require the first positional argument to be an
`nn.Module`. This covers normal marked methods such as
`Llama3TransformerBlock.forward(self, ...)` and gives failures a clear boundary.

The handler performs these steps:

1. Extract ordered state with `named_parameters(remove_duplicate=False)` and
   `named_buffers(remove_duplicate=False)`.
2. Record relative state names and the alias pattern.
3. Remove the module object from dynamic operands.
4. Flatten the remaining `(args, kwargs)` pytree.
5. Pass state tensors followed by supported dynamic leaves to `invoke_subgraph`.

The outer `minimal_fx_tracer` has already reparameterized the root model with
fake state derived from top-level graph inputs. Therefore, state read from a
block at this point refers to those fake inputs, not to constants from the live
module. Passing it to the HOP preserves the connection to top-level model state.

The nested body closes over the first module instance as a representative:

```python
def body(*flat_operands):
    state, call_operands = split_operands(flat_operands)
    args, kwargs = reconstruct_call(call_operands)
    with _reparametrize_module(representative, bind_state(state)):
        result = raw_fn(representative, *args, **kwargs)
    flat_result, output_spec = pytree.tree_flatten(result)
    return tuple(flat_result)
```

The representative supplies module topology and static Python attributes. Its
parameters and buffers are replaced for every call. Consequently, six calls
share Python and FX structure while consuming six distinct sets of state.

The `LeafModuleState` and `_reparametrize_module` flow in
`torch/_higher_order_ops/invoke_leaf_function.py` is the closest existing
make_fx-native model and should be reused where practical.

### 4. Argument flattening

The proof supports positional arguments only. Production capture must flatten
`(args, kwargs)` and preserve a `TreeSpec`.

Dynamic HOP operands may contain the types accepted by `invoke_subgraph`,
including tensors, integers, symbolic integers, generators, opaque objects, and
`None`.

Other make_fx-safe values, such as strings and floats, should be treated as
static values:

- Store them in the adapter.
- Include them in the compatibility signature.
- Reinsert them while reconstructing the function call.

Initially reject unsupported custom objects with a message naming the pytree
path and type. `BlockMask` support can build on TorchTitan's existing pytree
registration, but needs a dedicated test because it contains Python callables
as well as tensors.

### 5. Output flattening

`invoke_subgraph` bodies must return a flat tuple, including for a single tensor
output. The handler records the first body's output `TreeSpec` and reconstructs
the user-visible result after the HOP call.

All calls sharing an entry must have the same output spec. The initial version
should support tensor, integer, symbolic integer, and `None` leaves and reject
other output leaf types.

### 6. Stable training identifier

The training path must call `invoke_subgraph` with a stable identifier. It
should not call `invoke_subgraph_infer`, whose contract is inference-only and
does not provide the stable identity needed for forward/backward caching.

For an entry named `nested_region_0`, existing `invoke_subgraph` autograd logic
produces identifiers such as:

```text
fw_nested_region_0
bw_nested_region_0_0
```

Backward caching additionally considers incoming tangent metadata. If tangent
layouts differ, multiple backward variants are correct even when the forward
body is shared.

### 7. Mutation and alias checking

The initial implementation should require pure regions:

- No mutation of inputs or module state.
- No outputs aliasing inputs.
- No closure-captured tensors requiring gradients.
- No observable Python side effects.

`invoke_subgraph` functionalization has mutation support, but its implementation
notes that aliasing and mutation normally must be screened while forming the
HOP. The no-Dynamo path must not assume Dynamo performed that screening.

Use the existing HOP schema/materialization utilities to validate a newly traced
body. Until mutation semantics are tested end to end, reject a body whose schema
reports mutation or aliasing.

## Metadata and Diagnostics

Each HOP call site should carry:

- Stable region identifier.
- Reuse key.
- Absolute module FQN for that call, such as `layers.3`.
- Module type.
- Ordered mapping from relative state names to top-level state operands.
- Existing `call_id` used to pair forward and backward calls.

The shared body should carry relative module metadata. A consumer can combine
the call-site FQN with body-relative names without cloning the body six times.

Emit structured trace artifacts for:

- Region cache hit and miss summaries.
- Signature mismatch details.
- Final identifiers and call-site FQNs.
- Expanded nested graph text when capture fails.

The error for an incompatible forced reuse should name both call sites and the
first mismatched field, for example a buffer name, dtype, or static config key.

## TorchTitan Integration

Add a keyword-only option to `minimal_fx_tracer`:

```python
def minimal_fx_tracer(
    fn,
    module=None,
    optimizer=None,
    *,
    capture_nested_compile_regions=False,
    ...,
):
```

Inside `_trace_with_args`:

1. Construct the per-trace registry after creating the FakeTensor mode.
2. Install the PyTorch context-local handler in the existing context stack that
   wraps `make_fx`.
3. Let the root `_reparametrize_train_state` run before marked methods execute.
4. Dispose of the registry when tracing returns or raises.
5. Store a serializable region summary in `TracedResult` or `gm.meta`.

Do not put module objects, Python callbacks, or FakeTensors from a different
ShapeEnv in serialized graph metadata.

## Interaction With Graph-Trainer Passes

A whole transformer block becomes opaque at the outer graph level. Operations
inside it are visible in `repeated_subgraph0`, but not while a pass scans only
`traced.gm.graph.nodes`.

This affects passes that currently reason about a flat graph, including:

- FSDP collective scheduling and parameter-order analysis.
- Selective activation rematerialization and CPU offload.
- Expert-parallel transformations.
- Graph pipeline partitioning and split backward.
- Passes that find layer boundaries from stack traces or FQNs.

The initial feature should run only with a documented pass-light configuration.
Production enablement requires one of these strategies per pass:

1. Treat each HOP as an indivisible layer boundary.
2. Recurse into shared bodies and operate on relative FQNs.
3. Run the pass on a body before it is installed in the outer graph.
4. Explicitly reject the combination with a clear error.

Cloning the shared body once per call site defeats the main compile-time goal
and should not be the default compatibility strategy.

Cudagraph compatibility checks must also recurse into child graphs; an unsafe
operation hidden in a HOP cannot be treated as safe because the outer scan did
not see it. Precompile serialization requires a round-trip test and must reject
unserialized callbacks or region options.

Full Inductor compilation is the best first terminal mode because it already
walks graph modules recursively. This does not remove the need to gate earlier
flat-graph scheduling and transformation passes.

## Rejected Alternative: Nested Dynamo Compilation

PyTorch can compile a selected inner callable with
`backend="invoke_subgraph"` while outer make_fx runs with
`force_compile_during_fx_trace=True`. Existing PyTorch tests cover this path.

It is not the selected design because it still invokes Dynamo for each selected
region. The goal here is for `minimal_fx_tracer` to form the HOP and lift module
state directly, with no Dynamo bytecode or guard capture.

## `NestedCompileRegionOptions`

The initial no-Dynamo implementation should reject `options is not None` rather
than silently dropping it.

Later support must copy `NestedCompileRegionOptions` onto the traced body
`GraphModule.meta["nested_region_config"]` and the HOP node's custom metadata.
That is required for regional Inductor compilation and partitioner selection.

