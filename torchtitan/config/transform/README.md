# Model config transforms

A model config transform rewrites a complete model config tree. `build_model_config`
builds the base model before transforms run.

Transforms are one supported way to build and maintain configs. They are
optional, and users may build recipes with their own utilities.

## Using transforms

Set all training options first. Then call `apply_transforms` once.

```python
config = muse_glimmer_30b()
config.parallelism.context_parallel_degree = 8

config = apply_transforms(
    config,
    [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
)
```

Common attention and feed-forward configs contain synchronous tensor-parallel
projection roles by default. To overlap those collectives with their adjacent
GEMMs, select the asynchronous implementations with a transform:

```python
config.parallelism.tensor_parallel_degree = 8
config = apply_transforms(
    config,
    [
        AsyncTensorParallelTransform(
            enable_sequence_parallel=config.parallelism.enable_sequence_parallel
        )
    ],
)
```

Without a TP mesh, the synchronous projection classes behave as ordinary
linear modules.

The async TP transform matches projection owner classes exactly. It converts
`ColumnParallelLinear` and `RowParallelLinear`, but leaves arbitrary subclasses
unchanged because replacing one with an async base class would discard its
specialized forward behavior. `SharedExpertRowParallelLinear` is an explicit
exception: async TP requires sequence parallelism, and in that mode its
reduction is identical to `RowParallelLinear`, so it is safely converted to
`AsyncRowParallelLinear`. With sequence parallelism disabled, constructing the
async TP transform is an error before any conversion occurs.

`apply_transforms` deep-copies the trainer config. It orders and applies the
transforms, then validates the result. It returns the changed copy. The input
config stays unchanged if a transform fails.

Legacy `ModelConfigConverter` instances passed to `build_model_config` run before
all model config transforms. In particular, apply quantization in
`build_model_config` before applying `LoRATransform`; running a converter over a
LoRA-transformed tree can replace an adapter config.

NOTE: With quantization followed by LoRA, LoRA freezes the original weights,
but the current quantized linear implementations still regenerate quantized
weight operands on every forward or FSDP unshard. This is avoidable work for
frozen weights. TODO: Cache their quantized operands across forwards.

Synchronous tensor-parallel boundaries compose with quantization and LoRA.
Their converters replace the projection computation while preserving its
column- or row-parallel role. Async tensor parallelism does not yet support
converted projections; it conflicts with `LoRATransform` and rejects
quantized projection configs.

Use `transform_model_config_` when there is no trainer config. It rewrites the
model config in place and returns the root. It does not copy or validate the
config.

```python
model_config = build_model_config("0.6B", attn_backend="varlen")
model_config = transform_model_config_(model_config, [LMHeadCastTransform()])
```

## What belongs here

Use `build_model_config` to select the base architecture, attention algorithm, and
attention metadata format. For example, FlexInnerAttention consumes a `BlockMask`,
while VarlenInnerAttention consumes cumulative sequence offsets.

Use a transform for options that replace or wrap configs in the built tree.
Context parallelism, TP GEMM backends, MoE communication backends,
quantization, and LoRA belong in transforms. Quantized modules, tensors, and
kernels live in `torchtitan/quantization`.

A CP transform specializes the selected attention for distributed execution.
It may change input sharding and preprocessing, but it preserves the selected
attention algorithm and metadata format.

## Dependency direction

This package may import other `torchtitan` packages. Those packages must not
import this package. Recipes import and apply transforms.

Model config builders temporarily violate this direction while they accept
and apply the legacy `ModelConfigConverter` interface. This dependency will be
removed when converters are
replaced by `ModelConfigTransform`.

Keep shared types outside this package. For example, `CPInnerAttention` lives
with the attention code. Only the transform that installs it belongs here.

## Writing a transform

Subclass `ModelConfigTransform`. Use a keyword-only dataclass for transform options.
Implement `transform`, rewrite configs in place, and return the model root. Return
a different config only when replacing the root.

```python
from dataclasses import dataclass

from torchtitan.config.transform import (
    ModelConfigTransform,
    ModelConfigTransformContext,
)
from torchtitan.protocols.module import Module


class ExternalPrerequisiteTransform(ModelConfigTransform):
    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        del context
        return model


@dataclass(kw_only=True, slots=True)
class MyTransform(ModelConfigTransform):
    setting: int

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        del context
        ...
        return model
```

A transform sees only the model config. Pass any required training or
parallelism value to the transform.

Use `convert_config_type` to replace one config implementation with another.
The replacement config must inherit from the current config type. This preserves
fields and wrappers from earlier transforms.

Use `add_precedence(before=A, after=B)` to run `A` before `B`. Use
`add_conflict(A, B)` to reject incompatible transforms. Add relations after
defining the participating transform classes.

## Central relation policy

Built-in precedence and conflict pairs live in `relations.py`. Precedence pairs
are `(before, after)`. Conflict pairs are unordered. Every `TransformRelations`
instance starts with these built-in pairs.

When the `relations` argument is omitted, only these built-in pairs are used.
To add external relations, create a graph and pass it to `apply_transforms` or
`transform_model_config_`:

```python
from torchtitan.config.transform import TransformRelations

relations = TransformRelations()
relations.add_precedence(
    before=ExternalPrerequisiteTransform,
    after=MyTransform,
)
config = apply_transforms(
    config,
    [MyTransform(setting=1), ExternalPrerequisiteTransform()],
    relations=relations,
)
```

The caller owns the graph and should finish configuring it before passing it.
An external module may export a configured graph for recipes to reuse.

The complete central ordering relation may contain cycles. Only the relation
induced by one selected transform list must be resolvable.
Ordering selects the earliest currently ready entry in the caller-provided
list. Conflict and ordering validation complete before any transform is
invoked.

Relations are subclass-aware.

When adding a transform to `torchtitan.config.transform`, put its composition
policy in `relations.py`. Downstream transforms defined outside this package
should provide their own graph; downstream users should not modify an installed
`relations.py`.

## Validation

See [Configuration validation](../README.md#validation) for where validation
belongs. `apply_transforms` validates the final trainer config after the last
transform. The trainer validates it again after command-line overrides.

`__post_init__` also runs when a config is constructed. Set related training
options before calling `apply_transforms`. It can then validate the final
config.
