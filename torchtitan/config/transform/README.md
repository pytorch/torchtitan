# Model config transforms

A model config transform rewrites a complete model config tree. `model_registry`
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

`apply_transforms` deep-copies the trainer config. It orders and applies the
transforms, then validates the result. It returns the changed copy. The input
config stays unchanged if a transform fails.

Legacy `ModelConfigConverter` instances passed to `model_registry` run before
all model config transforms. In particular, apply quantization in
`model_registry` before applying `LoRATransform`; running a converter over a
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
model_config = model_registry("0.6B", attn_backend="varlen")
model_config = transform_model_config_(model_config, [LMHeadCastTransform()])
```

## What belongs here

Use `model_registry` to select the base architecture, attention algorithm, and
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

Model registry functions temporarily violate this direction while they accept
and apply the legacy `ModelConfigConverter` interface. This dependency will be
removed when config registries move to `torchtitan_recipes` and converters are
replaced by `ModelConfigTransform`.

Keep shared types outside this package. For example, `CPInnerAttention` lives
with the attention code. Only the transform that installs it belongs here.

## Writing a transform

Subclass `ModelConfigTransform`. Use a keyword-only dataclass for transform options.
Implement `transform`, rewrite configs in place, and return the model root. Return
a different config only when replacing the root.

```python
@dataclass(kw_only=True, slots=True)
class MyTransform(ModelConfigTransform):
    run_after = (QuantizationTransform,)
    setting: int

    def transform(self, model: Module.Config) -> Module.Config:
        ...
        return model
```

A transform sees only the model config. Pass any required training or
parallelism value to the transform.

Use `convert_config_type` to replace one config implementation with another.
The replacement config must inherit from the current config type. This preserves
fields and wrappers from earlier transforms.

Use `run_after` to set the order. Use `conflicts_with` to reject incompatible
transforms. `apply_transforms` checks conflicts and sorts transforms before
running them.

## Validation

See [Configuration validation](../README.md#validation) for where validation
belongs. `apply_transforms` validates the final trainer config after the last
transform. The trainer validates it again after command-line overrides.

`__post_init__` also runs when a config is constructed. Set related training
options before calling `apply_transforms`. It can then validate the final
config.
