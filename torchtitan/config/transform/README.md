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

`apply_transforms` deep-copies the trainer config. It orders and applies the
transforms, then validates the result. It returns the changed copy. The input
config stays unchanged if a transform fails.

Legacy `ModelConfigConverter` instances passed to `model_registry` run before
all model config transforms. In particular, apply quantization in
`model_registry` before applying `LoRATransform`; running a converter over a
LoRA-transformed tree can replace an adapter config.

Use `transform_model_config_` when there is no trainer config. It rewrites the
model config in place and returns the root. It does not copy or validate the
config.

```python
model_config = model_registry("0.6B", attn_backend="varlen")
model_config = transform_model_config_(model_config, [LMHeadCastTransform()])
```

## MX quantization-aware training

`MXQATTransform` keeps master parameters and optimizer state in the training
precision. It specializes the grouped-MM hook and, when selected, ordinary
`Linear` projections. Grouped experts fake-quantize weights and activations;
dense projections fake-quantize weights only.

```python
config = apply_transforms(config, [MXQATTransform()])
```

By default, every grouped-expert config is selected and dense projections are
unchanged. Use exact config FQNs in `grouped_expert_fqns` and `linear_fqns` for
explicit selection. An empty tuple selects none. `from_weight_fqns` translates
adapter-resolved parameter FQNs and rejects unsupported or partially selected
grouped modules. This translation requires parameter and config paths to agree;
models with renamed or repeated configs need adapter-specific translation.

Pass TorchAO `MXFakeQuantizeConfig` instances through `weight_fake_quant_config`
and `activation_fake_quant_config`. Their existing `kernel_preference` controls
both quantization and grouped execution: `EMULATED` uses dequantized operands,
while `AUTO` uses native MXFP8 grouped forward kernels with a high-precision STE
backward. Both preferences must agree. Native execution requires supported CUDA
hardware and TorchAO kernel dependencies.

The transform preserves inherited config fields, rejects existing incompatible
execution overrides, and runs before `LoRATransform`. Packed checkpoint import
resolves its own quantization metadata and checks that the QAT selection agrees.

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
