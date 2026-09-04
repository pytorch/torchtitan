> Copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running; pending rebase and reconcile.

# Model transforms

A model transform rewrites a complete model config tree. `model_registry`
builds the base model before transforms run.

Transforms are optional. Users may build configs with their own utilities.

## Using transforms

Set all training options first. Then call `apply_transforms` once.

```python
config = muse_glimmer_30b()
config.parallelism.context_parallel_degree = 8

config = apply_transforms(
    config,
    [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
)
```

`apply_transforms` deep-copies the trainer config. It orders and applies the
transforms, then validates the result. It returns the changed copy. The input
config stays unchanged if a transform fails.

Use `transform_model` when there is no trainer config, such as with a bare
`ModelSpec`. It rewrites the model config in place and returns the root. It does
not copy or validate the config.

```python
spec = model_registry("0.6B", attn_backend="varlen")
spec.model = transform_model(spec.model, [LMHeadCastTransform.Config()])
```

## What belongs here

Use `model_registry` for options that define the base architecture or input
contract. These include model dimensions and the attention backend.

Use a transform for options that replace or wrap nodes in the built tree.
Context parallelism, TP GEMM backends, MoE communication backends,
quantization, and LoRA belong in transforms.

Needing training settings for validation does not make an option a transform.
For example, the attention backend still belongs in `model_registry`.

## Dependency direction

This package may import other `torchtitan` packages. Those packages must not
import this package. Recipes import and apply transforms.

Keep shared types outside this package. For example, `ContextParallelKernel`
lives with the attention code. Only the transform that installs the kernel
belongs here.

## Writing a transform

Subclass `ModelTransform`. Define its config and implement `transform`. Rewrite
nodes in place and return the model root. Return a different config only when
replacing the root.

```python
class MyTransform(ModelTransform):
    run_after = (QuantizationTransform,)

    @dataclass(kw_only=True, slots=True)
    class Config(ModelTransform.Config):
        setting: int

    def transform(self, model: Module.Config) -> Module.Config:
        ...
        return model
```

A transform sees only the model config. Pass any required training or
parallelism value through the transform config.

Use `retype_node` to change a node implementation. The replacement config must
inherit from the current config type. This preserves fields and wrappers from
earlier transforms.

Use `run_after` to set the order. Use `conflicts_with` to reject incompatible
transforms. `apply_transforms` checks conflicts and sorts transforms before
running them.

## Validation

Keep full trainer config validation in `__post_init__`. `apply_transforms` runs
it after the last transform. The trainer runs it again after command-line
overrides.

Place each check based on the data it needs.

- Keep a feature's checks in its package.
- Call a check from the first config that has every required value.
- Put checks that need both model and parallelism configs in
  `Trainer.Config.__post_init__`.

`__post_init__` also runs when a config is constructed. Set related training
options before calling `apply_transforms`. It can then validate the final
config.
