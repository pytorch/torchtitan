To support rapid experimentation with torchtitan, we provide several extension points. The principle for adding these extension points is to support various use cases with flexible component swapping and reuse, while trying to keep the code clean and minimal.

The extension points and protocols mentioned in this note are subject to change.


### `TrainSpec`

[`TrainSpec`](../torchtitan/protocols/train_spec.py) supports configuring high-level components in model training, including
- definitions of model class and model args
- model parallelization functions
- loss functions
- factory methods for creating dataloader / tokenizer / optimizer / learning rate scheduler / metrics processor

The coarse level abstraction tries to hit a balance between flexible component swapping and a straightforward train script ([train.py](../torchtitan/train.py)).
Note that among all training components, currently [`CheckpointManager`](../torchtitan/components/checkpoint.py) and [`FTManager`](../torchtitan/components/ft.py) are not configurable since we do not expect them to be customized, but we are open to requests.

To register a `TrainSpec`, please use the `register_train_spec` API, and make sure registration happens before `get_train_spec` is called during training initialization. In torchtitan, `get_train_spec` will dynamically look for models in `torchtitan/models` or `torchtitan/experiments`.


### `ModelConverter`

Originated from a [request](https://github.com/pytorch/torchtitan/issues/790) to unify quantization interface and supports dynamic registration,
[`ModelConverter`](../torchtitan/protocols/model_converter.py) defines the following general interface:
- `convert` is called after model definition and meta device initialization, but before model parallelization. It can perform general module rewrite, e.g. [Float8](../torchtitan/components/float8.py) module swapping, as long as it is compatible with other components.
- `post_optimizer_hook`, as its name suggests, would be registered (via `torch.optim.Optimizer.register_step_post_hook`) to perform necessary post optimizer step operations. As an example, the [Float8](../torchtitan/components/float8.py) component in torchtitan uses this hook to issue a single all-reduce for all FSDP2 parameters (at once for better performance) to calculate the dynamic scale.

To register a `ModelConverter`, please follow the example of [Float8](../torchtitan/components/float8.py) to `register_model_converter`. Please make sure the registration code is called before training initialization. In torchtitan, it is performed during  [module import](../torchtitan/__init__.py).


### Train script

To perform various tasks, from adding a new model (possibly with a new modality), to trying out a new training paradigm (e.g. async training), a single train script cannot handle all the cases, unless customization points are inserted everywhere to make it less readable. Instead of always starting and maintaining a standalone train script, we group code in [train.py](../torchtitan/train.py) into functions to allow for reuse.

This is an ongoing effort, and the level of grouping is subject to change.


### Extending `JobConfig`

[`JobConfig`](../torchtitan/config/job_config.py) supports custom extension through the `--job.custom_config_module` flag.
This lets you define a custom module that extends `JobConfig` with additional fields.

When specified, your custom `JobConfig` is merged with the default:
- If a field exists in both, the custom config’s value replaces the default.
- Fields unique to either config are retained.

#### Example

To add a custom `custom_config` section, define your own `JobConfig`:

```python
# torchtitan/experiments/your_folder/job_config.py
from dataclasses import dataclass, field

@dataclass
class CustomConfig:
    how_is_your_day: str = "good"
    """Just an example."""

@dataclass
class Training:
    steps: int = 500
    """Replaces the default value"""

    my_mini_steps: int = 10000
    """New field is added"""

    ... # Original fields are preserved

@dataclass
class JobConfig:
    custom_config: CustomConfig = field(default_factory=CustomConfig)
    training: Training= field(default_factory=Training)
```

Then run your script with:

```bash
--job.custom_config_module=torchtitan.experiments.your_folder.job_config
```

Or specify it in your `.toml` config:

```toml
[job]
custom_config_module = "torchtitan.experiments.your_folder.job_config"
```
