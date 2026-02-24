To support rapid experimentation with torchtitan, we provide several extension points. The principle for adding these extension points is to support various use cases with flexible component swapping and reuse, while trying to keep the code clean and minimal.

The extension points and protocols mentioned in this note are subject to change.


### `ModelSpec`

[`ModelSpec`](../torchtitan/protocols/model_spec.py) supports configuring high-level components in model training, including
- definitions of model config and model class
- model parallelization functions
- loss functions

The coarse level abstraction tries to hit a balance between flexible component swapping and a straightforward train script ([train.py](../torchtitan/train.py)).

To register a model, define a `model_registry(flavor)` function in your model's `__init__.py` that returns a `ModelSpec`. Then define training configs in a `config_registry.py` module. See [torchtitan/models/llama3](../torchtitan/models/llama3/) for an example.


### `ModelConverter`

[`ModelConverter`](../torchtitan/protocols/model_converter.py) defines the following general interface:
- `convert` is called after model definition and meta device initialization, but before model parallelization. It can perform general module rewrite, e.g. [Float8](../torchtitan/components/quantization/float8.py) module swapping, as long as it is compatible with other components.
- `post_optimizer_hook`, as its name suggests, would be registered (via `torch.optim.Optimizer.register_step_post_hook`) to perform necessary post optimizer step operations. As an example, the [Float8](../torchtitan/components/quantization/float8.py) component in torchtitan uses this hook to issue a single all-reduce for all FSDP2 parameters (at once for better performance) to calculate the dynamic scale.

To add a `ModelConverter`, create a class inheriting from `Configurable` with a nested `Config(Configurable.Config)` dataclass. Add the Config object to `model_converters` in your config_registry function. See [Float8LinearConverter](../torchtitan/components/quantization/float8.py) for an example.


### Train script

To perform various tasks, from adding a new model (possibly with a new modality), to trying out a new training paradigm (e.g. async training), a single train script cannot handle all the cases, unless customization points are inserted everywhere to make it less readable. Instead of always starting and maintaining a standalone train script, we group code in [train.py](../torchtitan/train.py) into functions to allow for reuse.

This is an ongoing effort, and the level of grouping is subject to change.


### Extending `Trainer.Config`

To add custom configuration for an experiment, subclass `Trainer.Config` (or `Trainer` itself) and add new fields. Define config_registry functions that return your custom Config type.

#### Example

To add a custom config section for an experiment:

```python
# torchtitan/experiments/your_folder/trainer.py
from dataclasses import dataclass, field
from torchtitan.trainer import Trainer

@dataclass
class CustomConfig:
    how_is_your_day: str = "good"
    """Just an example."""

class MyTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        custom_config: CustomConfig = field(default_factory=CustomConfig)
```

Then in your `config_registry.py`:

```python
# torchtitan/experiments/your_folder/config_registry.py
from .trainer import MyTrainer, CustomConfig

def my_experiment_debugmodel() -> MyTrainer.Config:
    return MyTrainer.Config(
        custom_config=CustomConfig(how_is_your_day="great"),
        training=TrainingConfig(steps=100),
        # ... other fields
    )
```

Then run with:

```bash
MODULE=your_folder CONFIG=my_experiment_debugmodel ./run_train.sh
```
