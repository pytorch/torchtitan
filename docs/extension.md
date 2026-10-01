To support rapid experimentation with torchtitan, we provide several extension points. The principle for adding these extension points is to support various use cases with flexible component swapping and reuse, while trying to keep the code clean and minimal.

The extension points and protocols mentioned in this note are subject to change.


### Models

[`BaseModel`](../torchtitan/protocols/model.py) defines the model-level training
lifecycle. A concrete model owns its nested configuration, parallelization and
pipeline behavior, checkpoint adapter, and optional optimizer hooks.

To support a model, define `MODEL_FLAVORS` and a
`build_model_config(flavor)` function in the model package's `flavors.py`.
The function returns the selected `BaseModel.Config`. Define complete training
recipes separately under `torchtitan_recipes`, or in an external Python module.
See [torchtitan/models/llama3](../torchtitan/models/llama3/) and
[torchtitan_recipes/models/llama3.py](../torchtitan_recipes/models/llama3.py).


### Train script

To perform various tasks, from adding a new model (possibly with a new modality), to trying out a new training paradigm (e.g. async training), a single train script cannot handle all the cases, unless customization points are inserted everywhere to make it less readable. Instead of always starting and maintaining a standalone train script, we group code in [train.py](../torchtitan/train.py) into functions to allow for reuse.

This is an ongoing effort, and the level of grouping is subject to change.


### Extending `Trainer.Config`

To add custom configuration for an experiment, subclass `Trainer.Config` (or `Trainer` itself) and add new fields. Define recipe functions that return your custom Config type.

Fields added this way are available to experiment config recipes. They do not
become command-line options; select a complete recipe with `--module` and
`--config`. See [the configuration doc](../torchtitan/config/README.md).

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

Then in your recipe module:

```python
# my_project/recipes.py
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
MODULE=my_project.recipes CONFIG=my_experiment_debugmodel ./run_train.sh
```
