## Configuration

A run is described by a **full configuration**: a function that returns a complete `Trainer.Config` -- the model, the parallelism degrees, and every optimization choice. Configurations are written in Python, so building a new one is doing configuration programming with TorchTitan components.

Select one with `--module` (the module that defines the function) and `--config` (the function):

```bash
NGPU=4 MODULE=torchtitan_recipes.tests.features CONFIG=llama3_debugmodel_fsdp2_cp2 ./run_train.sh
```

The parallelism degrees are in the configuration but the world size is not. So `NGPU` still has to match the product of them. If you want to change any behavior, change the configuration directly -- write your own function, instead of using CLI flags, to return a new `Trainer.Config`:

```python
# torchtitan_recipes/my_runs.py
def llama3_debugmodel_fsdp2_cp4() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_cp2()
    config.parallelism.context_parallel_degree = 4
    config.training.steps = 100
    return config
```

The command line deliberately does not expose general `--section.option`
configuration. Put training behavior in the recipe so a run can be reproduced
from its module and function name.

The supported operational options are:

- `--override TARGET[=JSON]`, repeatable, for registered component overrides
- `--comm-backend BACKEND`, for launch-environment communication selection
- `--output-dir PATH`, for run-specific output placement
- `--resume-step STEP`, for operational checkpoint resumption
- `--print-config`, to print the resolved config and exit

### Where configurations live

The [torchtitan_recipes](../../torchtitan_recipes/) package holds full configurations -- a recipe is one of these functions. It sits next to `torchtitan` rather than inside it because the two hold different kinds of thing: `torchtitan` ships the model definitions and the classes implementing each optimization; `torchtitan_recipes` only picks combinations of those components. A configuration is also tied to one cluster and one run, so it changes on a different schedule from the library, and shipping one is not the same promise as shipping a class.

`torchtitan_recipes` is a second top-level package, so an editable install made before it existed does not know about it and `import torchtitan_recipes` fails outside the torchtitan repository root. Re-run `pip install -e .` once if you run outside the repository root. Running from the repository root, as `run_train.sh` and CI do, works either way.

### Writing your own

A different cluster usually means a different sharding layout, and therefore a different configuration. That needs no code change: add a function to `torchtitan_recipes`, in a module named for the model, and name it on the command line. (`torchtitan_recipes/tests/` is separate -- it holds the configurations the integration tests run, one module per suite.)

```python
# torchtitan_recipes/llama3.py
def llama3_8b_fsdp8_tp2_h200() -> Trainer.Config:
    model_config = model_registry("8B", attn_backend="flex")
    return Trainer.Config(
        model=model_config,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            tensor_parallel_degree=2,
        ),
        ...
    )
```

`--module` takes any importable module, so a configuration kept outside this repository works the same way:

```bash
MODULE=my_company_configs.experiments CONFIG=llama3_ablation_7 ./run_train.sh
```

### Configuring optimizers

Optimizer configs are typed and intentionally expose only the options used by
verified TorchTitan recipes. For example, `Adam.Config` and `AdamW.Config`
both expose `weight_decay`. They do not forward an unrestricted dictionary of
keyword arguments to PyTorch optimizers. Add a typed field to the corresponding
optimizer config when a further upstream option is needed; this keeps recipes
validated and makes their supported behavior explicit.

### Transforming model configs

[Model config transforms](transform/README.md) are one supported way to modify
a completed model config. They are optional, and users may use their own tools.

### Validation

Keep simple invariants that use one config's fields in its `__post_init__`.
Keep checks that span config sections in [validation.py](validation.py).
`Trainer.Config.__post_init__` calls these checks after its local validation.

### Adding configuration fields

New fields belong in the config tree: on the component they configure, on the
model, or -- for the few options with no other home, such as
`training.num_tokens_per_microbatch_per_dp_rank` -- in [configs.py](configs.py),
after discussing with the maintainers. Adding a dataclass field does not change
the CLI surface.

### What belongs in `torchtitan_recipes`

What this repository ships, which is deliberately a small set:

- `tests/` -- the configurations the integration tests run, one module per suite
- golden configurations verified on specific hardware, named for that hardware so a benchmark run is reproducible from its name alone
- configurations that demonstrate new features

We do not ship every combination of model, degrees and optimization, because that set is exponential. Your run is your own configuration: add it here without committing it, or keep it in your own package and point `--module` at that. Deriving from a shipped one is a few lines, as above.

The per-model `config_registry.py` modules, selected with `--module <model> --config <function>`, are the earlier location for the same thing. They keep working and they will eventually be deleted: the model-size baselines they hold, `llama3_8b` and the like, move to `torchtitan_recipes`, so the command line becomes `--module torchtitan_recipes.llama3`. There is no plan for a shim, since a re-export in every model directory would just be a second name for every configuration.
