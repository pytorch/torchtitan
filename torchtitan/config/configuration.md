## Configuration

A run is described by a **full configuration**: a function that returns a complete `Trainer.Config` -- the model, the parallelism degrees, and every optimization choice. Configurations are written in Python, so building a new one is doing configuration programming with TorchTitan components.

Select one with `--module` (the module that defines the function) and `--config` (the function):

```bash
MODULE=torchtitan_recipes.tests CONFIG=llama3_debugmodel_fsdp2_cp2 ./run_train.sh
```

That is the whole command. To change something, change the configuration -- write your own function instead of using CLI flags:

```python
# torchtitan_recipes/my_runs.py
def llama3_debugmodel_cp4() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_cp2()
    config.parallelism.context_parallel_degree = 4
    config.training.steps = 100
    return config
```

The `--section.option` CLI flags still work and still take precedence over the configuration, but only so existing scripts keep running. They are not the way to configure a run any more, and they will go away.

### Where configurations live

The [torchtitan_recipes](../../torchtitan_recipes/) package holds full configurations. It sits next to `torchtitan` rather than inside it because the two hold different kinds of thing: `torchtitan` ships the mechanisms -- `model_registry` for architectures, and the classes implementing each optimization -- while a configuration only picks among them. A configuration is also tied to one cluster and one run, so it changes on a different schedule from the library, and shipping one is not the same promise as shipping a class.

### Writing your own

A different cluster usually means a different sharding layout, and therefore a different configuration. That needs no code change: add a function to `torchtitan_recipes`, in a module named for the model, and name it on the command line. (`torchtitan_recipes/tests.py` is separate -- it holds the configurations the integration tests run.)

```python
# torchtitan_recipes/llama3.py
def llama3_8b_fsdp8_tp2_h200() -> Trainer.Config:
    model_spec = model_registry("8B", attn_backend="flex")
    return Trainer.Config(
        model_spec=model_spec,
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

### The command-line options are frozen

The set of `--section.option` CLI flags will not grow. New features express their knobs in the config tree instead, so the way to introduce a new feature is by adding a new configuration, not a new CLI flag.

Everything already on the command line keeps working, for backward compatibility rather than because it is the recommended path. The eventual goal is to remove the flags entirely and keep only `--module` and `--config`, or even remove tyro completely.

Frozen means the CLI, not the config dataclasses. A few options genuinely have no other home, such as `training.local_batch_size`, so [configs.py](configs.py) is not closed -- discuss with the maintainers first. Annotate the field with `tyro.conf.Suppress` and a configuration can set it while the CLI stays as it is:

```python
new_job_level_knob: Annotated[int, tyro.conf.Suppress] = 3
```

`Trainer.Config.model_spec` is annotated this way, which is what keeps the whole model config tree off the CLI.

### What belongs in `torchtitan_recipes`

What this repository ships, which is deliberately a small set:

- `tests.py` -- the configurations the integration tests run
- golden configurations verified on specific hardware, named for that hardware so a benchmark run is reproducible from its name alone
- configurations that demonstrate new features

These are examples, not supported entry points. A configuration encodes one cluster -- a GPU count, an interconnect, a memory budget -- so we verify the ones we ship on the hardware in their name and nowhere else, and we may change or delete them without a deprecation. Copy one and own the copy rather than importing ours into your own code.

We do not ship every combination of model, degrees and optimization, because that set is exponential. Your run is your own configuration: add it here without committing it, or keep it in your own package and point `--module` at that. Deriving from a shipped one is a few lines, as above.

The per-model `config_registry.py` modules, selected with `--module <model> --config <function>`, are the earlier location for the same thing. They keep working and take no new entries, and they will eventually be deleted: the model-size baselines they hold, `llama3_8b` and the like, move to `torchtitan_recipes`, so the command line becomes `--module torchtitan_recipes.llama3`. There is no plan for a shim, since a re-export in every model directory would just be a second name for every configuration.
