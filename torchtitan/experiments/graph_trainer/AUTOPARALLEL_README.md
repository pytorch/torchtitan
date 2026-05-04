## GraphTrainer AutoParallel

GraphTrainer can use AutoParallel to solve model placement for `aot_fx_trace`
training. AutoParallel chooses placements for the model, while GraphTrainer
still owns the forward, loss, and backward trace through `make_fx`.

Two execution modes are supported:

- **Native GraphTrainer backend mode**: AutoParallel places the model, then
  GraphTrainer runs its normal `aot_fx_trace` pass and compile pipeline.
- **AutoParallel backend mode**: AutoParallel places the model, then
  GraphTrainer applies AutoParallel's exported backend policy helpers to the
  traced train-step graph before full Inductor compilation.

Both modes use the same AutoParallel placement path. The difference is only the
terminal pass and backend policy applied after GraphTrainer has traced the
train-step graph.

### Requirements

Install TorchTitan's normal development dependencies and make the AutoParallel
package importable in the same environment:

```bash
pip install -r requirements.txt -r requirements-dev.txt
git clone git@github.com:meta-pytorch/autoparallel.git /path/to/autoparallel
pip install -e /path/to/autoparallel
```

GraphTrainer also requires a recent PyTorch nightly. See
`torchtitan/experiments/graph_trainer/README.md` for the current nightly install
command.

### Native GraphTrainer Backend Mode

Use native mode when you want AutoParallel placement with GraphTrainer's own
`aot_fx_trace` graph passes and compile path.

```bash
NGPU=4 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.autoparallel \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```

In this mode:

- AutoParallel solves and applies model placement.
- GraphTrainer traces forward, loss, and backward with `make_fx`.
- GraphTrainer uses its normal `aot_fx_trace` cleanup, memory policy, bucketing,
  regional/full Inductor, custom codegen, and cudagraph pass choices when they
  are enabled by config.

### AutoParallel Backend Mode

Use AutoParallel backend mode when you want the placed model to use
AutoParallel's backend policy for terminal full-Inductor compilation.

```bash
NGPU=4 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.autoparallel \
    --compile.inductor_compilation autoparallel_backend \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```

In this mode:

- AutoParallel solves and applies model placement.
- GraphTrainer traces forward, loss, and backward with `make_fx`.
- GraphTrainer runs only the shared trace cleanup passes before the backend
  compile pass.
- GraphTrainer then runs
  `autoparallel_backend_full_inductor_compilation_pass`, which imports
  AutoParallel's `get_autoparallel_backend_policy_helpers(...)`, applies the
  AutoParallel AOTAutograd joint policy, and compiles the traced train-step
  graph through GraphTrainer's full-Inductor path.

The shared cleanup passes are `remove_detach_pass`, `remove_identity_view_pass`,
`remove_identity_slice_pass`, and `normalize_view_ops_as_reshape`.

AutoParallel backend mode does not run GraphTrainer's activation or memory
policy passes, CPU offload pass, bucketing passes, regional Inductor path,
custom codegen, or cudagraph pass. Those policy choices remain part of native
GraphTrainer `aot_fx_trace` mode.

### Supported Models

#### Llama3

Llama3 AutoParallel uses the dense mesh and supports FSDP plus TP placement.
Loss parallel requires TP.

Native backend:

```bash
NGPU=4 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.autoparallel \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```

AutoParallel backend:

```bash
NGPU=4 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.autoparallel \
    --compile.inductor_compilation autoparallel_backend \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```

#### DeepSeek V3

DeepSeek V3 AutoParallel uses the sparse EFSDP plus EP mesh. TP loss parallel is
not supported in this path, so disable loss parallel.

Native backend:

```bash
NGPU=4 MODULE=graph_trainer.deepseek_v3 \
    CONFIG=graph_trainer_deepseek_v3_debugmodel_ep \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.autoparallel \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.expert_parallel_degree 2 \
    --parallelism.disable_loss_parallel
```

AutoParallel backend:

```bash
NGPU=4 MODULE=graph_trainer.deepseek_v3 \
    CONFIG=graph_trainer_deepseek_v3_debugmodel_ep \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.autoparallel \
    --compile.inductor_compilation autoparallel_backend \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.expert_parallel_degree 2 \
    --parallelism.disable_loss_parallel
```

### Configuration Rules

`--compile.inductor_compilation autoparallel_backend` is valid only with:

```bash
--compile.mode aot_fx_trace
--compile.autoparallel
```

AutoParallel backend mode rejects these options:

- `--compile.enable_passes false`
- explicit `--compile.passes`
- explicit `--compile.joint_passes`
- `--compile.precompile_artifact_dir`

The backend selector owns the terminal compile policy in this mode. Use native
GraphTrainer backend mode for GraphTrainer regional Inductor, custom pass lists,
precompile, cudagraph, bucketing, CPU offload, or activation memory policy work.

### FSDP Resharding

AutoParallel placement follows GraphTrainer's resolved
`fsdp_reshard_after_forward` policy:

- without pipeline parallelism, the default is to reshard after forward;
- with pipeline parallelism, the default is not to reshard after forward.

The resolved boolean is passed into `AutoParallelGraph` for both native and
AutoParallel backend modes.

### Tests

Run the focused unit tests with:

```bash
pytest torchtitan/experiments/graph_trainer/tests/test_autoparallel_backend.py -q
```

The distributed integration coverage has paired native and AutoParallel backend
tests for each model:

- `autoparallel_llama3_fsdp_tp`
- `autoparallel_backend_llama3_fsdp_tp`
- `autoparallel_deepseek_v3_efsdp_ep`
- `autoparallel_backend_deepseek_v3_efsdp_ep`

In CI, the GraphTrainer 8 GPU integration workflow installs AutoParallel from
GitHub before running this suite:

```bash
python -m pip install git+https://github.com/meta-pytorch/autoparallel.git
python -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    --test_suite graph_trainer_autoparallel \
    --gpu_arch_type cuda \
    "$RUNNER_TEMP/artifacts-to-be-uploaded/autoparallel" \
    --ngpu 4
```

When changing placement, tracing, graph passes, or backend policy behavior,
compare numerics between native GraphTrainer backend mode and AutoParallel
backend mode with `--debug.seed=42` and `--debug.deterministic`.

### Current Limitations

- AutoParallel backend mode is a full-Inductor terminal compile strategy and is
  not composable with GraphTrainer regional Inductor or custom graph pass
  selections.
- Precompile is not supported for AutoParallel backend mode.
- DeepSeek V3 currently uses an AutoParallel-compatible DeepSeek model from the
  AutoParallel package, and coverage is limited to EFSDP plus EP with loss
  parallel disabled.
- The GraphTrainer adapter depends on AutoParallel and PyTorch internals that
  may change; keep integration tests updated when upgrading either dependency.
