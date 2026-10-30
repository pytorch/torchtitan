# Model Port Test Coverage

Add three types of tests for every model port. Record what each test covers, its
recipe, expected result, hardware, and whether it has run. Keep the model's
important features in the smaller debug model, including any MoE routing,
balancing, hybrid attention, auxiliary objectives, or multimodal inputs.

## 1. Important feature tests, mostly CPU

List new math, routing, state updates, and checkpoint behavior. Reuse tests
that exercise those paths and add focused tests where coverage is missing. Put
tests for shared components in `tests/unit_tests/cpu/models/common/` and tests
for individual models in their CPU test packages; extend shared config/adapter
tests as needed.

Use small inputs and expected results calculated by hand or by an independent
reference algorithm. Choose cases that expose likely errors, and check gradients
for new operations that require them. Include edge cases.

Test stateful features through the trainer and optimizer. Check when statistics
accumulate, when updates and resets happen, evaluation behavior, checkpoint
loading, and resetting temporary state. Recomputing activations must not count
statistics twice.

Examples:

- Quantile balancing: quantile math and state updates.
- New attention math: output and gradient checks against a simple reference.

Keep feature tests runnable on CPU. Also check outputs and gradients of custom
CUDA kernels on GPU. Check that state shared across ranks stays consistent in
the combined training test.

Run the relevant CPU tests and confirm CI runs them. The CPU workflow uses
`pytest tests/unit_tests/cpu`.

## 2. Combined parallelism training on real GPUs

Add one training test that enables FSDP, TP, EP, and PP together for
an MoE model, normally on a remote node with 8 H100s. For a dense model, cover
FSDP/TP/PP and record EP as not applicable. Use this run to catch interactions
between parallelisms. Add separate tests for features it cannot cover.

Build the parallelism config first and pass it to the model's flavor builder.
This is a valid starting configuration for 8 GPUs:

```python
from torchtitan.config.parallelism import ParallelismConfig

parallelism = ParallelismConfig(
    data_parallel_replicate_degree=1,
    data_parallel_shard_degree=2,
    tensor_parallel_degree=2,
    expert_parallel_degree=2,
    pipeline_parallel_degree=2,
    context_parallel_degree=1,
    pipeline_parallel_schedule="1F1B",
    num_pp_microbatches=4,
)
```

The GPU count is `dp_replicate * dp_shard * cp * tp * pp = 8`.
EP shares the ranks used by `dp_shard`, `cp`, and `tp`:
`edp_shard * ep = dp_shard * cp * tp`. Here expert FSDP has `edp_shard = 2`.
Setting EP=4 gives `edp_shard = 1`. Choose degrees that fit the expert count,
attention heads, and model dimensions, then check the resulting sharding.
See Qwen3.5 in `tests/integration_tests/models.py` for an example with
FSDP=2, TP=2, EP=4, PP=2.

Use a supported pipeline schedule, enough layers for every stage, and enough
tokens for the microbatches. Train for at least 10 optimizer steps with the new
features enabled. Check finite loss and grad_norm, successful backward and
optimizer steps, and feature updates across steps. Check expected behavior,
such as matching router state across ranks.

Register an importable recipe factory in an `IntegrationTestDefinition` in
`tests/integration_tests/models.py` or `tests/integration_tests/h100.py`.
Set `ngpu=8` and `use_real_pg=True`. Check the actual model, parallelism, sharding,
features, and optimizer. Confirm the test runs with real communication on the
intended GPUs. PP and checkpoint tests need `use_real_pg=True`.

Choose PP splits that test state passed between layers. Kimi K3's attention
residual cache needs multiple virtual stages, exact gradient routing, and cleanup
between microbatches, between steps, and after evaluation. Check input preparation
too: Qwen3.5 uses different positions for attention masks and MRoPE, and keeps
vision gradients connected when images or videos are absent. Packed compression
and recurrence must match processing each document separately, including short
documents and incomplete groups. Add CP or modality tests when the combined
recipe cannot cover them.

Record which optimizers are tested. Kimi K3's combined TP test uses AdamW, and
a separate test covers DistMuon. Add another recipe if the combined test cannot
cover a required feature or optimizer.

The H100 workflow calls `.github/scripts/run_8xgpu_integration_tests.sh`, which
runs the `h100` suite. Register the test in that suite and check the filters,
GPU count, and disabled settings so CI runs it.

## 3. Numerical golden baseline from a remote debug run

### Prove the debug recipe is deterministic

After reference parity and feature tests pass, choose a small debug recipe that
keeps the important computation and state updates. Enable deterministic execution
and prove that independent runs produce identical metrics before creating a golden.

Pin the recipe. Record the source commit, GPU model and count, parallelism,
PyTorch, CUDA and dependency versions, dtype, attention backend, compile/remat
settings, optimizer and scheduler, tokenizer and data revisions, input order,
and initialization. Set `config.debug.seed = 42` and
`config.debug.deterministic = True`; never enable `deterministic_warn_only`.
Use the same hardware, settings, and data order for replay.

Run at least two independent training jobs on the target remote GPUs, with the
same code and config and at least 10 optimizer steps each. Start with fresh
optimizer, scheduler, router/balancer, and dataloader state. Use reproducible
initialization or the same pinned checkpoint. Give each run its own output
directory. When using generated seed checkpoints, independently recreate the
checkpoint for each run to check initialization too.

Save temporary loss and grad_norm traces in the run outputs. Both runs must have
the same expected step numbers, finite values, and bitwise-identical loss and
grad_norm at every step. Compare full-precision TensorBoard values; stdout rounds
them. Record the runs and comparison result. Repeat if differences are intermittent.

`loss_compare.py` needs import/export for a same-config `. .` comparison. First
capture one run's metrics, then compare a new run against them. Define importable
debug and matching seed recipe factories before using this template. Adjust the
module, hardware, and GPU count:

```bash
python3 scripts/loss_compare.py . . \
  --baseline-module=tests.integration_tests.h100 \
  --baseline-config=new_model_debugmodel_all_parallelism \
  --seed-config=new_model_debugmodel_seed \
  --baseline-ngpus=8 --test-ngpus=8 --steps=10 --metrics=loss,grad_norm \
  --job-dump-folder=outputs/new-model-repeatability/run-a \
  --output-folder=outputs/new-model-repeatability/run-a-logs \
  --export-result=outputs/new-model-repeatability/run-a-metrics.txt

python3 scripts/loss_compare.py . . \
  --baseline-module=tests.integration_tests.h100 \
  --baseline-config=new_model_debugmodel_all_parallelism \
  --seed-config=new_model_debugmodel_seed \
  --baseline-ngpus=8 --test-ngpus=8 --steps=10 --metrics=loss,grad_norm \
  --job-dump-folder=outputs/new-model-repeatability/run-b \
  --output-folder=outputs/new-model-repeatability/run-b-logs \
  --import-result=outputs/new-model-repeatability/run-a-metrics.txt \
  --assert-equal
```

These commands independently create seed checkpoints and capture temporary
metrics from the same recipe. Keep the required features, optimizer, and backend
active. If the runs differ, investigate data order, initialization, state updates,
kernel reductions/atomics, routing ties, and compile/remat behavior. Use
`/numerics_debugging` when available. Fix the cause and repeat the check. Keep
exact equality checks, required state updates, and the model's own routing;
never enable `deterministic_warn_only`.

Use the combined training recipe for the golden when it passes this check. If
its parallelism or backend cannot be made repeatable, record the limitation and
use a separate numerical recipe that keeps the important features. Prove that
recipe repeats exactly before creating its golden.

### Create the golden after the runs match

Use the integration runner and `scripts/loss_compare.py` to export full-precision
TensorBoard loss and grad_norm to a small text file. The runner adds the config,
GPU count, and parallelism header. Record other run details in comments or
documentation beside the golden. Keep large logs, checkpoints, and TensorBoard
files in persistent remote storage.

Define both recipe factories before using this registration template:

```python
from tests.integration_tests import IntegrationTestDefinition

IntegrationTestDefinition(
    configs=[new_model_debugmodel_all_parallelism],
    test_name="new_model_fsdp+tp+ep+pp",
    test_descr="New model combined parallelism and numerical regression",
    ngpu=8,
    use_real_pg=True,
    golden_numerics_path=(
        "tests/assets/losses/real_pg/{gpu_arch}/new_model_all_parallelism.txt"
    ),
    loss_compare_seed_config=new_model_debugmodel_seed,
)
```

A test with a golden must have exactly one config. The `real_pg` runner creates
a single-GPU seed checkpoint by default. The seed recipe must build the same
model and weights with parallelism disabled and no transforms that need multiple
GPUs. Preserve feature math and initialization. Set `loss_compare_seed_config`
when the training recipe cannot create this seed. Keep stateful features active.

After the repeatability check passes and the test is registered in the `h100`
suite, export on the remote GPU node using its supported launcher and the
validated recipe. Replace the template names and paths. Use an empty output
directory for each run:

```bash
python3 -m tests.integration_tests.run_tests outputs/new-model-export \
  --test_suite h100 --test_name new_model_fsdp+tp+ep+pp \
  --execution_mode real_pg --gpu_arch_type cuda --gpu_arch h100 \
  --ngpu 8 --no-parallel --export-numerics
```

Set the recipe's `training.steps` to at least 10 before export. The runner writes
the result to `outputs/new-model-export/real_pg/h100/` using the golden's filename.
Check every expected step, both metrics, run settings, and enabled features. Copy
the checked file into the registered `tests/assets/losses/real_pg/h100/` path.
Keep separate goldens for each GPU architecture and communication mode.

Replay against the committed golden using another empty output directory:

```bash
python3 -m tests.integration_tests.run_tests outputs/new-model-replay \
  --test_suite h100 --test_name new_model_fsdp+tp+ep+pp \
  --execution_mode real_pg --gpu_arch_type cuda --gpu_arch h100 \
  --ngpu 8 --no-parallel
```

Set `golden_numerics_path` and confirm the replay log shows that the golden was
imported and compared. Require matching step numbers and exact full-precision
loss and grad_norm at every step. Check the `# step loss grad_norm` header so
both metrics are covered. CI must compare against the golden without export.

Investigate differences before updating the golden. For an intentional computation
or dependency change, include new numerical checks, convergence evidence where
required, old and new values at each step, and the reason for the update. Keep
the old golden while debugging.

## What to record

- Tests for each feature, including new math, gradients, and state updates.
- Combined training recipe, actual mesh, hardware, steps, and CI selection.
- Independent remote debug runs, with exact loss and grad_norm comparisons at
  every expected step before golden generation.
- Golden file, run details, and a successful independent replay that compares
  full-precision loss and grad_norm.
- Checks not yet run and unsupported combinations.

If remote GPUs are unavailable, prepare the tests and recipes and list the
pending GPU runs. Mark golden validation complete only after an independent
replay passes.
