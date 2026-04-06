# GraphTrainer Development Guide

This supplements the root `.claude/CLAUDE.md` which covers core torchtitan
conventions (code style, naming, testing, PR expectations, etc.). Rules there
apply here too unless overridden below.

## Graph Pass Signature

All graph passes must follow the signature:
```python
def my_pass(gm: torch.fx.GraphModule, example_inputs, *, other_kwargs) -> torch.fx.GraphModule:
```
The first two positional args are always `(gm, example_inputs)`. Any additional
parameters must be keyword-only. The pass must return the (possibly transformed)
`GraphModule`.

## Don't Modify Core for This Experiment

Do not add `if graph_trainer:` branches to `torchtitan/train.py`
or other core files. GraphTrainer extends `Trainer` and overrides behavior through
subclassing.


### Local development (debug models, 8 GPUs)

```bash
# Llama3 with FSDP + TP
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2

# DeepSeek-v3 with FSDP + TP + EP (requires H100)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --parallelism.expert_parallel_degree=4 \
    --parallelism.expert_tensor_parallel_degree=1
```

### Tests

```bash
# Unit tests (GPU)
pytest torchtitan/experiments/graph_trainer/tests/test_passes.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_precompile.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_trace_module.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_numerics.py -x

# Integration tests (8 GPUs)
python torchtitan/experiments/graph_trainer/tests/integration_tests.py <output_dir> \
    --test_suite graph_trainer_default --ngpu 8
```
