# LLM Trainer: Optimizing Flattened PyTorch Models

## Overview

You are tasked with optimizing a flattened PyTorch training step. The training
step (forward + loss + backward) has been traced into a straight-line sequence
of `torch.ops.aten.*` operations. Your job is to rewrite these operations to
run faster while maintaining convergence — the candidate's validation loss
after 100 training steps must be **less than or equal to** the baseline's.

## Directory Layout

Models are organized under `targets/` by **fingerprint** — a string that
encodes both the hardware and the parallelism configuration. You provide
the hardware label via `HARDWARE=<name>` (e.g. `h100-sm90`), and the
tools automatically append non-trivial parallelism dimensions from the
config (e.g. `tp2`, `fsdp4`, `ep8`).

Examples:
- Single GPU: `HARDWARE=h100-sm90` → `targets/h100-sm90/`
- TP=2, FSDP=4: `HARDWARE=h100-sm90` → `targets/h100-sm90_tp2_fsdp4/`
- TP=2, EP=8: `HARDWARE=h100-sm90` → `targets/h100-sm90_tp2_ep8/`

```
torchtitan/experiments/llm_trainer/
└── targets/
    └── h100-sm90_tp2_fsdp4/          # Fingerprint = hardware + parallelism
        ├── flattened_models/         # Original traced models (DO NOT EDIT)
        │   ├── llama3_8b_rank0.py
        │   ├── llama3_8b_rank0_meta.json
        │   ├── llama3_8b_rank1.py
        │   └── ...
        ├── optimized_models/         # Current best version (auto-managed)
        │   ├── llama3_8b_rank0.py
        │   └── ...
        └── candidate_models/         # YOUR workspace — put optimized code here
            ├── llama3_8b_rank0.py
            └── ...
```

- `flattened_models/` — The raw traced output. Never modify these files.
  They serve as a reference for what the original model does.
- `optimized_models/` — The current best-performing version. Initially a
  copy of the flattened model. Updated when a candidate is promoted.
- `candidate_models/` — Your working directory. Copy an optimized model here,
  optimize it, then benchmark.

## Workflow

### Step 1: Generate the flattened model (one-time setup)

```bash
NGPU=8 HARDWARE=h100-sm90 ./torchtitan/experiments/llm_trainer/run_flattener.sh \
    --module graph_trainer.llama3 \
    --config graph_trainer_llama3_debugmodel
```

This traces the model across GPUs via torchrun. The fingerprint directory
is auto-constructed from `HARDWARE` + the parallelism config. Each rank
writes its own files:
- `targets/<fingerprint>/flattened_models/<model>_rank{i}.py`
- `targets/<fingerprint>/flattened_models/<model>_rank{i}_meta.json`
- Copies both to `targets/<fingerprint>/optimized_models/` as the initial baseline

### Step 2: Create your candidate

```bash
# Check the printed fingerprint from step 1, then copy the optimized model
cp torchtitan/experiments/llm_trainer/targets/h100-sm90_tp2_fsdp4/optimized_models/llama3_8b_rank0.py \
   torchtitan/experiments/llm_trainer/targets/h100-sm90_tp2_fsdp4/candidate_models/llama3_8b_rank0.py
```

Now edit the candidate file with your optimizations.

### Step 3: Benchmark

```bash
NGPU=8 HARDWARE=h100-sm90 ./torchtitan/experiments/llm_trainer/run_benchmarker.sh \
    --module graph_trainer.llama3 \
    --config graph_trainer_llama3_debugmodel
```

This will:
1. Load both the optimized and candidate models
2. Run both for 100 training steps (SGD updates) from identical initial state
3. Check that the candidate's final loss is **<= the baseline's**
4. Measure execution time and MFU for both
5. Print a comparison report

### Step 4: Promote (if candidate passes)

```bash
NGPU=8 HARDWARE=h100-sm90 ./torchtitan/experiments/llm_trainer/run_benchmarker.sh \
    --promote \
    --module graph_trainer.llama3 \
    --config graph_trainer_llama3_debugmodel
```

`--promote` copies the candidate to `optimized_models/` **only if** the
candidate's final validation loss is <= the baseline's AND it is at least
1% faster on every benchmark run (default: 3 consecutive runs; override
with `--promote-runs N`). A comment is inserted at the top of the promoted
file recording the MFU and timestamp, making the optimization history
self-documenting.

## Understanding the Model File

The generated `.py` file contains a single `GraphModule` class with a
`forward` method. Here's the structure:

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32768, 256]", arg1_1: "f32[256, 256]", ...):
        # Each line is one ATen operation
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype=torch.float32)
        embedding = torch.ops.aten.embedding.default(_to_copy, arg42_1)
        view = torch.ops.aten.view.default(embedding, [256, 256])
        mm = torch.ops.aten.mm.default(view, arg1_1)
        ...
        return (loss, grad_0, grad_1, ...)
```

**Inputs** (arguments to `forward`):
- The first N arguments are model state (parameters and buffers), listed
  in the docstring at the top of the file
- The remaining arguments are user data: input tokens, labels, positional
  embeddings (e.g., `freqs_cis`), etc.

**Outputs** (return tuple):
- `[0]` is the loss (scalar tensor)
- `[1..N]` are gradients for each trainable parameter

## Optimization Rules

### MUST follow:
1. **Validation loss <= baseline.** When both models are run for 100 training
   steps (SGD updates from identical initial state), the candidate's final
   loss must be less than or equal to the baseline's. Numerical approximations
   are allowed as long as they do not hurt convergence.

2. **Same function signature.** The `forward` method must accept the same
   arguments in the same order with the same types. Do not add, remove, or
   reorder arguments.

3. **Same output count and order.** Return the same number of tensors in the
   same order.

4. **Keep it a single `GraphModule` class.** The benchmarker imports
   `GraphModule` from the file.

### CAN do:
- **Fuse operations.** Replace sequences of elementwise ops with fused
  alternatives. E.g., replace separate `mul` + `add` with `addcmul`, or
  use `F.silu` instead of `sigmoid` + `mul`.

- **Use higher-level APIs.** Replace manual attention patterns with
  `F.scaled_dot_product_attention`. Replace manual RMS norm with a
  fused kernel.

- **Eliminate redundant ops.** Remove no-op views, unnecessary copies,
  identity slices, redundant type casts.

- **Reorder independent ops.** If two operations don't depend on each
  other, you can reorder them for better memory locality or to enable
  kernel fusion.

- **Use custom Triton kernels.** Write inline Triton kernels for hot
  sequences of operations. Define them at module level and call from
  `forward`.

- **Add helper functions.** Define helper functions or classes in the
  same file if it helps organize the code.

### CANNOT do:
- Introduce changes that cause the final validation loss to exceed the baseline
- Change input/output signatures
- Import external packages not already available (torch, triton are fine)
- Remove or skip gradient computations
- **Modify files outside of `candidate_models/` (and helper files you
  create alongside them).** All optimizations — including custom CUDA/Triton
  kernels, helper functions, and binding code — must live within or next to
  the candidate model files under `targets/<fingerprint>/candidate_models/`.
  Never modify core torchtitan source, `flattened_models/`, or any other
  files in the repository. The `optimized_models/` directory is managed
  exclusively by the `--promote` workflow.

## Common Options

Both shell scripts require:

```
HARDWARE=<name>    Hardware label (e.g. h100-sm90). Combined with parallelism
                   config to form the fingerprint directory name.
NGPU=<n>           Number of GPUs (default: 1)
```

### Benchmarker-specific Options

```
--promote                  Auto-promote candidate if valid AND >=1% faster
                           on all benchmark runs
--promote-runs N           Number of consecutive runs that must all pass (default: 3)
--num-model-warmup N       Model warmup calls BEFORE validation check (default: 3)
--num-validation-steps N   Training steps for validation (default: 100)
--validation-lr LR         SGD learning rate for validation steps (default: 1e-3)
--num-warmup N             Warmup iterations before timing (default: 5)
--num-bench N              Benchmark iterations per run (default: 20)
```

### How Benchmarking Works

The benchmarker runs each model through three phases:

1. **Model warmup** (`--num-model-warmup`, default 3): The model is called
   N times *before* the validation check or any timing. These calls are not
   timed and not checked for correctness. This phase exists so that models
   can initialize internal state — for example, populating caches, warming
   up JIT compilers, or recording CUDA graphs. Your candidate model can do
   arbitrary work during these calls (build lookup tables, capture graphs,
   profile and specialize) as long as it produces correct outputs from call
   N+1 onward.

2. **Validation check** (`--num-validation-steps`, default 100): Both models
   are run for N training steps from identical initial state. On each step
   the model is called, the loss is recorded, and the state parameters are
   updated via SGD (`param -= lr * grad`) using `--validation-lr` (default
   1e-3). The candidate passes if its final loss is <= the baseline's.

3. **Benchmark loop** (`--num-warmup` + `--num-bench`): Additional warmup
   iterations (not timed), then timed iterations measured with CUDA events.
   The median time across `--num-bench` iterations is reported.

Because model warmup calls are never timed or checked, you are free to use
compile-like transformations in your candidate. For example, you can cache
CPU-side scalar values during warmup and replay them in later calls, or
capture the entire forward pass as a CUDA graph during warmup and replay it
during benchmark. The only requirement is that the candidate's validation
loss does not exceed the baseline's after the configured number of steps.

## Supported Models

The flattener works with any model supported by graph_trainer. Currently:

| Model | Module flag | Config flag |
|-------|-----------|-------------|
| Llama 3 (debug) | `graph_trainer.llama3` | `graph_trainer_llama3_debugmodel` |
| Llama 3 (8B) | `graph_trainer.llama3` | `graph_trainer_llama3_8b` |
| DeepSeek V3 (debug) | `graph_trainer.deepseek_v3` | `graph_trainer_deepseek_v3_debugmodel` |
| Qwen 3 (debug) | `graph_trainer.qwen3` | `graph_trainer_qwen3_debugmodel` |
