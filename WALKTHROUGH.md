# TorchTitan Codebase Walkthrough

A guided reading order for understanding this distributed LLM training framework.
Read files in the order presented — each section builds on the previous one.

---

## Reading Order Overview

```
1. Entry Point        run_train.sh → train.py
2. Configuration      config/job_config.py → config/manager.py
3. Model Protocol     protocols/model.py → protocols/train_spec.py
4. A Concrete Model   models/llama3/__init__.py → args.py → model_def.py
5. Data Pipeline      hf_datasets/text_datasets.py → components/dataloader.py → components/tokenizer.py
6. Loss & Optimization components/loss.py → components/optimizer.py → components/lr_scheduler.py
7. Training Loop      train.py (re-read with full context)
8. Metrics & Logging  components/metrics.py → tools/logging.py
9. Checkpointing      components/checkpoint.py → protocols/state_dict_adapter.py
10. Validation        components/validate.py
11. Parallelism       distributed/parallel_dims.py → distributed/utils.py
12. MoE               models/moe/moe.py → models/deepseek_v3/__init__.py
```

---

## 1. Entry Point

### `run_train.sh`
The shell script that launches training. Read this first to understand how the system starts.

```
User runs ./run_train.sh
  → sets NGPU, CONFIG_FILE, LOG_RANK
  → if COMM_MODE set: runs single-process debug mode (fake_backend or local_tensor)
  → else: runs torchrun with NCCL for real distributed training
  → invokes: python -m torchtitan.train
```

**Key env vars:**
- `NGPU` — number of GPUs (default 8)
- `CONFIG_FILE` — path to TOML config file
- `COMM_MODE` — `fake_backend` for dry-run, `local_tensor` for single-GPU simulation

### `torchtitan/train.py` (first pass — just read `main()` at the bottom)
Skip the `Trainer` class for now. Start at the bottom:

```python
def main(trainer_class):
    config_manager = ConfigManager()
    config = config_manager.parse_args()   # parse TOML + CLI
    trainer = trainer_class(config)         # build everything
    trainer.train()                        # run training loop
```

That's the entire flow. Everything else is details.

---

## 2. Configuration

### `torchtitan/config/job_config.py`
All configuration lives here as nested dataclasses. Skim the field names and docstrings — don't memorize values.

**Key sections and what they control:**

| Dataclass | Controls | Key Fields |
|-----------|----------|------------|
| `Job` | Output paths | `dump_folder`, `config_file` |
| `Model` | Which model to train | `name`, `flavor`, `hf_assets_path` |
| `Training` | Training hyperparams | `local_batch_size`, `seq_len`, `steps`, `max_norm`, `dataset` |
| `Optimizer` | Adam/AdamW config | `name`, `lr`, `beta1`, `beta2`, `weight_decay` |
| `LRScheduler` | LR schedule | `warmup_steps`, `decay_ratio`, `decay_type` |
| `Parallelism` | How to distribute | `data_parallel_shard_degree`, `tensor_parallel_degree`, `pipeline_parallel_degree` |
| `Checkpoint` | Save/load behavior | `enable`, `folder`, `interval`, `async_mode` |
| `Metrics` | Logging | `log_freq`, `enable_wandb` |
| `Validation` | Periodic eval | `enable`, `freq`, `steps` |
| `ActivationCheckpoint` | Memory optimization | `mode` (selective/full/none) |
| `Compile` | torch.compile | `enable`, `components` |
| `Profiling` | Performance tracing | `enable_profiling`, `enable_memory_snapshot` |
| `Debug` | Reproducibility | `seed`, `deterministic` |

**The container:**
```python
@dataclass
class JobConfig:
    job: Job
    model: Model
    training: Training
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    parallelism: Parallelism
    checkpoint: Checkpoint
    ...
```

### `torchtitan/config/manager.py`
Parses configuration from multiple sources with clear precedence:

```
CLI args  >  TOML file  >  dataclass defaults
```

**How it works:**
1. `_maybe_load_toml()` — finds `--job.config_file` in CLI args, loads the TOML
2. `_dict_to_dataclass()` — converts TOML dict into JobConfig dataclass (recursive)
3. `_apply_cli_overrides()` — parses `--section.key value` CLI args, coerces types, applies on top
4. `_validate_config()` — checks hf_assets_path exists

**CLI override format:** `--training.steps=100` or `--training.steps 100`

---

## 3. Model Protocol

### `torchtitan/protocols/model.py`
Defines the interface every model must implement:

```python
class BaseModelArgs:
    def update_from_config(self, job_config): ...   # pull training config into model args
    def get_nparams_and_flops(self, model, seq_len): ...  # for MFU calculation

class ModelProtocol:
    def __init__(self, model_args): ...
    def init_weights(self, buffer_device=None): ...      # called after to_empty()
    def get_attention_masks(self, ...): ...               # optional, for flex/varlen attn
```

### `torchtitan/protocols/train_spec.py`
The registration system. A `TrainSpec` is a bag of function pointers that tells the Trainer how to build everything for a given model:

```python
@dataclass
class TrainSpec:
    model_cls               # the nn.Module class
    model_args              # dict of flavor_name → BaseModelArgs
    parallelize_fn          # applies TP/FSDP/compile to model
    pipelining_fn           # optional: sets up pipeline parallel
    build_optimizers_fn     # creates optimizer(s)
    build_lr_schedulers_fn  # creates LR scheduler(s)
    build_dataloader_fn     # creates dataloader
    build_tokenizer_fn      # creates tokenizer (optional)
    build_loss_fn           # creates loss function
    build_validator_fn      # creates validator (optional)
    state_dict_adapter      # HF checkpoint conversion (optional)
```

**Model lookup:** `get_train_spec("llama3")` → imports `torchtitan.models.llama3` → calls `get_train_spec()` → returns the wired TrainSpec.

**Why this pattern?** The Trainer doesn't know about any specific model. It only talks to TrainSpec. This lets you add a new model by writing a new module with `get_train_spec()` and registering it.

---

## 4. A Concrete Model: Llama 3

### `torchtitan/models/llama3/__init__.py`
This is where the wiring happens. Read this to see how a model connects to the framework:

```python
llama3_args = {
    "debugmodel": TransformerModelArgs(dim=256, n_layers=6, ...),
    "8B": TransformerModelArgs(dim=4096, n_layers=32, ...),
    "70B": ...,
    "405B": ...,
}

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,           # from ./parallelize.py
        pipelining_fn=pipeline_llm,                  # from distributed/pipeline_parallel.py
        build_optimizers_fn=build_optimizers,         # from components/optimizer.py
        build_lr_schedulers_fn=build_lr_schedulers,   # from components/lr_scheduler.py
        build_dataloader_fn=build_text_dataloader,    # from hf_datasets/text_datasets.py
        build_tokenizer_fn=build_hf_tokenizer,        # from components/tokenizer.py
        build_loss_fn=build_cross_entropy_loss,       # from components/loss.py
        build_validator_fn=build_validator,            # from components/validate.py
        state_dict_adapter=Llama3StateDictAdapter,    # from ./state_dict_adapter.py
    )
```

### `torchtitan/models/llama3/args.py`
Model hyperparameters as a dataclass:

```python
@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1       # set from tokenizer
    max_seq_len: int = 2048    # updated from job_config.training.seq_len
    rope_theta: float = 500000
    attn_type: str = "sdpa"    # or "flex", "varlen"
    ...
```

`update_from_config()` pulls `seq_len` from the training config and sets `vocab_size` from the tokenizer.

### `torchtitan/models/llama3/model_def.py`
The actual transformer. Standard architecture:

```
Transformer
  ├── tok_embeddings (nn.Embedding)
  ├── layers (nn.ModuleDict of TransformerBlock)
  │     ├── attention_norm (RMSNorm)
  │     ├── attention (Attention with RoPE)
  │     ├── ffn_norm (RMSNorm)
  │     └── feed_forward (SwiGLU FFN: w1, w2, w3)
  ├── norm (RMSNorm)
  └── output (nn.Linear)
```

**Key methods:**
- `__init__()` — builds all layers
- `init_weights()` — initializes weights (called after model is moved to device)
- `forward(tokens)` — embedding → transformer blocks → norm → output logits

**Don't read every line.** Focus on the class structure and `forward()`.

---

## 5. Data Pipeline

### `torchtitan/hf_datasets/text_datasets.py`
How data goes from files to tokens:

```python
DATASETS = {
    "c4": DatasetConfig("allenai/c4", "en", "train"),
    "c4_test": DatasetConfig("tests/assets/c4_test", ...),   # local test data
}
```

**`HuggingFaceTextDataset`** — the core dataset class:
1. Loads dataset via HuggingFace `datasets` library
2. Tokenizes text samples into token IDs
3. Buffers tokens and yields fixed-length chunks of `(seq_len + 1)` tokens
4. Input = first `seq_len` tokens, labels = last `seq_len` tokens (shifted by 1)
5. Supports infinite looping for training
6. Stateful — saves buffer position for checkpoint resumption

**`build_text_dataloader()`** wraps the dataset in `ParallelAwareDataloader`.

### `torchtitan/components/dataloader.py`
Distributed-aware data loading wrapper:

- `ParallelAwareDataloader` — extends `StatefulDataLoader` from torchdata
- Handles data parallel sharding (each rank gets different data)
- Stateful for checkpoint resumption (saves per-rank position)
- Raises `DataloaderExhaustedError` (not `StopIteration`) when data runs out mid-accumulation

### `torchtitan/components/tokenizer.py`
Loads HuggingFace tokenizers from `tokenizer.json`:

- Auto-detects BOS/EOS tokens from `tokenizer_config.json`
- Determines if the tokenizer adds BOS/EOS automatically (some do, some don't)
- `encode()` adds BOS/EOS only if the tokenizer doesn't do it already

---

## 6. Loss & Optimization

### `torchtitan/components/loss.py`
Two key things happen here:

1. **Loss function:** Standard cross-entropy for language modeling
2. **Gradient accumulation rescaling:** Loss is divided by `gradient_accumulation_steps`

```python
# Without rescaling: loss would be N times larger with N accumulation steps
# With rescaling: loss magnitude is independent of accumulation count
loss_fn = rescale_accumulated_loss(cross_entropy_loss, accumulation_steps)
```

The `no_rescale()` context manager temporarily disables rescaling during validation.

### `torchtitan/components/optimizer.py`
Manages one optimizer per model part (needed for pipeline parallelism):

```python
class OptimizersContainer:
    # Wraps N optimizers (one per PP stage)
    # step() calls all of them
    # state_dict() flattens for resharding compatibility
```

**`build_optimizers_with_moe_load_balancing()`** — used by DeepSeek V3. Adds a pre-step hook that updates expert routing bias based on token-per-expert statistics. This prevents some experts from being starved of tokens.

### `torchtitan/components/lr_scheduler.py`
Implements **Warmup-Stable-Decay (WSD)** schedule:

```
LR
 ^
 |     ___________
 |    /           \
 |   /             \___
 |  /                  \___
 | /                       \
 +--+----------+-------+---→ steps
    warmup    stable   decay
```

- **Warmup:** linear 0 → 1
- **Stable:** constant at peak LR
- **Decay:** linear, sqrt, or cosine curve down to `min_lr_factor`

---

## 7. Training Loop (re-read `train.py` with full context)

Now re-read `Trainer` class in `train.py`. You'll understand every component it references.

### `Trainer.__init__(job_config)`
Builds everything in this order:

```
1. Set device (CUDA GPU from LOCAL_RANK)
2. init_distributed() → create ParallelDims mesh
3. Set random seeds for reproducibility
4. get_train_spec(model_name) → get all builders
5. Build tokenizer
6. Build dataloader
7. Build model on meta device (no memory used yet)
8. Build metrics processor
9. Calculate param count and FLOPs
10. Apply parallelism (TP/FSDP/PP) → model moves to real device
11. Build loss function with accumulation rescaling
12. Build optimizer and LR scheduler
13. Initialize CheckpointManager
14. Build validator (if enabled)
```

### `Trainer.train()` — the main loop

```python
def train(self):
    self.checkpointer.load()           # resume from checkpoint if exists

    with profiling_context:
        data_iterator = self.batch_generator(self.dataloader)

        while self.step < config.training.steps:
            self.step += 1
            self.train_step(data_iterator)     # forward + backward + optimizer
            self.checkpointer.save(self.step)  # save checkpoint if interval hit
            self.validator.validate(...)       # validate if freq hit
```

### `Trainer.train_step(data_iterator)` — one optimizer step

```python
def train_step(data_iterator):
    optimizers.zero_grad()

    # Gradient accumulation: multiple forward-backward before one optimizer step
    for microbatch in range(gradient_accumulation_steps):
        input_dict, labels = next(data_iterator)
        loss = self.forward_backward_step(input_dict, labels)

    clip_grad_norm_(...)    # prevent exploding gradients
    optimizers.step()       # update weights
    lr_schedulers.step()    # update learning rate

    # Log metrics (loss, throughput, MFU, memory)
```

### `Trainer.forward_backward_step()` — one microbatch

```python
def forward_backward_step(input_dict, labels):
    if pp_enabled:
        # Pipeline parallel: pp_schedule handles everything
        pp_schedule.step(inputs, target=labels, losses=losses)
    else:
        # Standard: forward → loss → backward
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        loss.backward()
    return loss
```

---

## 8. Metrics & Logging

### `torchtitan/components/metrics.py`
Collects and logs training metrics:

**What it tracks:**
- `global_avg_loss`, `global_max_loss` — reduced across all ranks
- `tps` — tokens per second per device (throughput)
- `tflops` — teraFLOPs achieved
- `mfu` — Model FLOPs Utilization (% of hardware peak)
- `memory` — peak GPU memory usage
- `lr` — current learning rate
- `grad_norm` — gradient norm after clipping

**MFU formula:** `mfu = 100 * flops_per_token * tokens_per_second / gpu_peak_flops`

**Logging backends:** W&B (via `WandBLogger`) and console (with color formatting).

### `torchtitan/tools/logging.py`
Simple logging setup: `init_logger()` configures a StreamHandler with `[titan]` prefix.

### `torchtitan/tools/utils.py`
Key utilities:
- `get_peak_flops(device_name)` — lookup table for GPU BF16 peak FLOPS (A100: 312 TFLOPS, H100: 989 TFLOPS, etc.)
- `GarbageCollection` — disables Python GC and runs it manually at intervals (prevents GC pauses during training)
- `Color` / `NoColor` — ANSI terminal colors for pretty console output

---

## 9. Checkpointing

### `torchtitan/components/checkpoint.py`
Distributed checkpoint save/load using PyTorch DCP (Distributed Checkpoint):

**What gets saved:**
- Model weights (flattened across PP stages)
- Optimizer states
- LR scheduler states
- Dataloader position (for resumption)
- Training state (step number, tokens seen)

**Key features:**
- **Resharding:** Flattened state dicts allow loading with different parallelism config
- **Async modes:** `disabled` (blocking), `async` (background thread), `async_with_pinned_mem` (pinned memory staging)
- **HF format:** Can save/load HuggingFace safetensors format via `StateDictAdapter`
- **Pruning:** Keeps only latest K checkpoints, deletes old ones in background

**Save flow:**
```
Trainer.train() → checkpointer.save(step)
  → _should_save(step)? (checks interval, last_step)
  → _flattened_model_states_sd() (merge model parts)
  → dcp_save() (write to disk, optionally async)
  → _purge_stale_checkpoints() (delete old ones)
```

### `torchtitan/protocols/state_dict_adapter.py`
Interface for converting between native and HuggingFace checkpoint formats:

```python
class BaseStateDictAdapter:
    def to_hf(state_dict) → hf_state_dict     # native → HF key mapping
    def from_hf(hf_state_dict) → state_dict    # HF → native key mapping
    def get_hf_storage_reader(path)             # for loading HF safetensors
```

Each model (llama3, deepseek_v3) provides its own adapter with model-specific key mappings.

---

## 10. Validation

### `torchtitan/components/validate.py`
Periodic evaluation during training:

```python
class Validator:
    def should_validate(step):
        return step == 1 or step % freq == 0

    def validate(model_parts, step):
        for model in model_parts:
            model.eval()                    # disable dropout
        with torch.no_grad():
            for batch in validation_dataloader:
                loss = forward(batch)       # handles PP if enabled
            metrics_processor.log_validation(loss, step)
        for model in model_parts:
            model.train()                   # re-enable dropout
```

---

## 11. Parallelism Infrastructure

### `torchtitan/distributed/parallel_dims.py`
The central parallelism abstraction. Creates a multi-dimensional device mesh:

```
Example: 32 GPUs with TP=4, PP=2, FSDP=4

Mesh dimensions: [pp=2, dp_replicate=1, fsdp=4, tp=4]
Total: 2 * 1 * 4 * 4 = 32 GPUs

Each dimension has a separate communication group.
```

**Key concept:** `build_mesh()` creates overlapping sub-meshes:
- `dense_mesh` — for non-MoE layers: [pp, dp_replicate, fsdp, tp]
- `sparse_mesh` — for MoE layers: [pp, dp_replicate, efsdp, ep, etp]
- `batch_mesh` — for data loading: flattened dp_replicate * fsdp * cp
- `loss_mesh` — for loss reduction: flattened batch * cp

### `torchtitan/distributed/utils.py`
Distributed utility functions:

- `init_distributed()` — initializes process groups, sets timeouts
- `dist_mean/max/sum()` — all-reduce operations across meshes
- `clip_grad_norm_()` — gradient clipping that handles PP + EP correctly
- `set_determinism()` — sets seeds consistently across ranks
- `create_context_parallel_ctx()` — context manager for sequence splitting in CP

### Other distributed files (read when needed):
- `activation_checkpoint.py` — applies selective or full gradient checkpointing
- `pipeline_parallel.py` — PP scheduling (splits model into stages)
- `tensor_parallel.py` — async TP toggle
- `expert_parallel.py` — EP dispatch/combine for MoE
- `dual_pipe_v.py` — overlapped PP+EP schedule (advanced)

---

## 12. MoE (Mixture of Experts)

### `torchtitan/models/moe/moe.py`
The MoE building blocks used by DeepSeek V3:

```
Input tokens
    │
    ▼
  Router (linear layer)
    │ scores each expert
    ▼
  Top-K selection
    │ pick best K experts per token
    ▼
  Token dispatch (scatter tokens to experts)
    │
    ▼
  Expert computation (grouped matmul or for-loop)
    │ each expert is a FeedForward (SwiGLU)
    ▼
  Token combine (gather results back)
    │
    ▼
  Output (weighted sum of expert outputs)
```

**Key classes:**
- `MoEArgs` — configuration (num_experts, top_k, score_func, load_balance_coeff)
- `FeedForward` — single expert (or shared expert) with SwiGLU
- `GroupedExperts` — batched expert computation
- `MoE` — full MoE layer with router, experts, optional shared experts
- `build_moe()` — factory function

### `torchtitan/models/deepseek_v3/__init__.py`
DeepSeek V3 differs from Llama 3 in two ways:
1. Uses MoE layers (some layers are dense, some are MoE)
2. Uses `build_optimizers_with_moe_load_balancing` for expert bias updates

---

## Data Flow Diagram

```
run_train.sh
    │
    ▼
ConfigManager.parse_args()
    │ TOML + CLI → JobConfig
    ▼
get_train_spec("llama3")
    │ → TrainSpec with all builders
    ▼
Trainer.__init__()
    │
    ├── build tokenizer ← tokenizer.json
    ├── build dataloader ← HuggingFace dataset → tokenize → chunk
    ├── build model (meta) → parallelize (TP/FSDP/PP) → init_weights
    ├── build loss_fn → wrap with accumulation rescaling
    ├── build optimizer(s) → one per PP stage
    ├── build lr_scheduler(s)
    ├── build CheckpointManager
    └── build Validator
    │
    ▼
Trainer.train()
    │
    ├── load checkpoint (if resuming)
    │
    └── for step in 1..N:
            │
            ├── train_step():
            │     ├── zero_grad
            │     ├── for microbatch in accumulation_steps:
            │     │     ├── get batch from dataloader
            │     │     ├── forward pass → logits
            │     │     ├── loss = cross_entropy(logits, labels) / accum_steps
            │     │     └── loss.backward()
            │     ├── clip_grad_norm
            │     ├── optimizer.step()
            │     └── lr_scheduler.step()
            │
            ├── log metrics (loss, throughput, MFU, memory)
            ├── save checkpoint (if interval)
            └── validate (if freq)
```

---

## File Index

| File | Lines | What to Look For |
|------|-------|-----------------|
| `train.py` | 721 | `Trainer.__init__`, `train()`, `train_step()`, `forward_backward_step()` |
| `config/job_config.py` | 780 | All the `@dataclass` definitions — skim field names |
| `config/manager.py` | 304 | `parse_args()`, `_apply_cli_overrides()`, `_coerce_value()` |
| `protocols/model.py` | 62 | `BaseModelArgs`, `ModelProtocol` |
| `protocols/train_spec.py` | 76 | `TrainSpec` dataclass, `get_train_spec()` |
| `models/llama3/__init__.py` | 110 | `llama3_args` dict, `get_train_spec()` wiring |
| `models/llama3/args.py` | 63 | `TransformerModelArgs` fields |
| `models/llama3/model_def.py` | 580 | `Transformer` class structure, `forward()` |
| `models/llama3/parallelize.py` | 310 | `parallelize_llama()` — how TP/FSDP/compile are applied |
| `models/deepseek_v3/__init__.py` | 166 | MoE model args with `MoEArgs` |
| `models/moe/moe.py` | 572 | `MoE` class, `GroupedExperts`, routing logic |
| `models/parallelize.py` | 390 | `apply_fsdp()`, `apply_moe_ep_tp()` — shared MoE parallelization |
| `hf_datasets/text_datasets.py` | 247 | `HuggingFaceTextDataset.__iter__()` — token chunking |
| `components/dataloader.py` | 133 | `ParallelAwareDataloader` — distributed data loading |
| `components/tokenizer.py` | 155 | `HuggingFaceTokenizer` — BOS/EOS inference |
| `components/loss.py` | 74 | `rescale_accumulated_loss()` — why loss is divided by accum steps |
| `components/optimizer.py` | 346 | `OptimizersContainer`, MoE load balancing hook |
| `components/lr_scheduler.py` | 186 | WSD schedule implementation |
| `components/metrics.py` | 485 | `MetricsProcessor.log()` — what metrics are tracked |
| `components/checkpoint.py` | 725 | `CheckpointManager.save()` / `load()` |
| `components/validate.py` | 215 | `Validator.validate()` |
| `distributed/parallel_dims.py` | 363 | `ParallelDims.build_mesh()` — mesh creation |
| `distributed/utils.py` | 532 | `init_distributed()`, `clip_grad_norm_()` |
| `tools/utils.py` | 208 | `get_peak_flops()`, `GarbageCollection` |
| `tools/profiling.py` | 141 | `maybe_enable_profiling()` — optional, skim |
