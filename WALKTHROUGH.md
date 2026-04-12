# Codebase walkthrough

A reading guide for the MoE training codebase. Follow in order, or jump around —
each section is self-contained.

## 0. The 30-second version

You run `torchrun --nproc_per_node=4 train.py --config configs/moe_tiny.yaml`.
That loads a YAML → pydantic `Config` → calls `src.trainer.train(cfg)`, which
builds a MoE transformer with FSDP2 + Expert Parallel, iterates over a
pre-tokenized local dataset, computes cross-entropy loss, and writes
checkpoints. Everything is on a single file path from top to bottom.

## 1. Entry point and top-level layout

```
train.py                ← CLI shell (typer); just loads YAML and calls train()
config.py (src/config/) ← pydantic user-facing config
src/trainer.py          ← the actual training loop
src/data.py             ← dataset + dataloader
src/models/moe/model.py ← the MoE Transformer (Attention, RMSNorm, Block, Model)
src/models/parallelize.py ← applies EP and FSDP to the model
src/components/         ← optimizer, lr_scheduler, tokenizer, checkpoint (from torchtitan)
src/distributed/        ← ParallelDims, ExpertParallel, activation_checkpoint (from torchtitan)
src/logging.py          ← logfire console + stdlib FileHandler
src/utils.py            ← GarbageCollection, has_cuda_capability, device_type helpers
```

The `components/` and `distributed/` directories are TorchTitan code we kept
intact — everything else is our own.

## 2. train.py → src/trainer.py

Start here: [`train.py`](train.py)

```python
with open(config) as f:
    cfg = Config(**(yaml.safe_load(f) or {}))
train(cfg)
```

That's all `train.py` does. Typer is the CLI library. `Config` is a pydantic
model defined in `src/config/config.py`. All real logic lives in
`src.trainer.train`.

Open [`src/trainer.py`](src/trainer.py). The `train()` function is one long
sequential script — no class, no inheritance. Read it top to bottom. The
sections are roughly:

1. **Init** — logging, mutual-exclusion check (quack ↔ compile), build `JobConfig`
2. **Distributed** — `init_distributed`, build `ParallelDims`, set device
3. **Model config** — load tokenizer, read its vocab_size, make `model_cfg`
4. **Dataloader** — `build_text_dataloader` with the batch mesh
5. **Model** — build on meta device, verify expert/non-expert param split
6. **Parallelism** — `EP → AC → compile → FSDP`, then `model.to_empty()` + `init_weights()`
7. **Optimizer, LR scheduler, checkpoint manager**
8. **Cross-entropy dispatch** — closure captures either `F.cross_entropy` or `quack.cross_entropy`
9. **Training loop** — grad accum, backward, clip, step, checkpoint, log, eval

## 3. Config: src/config/config.py

[`src/config/config.py`](src/config/config.py) is the user-facing config. Every
field in a YAML file corresponds to an attribute on a pydantic model.

Sections (top-level keys):
- `model` — architecture (layers, dim, heads, experts, etc.)
- `training` — steps, batch sizes, lr, warmup, precision
- `parallelism` — dp_shard, ep
- `data` — tokenizer (HF id or local path), dataset_path
- `activation_checkpoint`, `compile`, `quack` — optimization toggles
- `logging` — log_step (how often to print), log_dump (where to write train.log)
- `eval` — enable, dataset_path, eval_step
- `checkpoint` — enable, checkpoint_step, checkpoint_dump

At the bottom of the file, `build_job_config(cfg: Config) -> JobConfig`
translates our pydantic config into TorchTitan's internal `JobConfig`
dataclass. The reason for this split: TorchTitan's components
(`CheckpointManager`, optimizer builders, etc.) expect the dataclass form, so
we keep them untouched and translate at the boundary.

The TorchTitan-side dataclasses live in [`src/config/job_config.py`](src/config/job_config.py).
You mostly never touch this file; it's an implementation detail.

## 4. Data: src/data.py

[`src/data.py`](src/data.py) has three layers:

1. **`load_text_dataset(path)`** — dispatches between `load_from_disk` (for HF
   `save_to_disk` format) and `load_dataset(path)` (for raw parquet/jsonl).
2. **`TextDataset`** — wraps an HF dataset, tokenizes and packs samples into
   fixed-length sequences of `seq_len + 1` tokens, yields `(input, label)`
   pairs where `label = input[1:]`. Supports both finite and infinite modes.
3. **`ParallelAwareDataloader`** — thin wrapper over `torchdata.StatefulDataLoader`
   that remembers each DP rank's state for checkpoint resume.

Key detail: each DP rank gets a *different* slice of the dataset via
`split_dataset_by_node(ds, dp_rank, dp_world_size)`. No rank sees the same
tokens as another.

`extract_text(sample)` pulls the text field out, trying `text`, `content`,
`raw_content` in order. If your dataset uses something else, add a case there
or write a wrapper.

## 5. Model: src/models/moe/model.py

[`src/models/moe/model.py`](src/models/moe/model.py) is where the math lives.
Read it in order:

### 5a. RMSNorm wrapper (lines ~12–40)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps, use_quack=False):
        ...
    def forward(self, x):
        if self.use_quack:
            from quack import rmsnorm
            return rmsnorm(x.reshape(-1, x.shape[-1]), self.weight, eps=self.eps).reshape(x.shape)
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
```

The `use_quack` flag is threaded through from `cfg.quack.enable` at train-time.
When True, calls into the QuACK CuTe-DSL kernel for ~speedup on memory-bound
norms. When False, uses PyTorch's native `F.rms_norm`.

### 5b. RoPE (lines ~42–85)

Standard complex-number RoPE with `theta=500000` (Llama3/Qwen convention).
`precompute_freqs_cis` creates the phase table once, and `apply_rotary_emb`
does the view-as-complex/multiply/view-as-real dance. Called once per
attention layer during forward.

### 5c. Attention (lines ~87–135)

GQA attention (`n_heads` query heads, `n_kv_heads` key/value heads, with
`n_heads` divisible by `n_kv_heads`). The only non-obvious bit:

```python
output = flash_attn_func(xq, xk, xv, causal=True)
```

**Why flash_attn directly instead of SDPA?** Two reasons:

1. **FA3-ready**. When flash-attn 3 lands, we swap the function name in one place.
2. **Native GQA**. `flash_attn_func` handles different head counts for Q vs K/V
   without us having to call `repeat_kv`. The attention class is ~25 lines
   shorter than it would be with SDPA.

Tensor layout is `(batch, seqlen, nheads, head_dim)` — no transpose needed
before calling flash_attn.

### 5d. TransformerBlock (lines ~137–175)

```
x = x + attention(attention_norm(x), freqs_cis)
x = x + moe(ffn_norm(x))
```

Pre-norm, residual connections, standard transformer block. The only layout
constraint: attributes must be named `attention`, `attention_norm`, `ffn_norm`,
`moe`, and there must be a `moe_enabled: bool` attribute. The parallelization
code in `src/models/parallelize.py` inspects these by name.

The `moe` submodule itself is from TorchTitan (`src/models/moe/moe.py`) — we
build it via `build_moe(args, dim, hidden_dim)`. Important MoE args (from
`MoEArgs`):

- `num_experts` — total routed experts
- `top_k` — experts per token
- `num_shared_experts` — **a single FFN whose hidden_dim is multiplied by this**
  (not literally N experts; see section 11 below)
- `score_func` — `"sigmoid"` or `"softmax"` for the router
- `use_grouped_mm` — use `torch._grouped_mm` for the batched expert computation
  (needs SM90+)
- `load_balance_coeff` — auxiliary-loss-free balancing (DeepSeek style)

### 5e. MoETransformer (lines ~177–227)

Top-level model. Has four attributes that `apply_fsdp` relies on by name:
`tok_embeddings`, `layers` (ModuleDict keyed by `"0"`, `"1"`, ...), `norm`,
`output`. If you rename any of these, FSDP wrapping breaks.

Forward is trivial: embed → loop over blocks → final norm → output projection.

## 6. Parallelism: src/models/parallelize.py + src/distributed/

[`src/models/parallelize.py`](src/models/parallelize.py) exports two functions:

- **`apply_moe_ep(model, ep_mesh)`** — applies `ExpertParallel` to each
  `block.moe.experts`, which shards the expert weight matrices along dim 0
  (expert dim) and injects all-to-all token dispatch/combine around the
  expert computation.
- **`apply_fsdp(model, dp_mesh, ...)`** — applies FSDP2 (`fully_shard`) to
  each block, the embedding, and the head. When `ep_degree > 1`, the expert
  submodule gets FSDP'd on `edp_mesh` (a separate mesh) and the gradient
  divide factor is set manually.
- **`apply_compile(model, ...)`** — wraps each block in `torch.compile`,
  carefully skipping the expert submodule because its FSDP hooks cause graph
  breaks.

The order in which `trainer.train` applies these matters:

```
EP  →  Activation Checkpoint  →  Compile  →  FSDP
```

(EP first because it shards parameters; AC wraps blocks in
`checkpoint_wrapper`; compile wraps blocks in `OptimizedModule`; FSDP wraps
everything last because it needs to see the final module tree.)

The mesh construction itself lives in
[`src/distributed/parallel_dims.py`](src/distributed/parallel_dims.py). You
don't usually touch this — `ParallelDims(dp_shard=4, ep=4, ...)` +
`parallel_dims.build_mesh()` gives you the meshes, and you fetch them by name
(`parallel_dims.get_mesh("fsdp")`, `get_mesh("ep")`, `get_mesh("batch")`,
etc.).

For a 4-GPU setup with `dp_shard=4, ep=4`:
- `fsdp` mesh = all 4 GPUs (for non-expert params)
- `ep` mesh = all 4 GPUs (experts sharded across them)
- `efsdp` mesh = size 1 (since `fsdp * tp / (etp * ep) = 4*1/(1*4) = 1`)
- `batch` mesh = size 4 (for dataloader sharding)

## 7. The training loop proper

Back in [`src/trainer.py`](src/trainer.py), the core of `train()`:

```python
while train_state.step < total_steps:
    train_state.step += 1
    step_start = time.perf_counter()

    optimizers.zero_grad()
    for _ in range(grad_accum_steps):
        input_dict, labels = next(data_iter)
        tokens = input_dict["input"].to(device)
        labels = labels.to(device)
        pred = model(tokens)
        loss = cross_entropy_fn(pred, labels)
        (loss / grad_accum_steps).backward()
        accumulated_loss += loss.detach().item() / grad_accum_steps

    grad_norm = dist_utils.clip_grad_norm_(..., ep_enabled=parallel_dims.ep_enabled)
    checkpointer.maybe_wait_for_staging()
    optimizers.step()
    lr_schedulers.step()
    train_state.ntokens_seen += tokens_per_step
    checkpointer.save(step, last_step=(step == total_steps))

    if step % log_step == 0: logger.info(...)
    if eval_step > 0 and step % eval_step == 0: run_eval(...)
```

### Gradient accumulation with FSDP2

We divide `loss` by `grad_accum_steps` and call `.backward()` on each
microbatch. FSDP2's reduce-scatter happens on every backward, but the
gradients accumulate into the same `.grad` tensors across microbatches so the
total is correct. **No `no_sync()` is needed** — that was an FSDP1 pattern.

### Gradient clipping with EP

`dist_utils.clip_grad_norm_(..., ep_enabled=True)` handles the fact that
expert parameters live on a smaller mesh than non-expert parameters — their
gradient norms need to be computed locally then combined correctly.

## 8. Eval: run_eval()

[`src/trainer.py`](src/trainer.py) defines `run_eval` near the top. When
`cfg.eval.enable=True`, it runs every `cfg.eval.eval_step` training steps:

1. `model.eval()`, build a fresh eval dataloader pointing at `cfg.eval.dataset_path`
2. Loop: `next(iter)` → compute loss + top-1 accuracy → accumulate
3. **FSDP trap**: all ranks must do the same number of forward passes (because
   FSDP forward issues collectives). We handle this via
   `dist.all_reduce(has_batch_flag, op=MIN)` — as soon as any rank runs out of
   data, all ranks stop together.
4. Reduce `(loss_sum, correct, tokens)` across the batch mesh
5. Return `{loss, ppl = exp(loss), top1_acc}`
6. `model.train()`

No generation metrics (those need autoregressive sampling which is expensive).
Everything comes from a single teacher-forced forward pass per batch.

## 9. Checkpoints

We reuse TorchTitan's `CheckpointManager` unchanged. It's a DCP-based
(`torch.distributed.checkpoint`) system that:
- Saves sharded state dicts in parallel across ranks
- Handles resharding on load (if you change `dp_shard`)
- Supports async staging via pinned memory for near-zero save overhead

Our config exposes three knobs:
- `checkpoint.enable` — master switch
- `checkpoint.checkpoint_step` — save frequency
- `checkpoint.checkpoint_dump` — absolute path to save to

The `CheckpointManager` is a no-op if `enable: false` — no folders are
created, save() is a no-op. Useful for smoke tests.

State saved: model weights, optimizer state, LR scheduler state,
`TrainState(step, ntokens_seen)`, and the dataloader position so resuming
picks up exactly where it left off.

## 10. Logging

[`src/logging.py`](src/logging.py) exports a `logger` object and `init_logger`.
`logger.info(msg)` goes to **two** places:

1. **Console** via `logfire` — rich formatting, can optionally stream to
   logfire cloud if you set a `LOGFIRE_TOKEN` env var
2. **File** via Python stdlib `logging.FileHandler` — plain text at
   `<log_dump>/train.log`

`init_logger(log_dir)` attaches the FileHandler, which is why `train()` calls
it with `cfg.logging.log_dump`.

## 11. MoE internals (read when curious)

[`src/models/moe/moe.py`](src/models/moe/moe.py) is imported verbatim from
TorchTitan and contains:

- `MoEArgs` dataclass
- `MoE` nn.Module (the full routed+shared expert layer)
- `TokenChoiceTopKRouter` (softmax or sigmoid router with optional group-limited routing)
- `GroupedExperts` (uses `torch._grouped_mm` for batched expert computation)
- `TokenReorderer` (permutes tokens by expert assignment before dispatch)
- Auxiliary-loss-free load balancing via `expert_bias` buffer updated in an
  optimizer pre-hook

**About `num_shared_experts`**: despite the name, this is *not* N separate
expert modules. It's a single FFN whose hidden_dim is multiplied by N. Shared
experts process *every* token (unlike routed experts which only see top_k tokens).
The intuition: universally-useful features (grammar, basic semantics) go into
the shared experts, freeing routed experts to specialize.

## 12. Mesh math for the curious

When you set `dp_shard=4, ep=4, tp=1, pp=1, cp=1` with world_size=4:

```
batch         = dp_replicate × dp_shard   = 1 × 4 = 4
fsdp          = dp_shard × cp             = 4 × 1 = 4
efsdp         = fsdp × tp / (etp × ep)    = 4 × 1 / (1 × 4) = 1
ep            = 4 (all GPUs)
```

Which means:
- Every GPU gets a different data shard (`batch=4`)
- Non-expert params are FSDP-sharded across all 4 GPUs (`fsdp=4`)
- Expert params are EP-sharded (each GPU holds `num_experts / 4` experts) and
  NOT additionally FSDP-sharded (`efsdp=1`), because there's nothing left to
  shard after EP took its cut

## 13. When something goes wrong

- **NaN loss** → check that `mixed_precision_param="bfloat16"` and
  `mixed_precision_reduce="float32"` (the fp32 reduce is critical for MoE
  stability)
- **Hang in eval** → check you're using `all_reduce MIN of has_batch`
  (already done, but if you modify `run_eval`, keep this pattern)
- **`expected stride 1` in quack backward** → the `_ContiguousGrad` workaround
  in `cross_entropy_fn` handles this; don't remove it
- **Mismatched EP correctness** → run `scripts/ep_correctness.sh`; with
  `force_load_balance=True` the EP=1 and EP=4 losses should match exactly
  (difference 0.0)
- **FSDP "parameters still on meta device"** → must call `model.to_empty(device=...)`
  before `model.init_weights()` when FSDP is applied; we do this in `train()`

## 14. Recommended reading order

If you want to read the whole codebase once, go in this order:

1. [`train.py`](train.py) — 25 lines, the entry point
2. [`src/config/config.py`](src/config/config.py) — pydantic models + `build_job_config`
3. [`configs/moe_tiny.yaml`](configs/moe_tiny.yaml) — an actual config
4. [`src/models/moe/model.py`](src/models/moe/model.py) — Attention → Block → Model
5. [`src/data.py`](src/data.py) — TextDataset and dataloader
6. [`src/models/parallelize.py`](src/models/parallelize.py) — apply_moe_ep, apply_fsdp
7. [`src/trainer.py`](src/trainer.py) — the loop that ties it all together

Optional deep-dives:
- [`src/models/moe/moe.py`](src/models/moe/moe.py) — MoE internals (torchtitan)
- [`src/distributed/parallel_dims.py`](src/distributed/parallel_dims.py) — mesh construction
- [`src/components/checkpoint.py`](src/components/checkpoint.py) — DCP checkpointing
- [`scripts/ep_correctness.py`](scripts/ep_correctness.py) — how we verify EP numerically
