# FlexShard as the FSDP backend

`parallelism.fsdp_backend` selects what shards parameters over the
data-parallel mesh:

- `FSDP2Config()`, the default, applies FSDP2's `fully_shard` (see
  [FSDP.md](FSDP.md)).
- `FlexShardConfig()` applies [FlexShard](https://github.com/meta-pytorch/flex_shard)
  instead. It is experimental.

The code is in `flex_shard_fsdp.py` and `fsdp.py`. The
[`flex_shard/`](flex_shard/README.md) package is a different thing: the
optimizer-compute layouts DistMuon uses.

## Using it

Set the backend in your configuration. Here `llama3_flex_shard` is
`llama3_fsdp2` with FlexShard:

```python
# my_runs.py
from torchtitan.config.parallelism import FlexShardConfig
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def llama3_fsdp2() -> Trainer.Config:
    return llama3_debugmodel(attn_backend="flex")


def llama3_flex_shard() -> Trainer.Config:
    config = llama3_fsdp2()
    config.parallelism.fsdp_backend = FlexShardConfig()
    return config
```

`parallelism.fsdp_reshard_after_forward` and the `precision` dtypes apply to
FlexShard as they do to FSDP2.

flex_shard has no release yet. CI installs it from git at a pinned commit, and
you can do the same from the repository root:

```bash
bash .github/scripts/install_flex_shard.sh
```

It installs with `--no-deps`, so the environment must already have what
flex_shard requires: a PyTorch 2.16 nightly and torchao. torchtitan imports
flex_shard only when `FlexShardConfig()` is selected, and raises an
`ImportError` when the training engine is built if it isn't installed.

## How it shards

`apply_fsdp_to_decoder` computes the decoder's FSDP groups once: the embedding,
the final norm with the output projection, each transformer block, and the
parameters left to the root. FSDP2 calls `fully_shard` on each group. FlexShard
gets one bucket per group, with the group's mesh, placements, mixed precision
and reshard policy, and `fsdp2_compatible=True`, so its all-gathers and
reduce-scatters match FSDP2's byte for byte. Under expert parallelism, each MoE
block gets a second bucket for its routed experts on `edp_shard`, as FSDP2
gives it a second group. Under context parallelism, the buckets shard over
`dp_shard` and `cp` flattened, as `fully_shard` does. Under tensor parallelism,
each parameter is its tp rank's shard and declares TP's split, or tp as the
group its grad is partial over, before FlexShard shards it.

Outside forward and backward, FlexShard parameters, their grads and the
optimizer states are plain `torch.Tensor` local shards, not DTensors:

- **Checkpoints.** The model's and the optimizer's state dicts declare where
  each local shard sits in the full tensor, with the attributes of DCP's
  `CheckpointableTensor` protocol (`global_shape`, `global_offsets`,
  `local_offsets`, `local_sizes`). DCP then writes the same checkpoints as for
  FSDP2's DTensors, and either backend loads the other's. A Hugging Face load
  reads full tensors, a group of parameters at a time, through the state dict
  adapters' conversions, and each shard keeps its part.
- **Initialization.** Right after sharding, each parameter's `param_init`
  function is wrapped with flex_shard's `init_from_full`, which runs it on the
  full parameter and keeps this rank's part, so each shard is its slice of the
  model a single device of the same type initializes from the same seed. FSDP2's
  DTensor init draws each shard's values on its own. From the same seed, the two
  backends therefore start from different weights. FlexShard briefly holds each
  full parameter on every rank while it initializes it, e.g. a routed-expert
  parameter with all its experts.
- **Grad-norm clipping.** `clip_grad_norm_` reduces the local shards' norms
  over each bucket's mesh in the order DTensor uses, so the norm matches
  FSDP2's.
- **DistMuon.** DistMuon plans its redistributions from each parameter's
  storage layout. For a FlexShard parameter, torchtitan gives it a
  `StorageLayout` (mesh, placements and global shape) derived from the
  parameter's bucket, which matches FSDP2's DTensor spec, so DistMuon runs the
  same collectives and updates. Its momentum is a plain local shard that
  declares its parameter's layout to DCP.

## Parity with FSDP2

From the same weights, with the same configuration and
`debug.deterministic=True`, losses, grad norms, grads and parameters are
bitwise identical to FSDP2's. Both backends start from the same weights when
they load the same checkpoint, e.g. a seed checkpoint, which is created without
sharding.

`tests/unit_tests/gpu/distributed/test_flex_shard_fsdp.py` compares the two
backends bitwise. The integration tests `llama3_flex_shard+tp+cp`,
`deepseek_v3_flex_shard+ep`, `deepseek_v3_flex_shard+cp+pp+ep` and
`deepseek_v3_dist_muon_flex_shard+ep` (with DistMuon) load the seed checkpoint their FSDP2 counterparts load and must
reproduce those counterparts' Real PG goldens, and
`flex_shard_checkpoint` resumes an FSDP2 checkpoint and loads an FSDP2 Hugging
Face export. To compare the backends on your own configuration,
`scripts/loss_compare.py` creates a seed checkpoint from the baseline config and
loads it in both runs. The configuration must reproduce bitwise from run to
run under FSDP2 alone, as the example's flex attention does:

```bash
python scripts/loss_compare.py . . \
    --baseline-module my_runs --baseline-config llama3_fsdp2 \
    --test-config llama3_flex_shard \
    --baseline-ngpus 2 --test-ngpus 2 --assert-equal --metrics loss,grad_norm
```

Memory and communication timing differ from FSDP2's in two places:

- FlexShard all-gathers the embedding again in backward, which FSDP2 skips
  (`set_unshard_in_backward(False)`).
- Under expert parallelism, FlexShard reshards the routed experts as soon as
  they finish forward or backward. FSDP2 waits for the whole block.

## What it supports

| | Supported with `FlexShardConfig()` |
| --- | --- |
| Models | Decoders that shard through `Decoder._apply_fsdp`, also with tied embedding and output weights (`enable_weight_tying`). The parity tests cover Llama 3 (also with tied weights), Qwen3 MoE, and DeepSeek V3, also with an MTP layer |
| Parallelism | FSDP over `dp_shard`; tensor parallelism, with or without sequence parallelism; context parallelism; expert parallelism with `AllToAllTokenDispatcher`; pipeline parallelism, with a PyTorch whose pipeline schedules accept FlexShard modules (pytorch/pytorch#205659); and their combinations |
| Training | `ChunkedLossWrapper`, gradient accumulation, activation remat, local compile, validation, EMA (`optim.ema`, also with `offload_to_cpu`), DistMuon without tensor parallelism, and accelerator graphs without expert parallelism (the four capture settings in [ACCELERATOR_GRAPH.md](ACCELERATOR_GRAPH.md)) |
| Checkpointing | `CheckpointManager` with synchronous or `async_mode="async"` saves, seed checkpoints, `export_dtype`, `initial_load_path` (including FSDP2 checkpoints), and `initial_load_in_hf` |

## What it rejects

`validate_flex_shard_training` in `flex_shard_fsdp.py` raises a `ValueError`
for these when the training engine is built, before anything else:

| Rejected | Why |
| --- | --- |
| HSDP and DDP (`data_parallel_replicate_degree > 1`) | torchtitan's FlexShard buckets don't replicate over a second mesh axis yet. |
| `fsdp_symm_mem_scope`, `fsdp_enable_cpu_offloading`, `fsdp_unshard_in_current_stream` | torchtitan's FlexShard path has no equivalent of these yet. |
| `fsdp_defer_gradient_reduction` | FlexShard has no equivalent of FSDP2's `set_is_last_backward`. |
| Models that override `_apply_fsdp`: multimodal models, DeepSeek V4, FLUX and the transformers modeling backend | Only `Decoder._apply_fsdp` passes `fsdp_backend` on, so these would shard with FSDP2. |
| Modules without `param_init` that have a `reset_parameters()` | FlexShard initializes parameters only through `param_init`. `reset_parameters()` would draw each rank's shard as the whole parameter, with the same values on every rank. The check reads the model config, so it also rejects modules with no parameters, such as the `RMSNorm(elementwise_affine=False)` QK norms of muse_glimmer and muse_spark. |
| `preserve_dtype_params` | A bucket all-gathers in one dtype. |
| MXFP8 and NVFP4 weights | FlexShard's plain-tensor parameters drop the quantized all-gather these weights implement for FSDP2. |
| With expert parallelism: Dist-MoE, token dispatchers other than `AllToAllTokenDispatcher`, and `output_postprocess` | The others aren't checked against FSDP2 yet. `output_postprocess` parameters are replicated over `ep`, and the expert bucket wouldn't sum their grads over it. |
| Checkpointers other than `CheckpointManager` | `torch_checkpointing`'s resharder reads only DTensor placements. |
| `checkpointer.async_mode="async_with_pinned_mem"` | The save process receives the state dict through `torch.multiprocessing`, which drops the shard layouts. |
| `checkpointer.initial_load_in_hf` with pipeline parallelism | The Hugging Face load groups and broadcasts over all ranks, which assumes every rank holds the same parameters. Pipeline stages don't. |
| `checkpointer.initial_load_in_hf_quantized`, `checkpointer.last_save_in_hf` | Not supported yet. The state dict adapters' save path needs DTensor parameters. |
| DistMuon with tensor parallelism | DistMuon's storage layouts for FlexShard parameters hold the buckets' split, not TP's. DistMuon's recipes reject TP with FSDP2 too. |
| `debug.spmd_typechecking` | Not checked with FlexShard yet. |
| GraphTrainer, FaultTolerantTrainer, the RL trainer, and any trainer other than `torchtitan.trainer.Trainer` | GraphTrainer shards with SimpleFSDP, TorchFT hooks FSDP2's parameter groups, and the RL trainer would publish local shards as full weights. |

Accelerator graphs with expert parallelism are rejected too, by two other
checks: torchtitan captures graphs under expert parallelism only with HybridEP's
`non_blocking_capacity_factor` or with Dist-MoE, and FlexShard rejects both.
`ChunkedLossWrapper` with a frozen decoder and a trainable `lm_head` raises at
the first loss call: nothing would finish `lm_head`'s bucket after its calls.
