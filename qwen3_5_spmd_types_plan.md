# Minimal Qwen3.5 spmd_types Enablement Plan

Goal: make `--parallelism.spmd_backend spmd_types` work for Qwen3.5 with the
smallest change set, following the existing decoder model enablement pattern
from Qwen3, DeepSeek V3, GPT-OSS, and Llama3.

`full_dtensor` is not in scope. Keep Qwen3.5's current
`full_dtensor` `NotImplementedError` unless a separate effort explicitly
enables it.

## Supported Parallelisms

Qwen3.5 currently supports these parallelisms in the default backend path:

- FSDP / HSDP via `data_parallel_shard_degree` and
  `data_parallel_replicate_degree`.
- TP via `tensor_parallel_degree`.
- EP for MoE variants via `expert_parallel_degree`.
- PP via `pipeline_parallel_degree`, with the vision encoder assigned to the
  first pipeline stage.
- Activation checkpointing and compile.

Qwen3.5 explicitly does not support CP today. Preserve the existing CP rejection:
GatedDeltaNet needs full-sequence recurrence, and multimodal CP needs vision
scatter before CP sharding.

The minimal `spmd_types` target should cover the same non-CP parallelisms:

- dense Qwen3.5: FSDP/HSDP, TP, PP, and combinations already used by configs
- MoE Qwen3.5: FSDP/HSDP, TP, EP, PP, and combinations already used by configs

Dense configs are `debugmodel`, `0.8B`, `2B`, `4B`, `9B`, and `27B`.
MoE configs are `debugmodel_moe`, `35B-A3B`, `122B-A10B`, and `397B-A17B`.

## 1. Mirror the Standard Decoder Parallelize Path

Update `torchtitan/models/qwen3_5/parallelize.py` to follow the `spmd_types`
parts of `qwen3/parallelize.py`.

- Import `validate_config`, `resolve_fsdp_mesh`, and `resolve_sparse_fsdp_mesh`.
- Keep the existing `full_dtensor` rejection.
- Add a dedicated `parallelism.spmd_backend == "spmd_types"` path.
- In the `spmd_types` path, call:
  - `validate_config(parallel_dims, model)`
  - `model.parallelize(parallel_dims)`
- Keep Qwen3.5's existing CP rejection before this path, since CP is explicitly
  unsupported for GatedDeltaNet and multimodal scatter.
- Keep the existing async TP check.
- Use `resolve_fsdp_mesh(parallel_dims)` for dense FSDP when backend is
  `spmd_types`.
- Use `resolve_sparse_fsdp_mesh(parallel_dims)` for expert FSDP when EP is
  enabled under `spmd_types`.
- Pass `dp_mesh_dims`, `edp_mesh_dims`, and
  `enable_symm_mem=parallelism.enable_fsdp_symm_mem` into
  `apply_fsdp_to_decoder`, matching Qwen3/GPT-OSS.

This should fix the current structural bug where Qwen3.5 asks for the legacy
`fsdp` mesh under `spmd_types`, even though that backend uses the `dp` /
`dp_shard` mesh layout.

## 2. Make Vision Encoder FSDP Compatible with spmd_types

Qwen3.5 applies FSDP to `model.vision_encoder` separately. Under
`spmd_types`, it should use the same dense FSDP mesh and `dp_mesh_dims` chosen
by `resolve_fsdp_mesh()`.

Minimal change:

- Extend `_apply_fsdp_to_vision_encoder(...)` with an optional
  `dp_mesh_dims: DataParallelMeshDims | None = None`.
- Pass `dp_mesh_dims` through to `fully_shard(...)` when it is not `None`.
- Reuse the same `dp_mesh` / `dp_mesh_dims` computed for the decoder.

## 3. Audit Qwen3.5 Sharding Contracts for spmd_types

This is the core model-specific work. Qwen3.5 already has many SPMD-style
`ShardingConfig`s, but several were written to work through DTensor inference.
The `spmd_types` path is stricter: every module-boundary type contract must be
explicit.

Audit and fix these rules:

- If a config declares `in_dst_shardings`, it must also declare matching
  `in_src_shardings` for the same tensor names.
- If a config declares `out_dst_shardings`, it must also declare
  `out_src_shardings`.
- Inter-block decoder activations under sequence parallelism should use
  `dense_sequence_parallel_placement()`, i.e. `V` with
  `PartitionSpec(dp, (cp, tp), None)`.
- Module internals that need full hidden states before a column-sharded matmul
  should gather TP to `spmd.R`.
- Tensors whose backward should slice the gathered value, not all-reduce it,
  should use `spmd.I`. The important existing examples are labels for
  loss-parallel CE and full-vocab gathers in logprob code.
- Add local-map regions only where the computation is genuinely local-only or
  not expressible through ordinary SPMD propagation.

Concrete Qwen3.5 sharding items to check first:

- `_replicate_norm()` declares `out_dst_shardings=R` without `out_src_shardings`.
  Add `out_src_shardings=R`.
- `_qk_norm_sharding()` declares `out_dst_shardings=head_plc` without
  `out_src_shardings`. Match Qwen3's QK norm pattern and set both.
- The Qwen3.5 tok embedding override declares `out_dst_shardings=R`; keep the
  current `out_src=P`, because embedding produces vocab-partial output before
  the boundary all-reduce.
- `_set_shared_expert_gate_sharding()` declares `out_dst_shardings=R` without
  `out_src_shardings`. Add `out_src_shardings=R`; the gate is replicated and
  multiplied with the shared-expert partial output.
- Vision patch embedding declares `out_dst_shardings=R` without
  `out_src_shardings`. Add `out_src_shardings=R`.
- `RMSNormGated` in GatedDeltaNet declares `out_dst_shardings=_norm_plc`
  without `out_src_shardings`. Add `out_src_shardings=_norm_plc`.
- `GatedDeltaKernel` declares only `in_dst_shardings` for `q/k/v/g/beta`.
  Add matching `in_src_shardings`; these tensors are already head-sharded
  (`tp=S(2)`) from upstream projections and reshapes.
- `GatedDeltaNet` declares `out_dst_shardings=dense_sequence_parallel_placement()`
  without `out_src_shardings`. Its `out_proj` rowwise output already
  reduce-scatters to SP, so set `out_src_shardings` to the same SP placement.

The existing dense/MoE structure should still be reused:

- dense layers go through `set_dense_ffn_sharding(...)`
- MoE layers go through `set_moe_sharding_config(...)`
- Qwen3.5's shared-expert sigmoid gate has an extra local sharding config

Do not fork dense vs. MoE model code. Use the existing per-layer
`feed_forward is not None` / `moe is not None` branches.

Also verify MRoPE. The generic `CosSinRoPE` path has SPMD annotations around
cache reshaping, but Qwen3.5 overrides `_reshape_cache()` for 3D
`mrope_positions` and manually unwraps/rewraps DTensors. Under `spmd_types`,
that path may need explicit `assert_type` or a small local-map region so
`mrope_positions`, the cache, and the returned broadcast cache have the
expected local types.

## 4. Handle Qwen3.5 Inputs Only If Runtime Requires It

The generic `annotate_input_spmd_types()` helper currently annotates only:

- `inputs`
- `labels`
- `positions`

Qwen3.5 may additionally pass:

- `mrope_positions`
- `pixel_values`
- `pixel_values_videos`
- `grid_thw`
- `grid_thw_videos`

Use two input contracts:

- Text-side tensors are regular global SPMD tensors:
  - `inputs`: `DP:S(0), CP:S(1), TP:R`
  - `positions`: `DP:S(0), CP:S(1), TP:R`
  - `mrope_positions`: `DP:S(0), CP:S(1), TP:R`
  - `labels`: `DP:S(0), CP:S(1), TP:I`
- Multimodal tensors are not globally representable as `spmd.S(dim)` because
  `num_vision` and `max_num_patch` are rank-local and data-dependent:
  - `pixel_values`
  - `pixel_values_videos`
  - `grid_thw`
  - `grid_thw_videos`

Annotate multimodal tensors as `spmd.V`, and handle the multimodal path inside
an `spmd.local` / local-map region. The local region should cover:

- vision encoder forward
- image/video token-count computation
- vision placeholder lookup
- scatter of vision embeddings into text embeddings

After scatter, re-enter the regular decoder contract with `[B, L, D]`
embeddings. Qwen3.5 already keeps the embedding path `TP:R` before the first
decoder block; the first block then restores sequence parallelism.

## 5. Check GatedDeltaNet's Conv1d Local Path

`GatedDeltaNet._causal_conv()` has a DTensor-only local-map workaround for
channel-sharded depthwise Conv1d.

Under `spmd_types`, tensors are local tensors with SPMD metadata, so the
`isinstance(x, DTensor)` branch will not run.

Minimal first attempt:

- Do not change this before running.
- Test TP with `spmd_types` and compare against default.
- If Conv1d shape/groups fail under TP, add a guarded `spmd_types` path that
  runs the same local convolution contract on local shards.
- If typechecking fails only, add local-map/type assertions around this region
  rather than changing math.

## 6. Add Targeted Coverage

Start with one narrow integration variant, modeled after existing model tests:

```bash
--module qwen3_5 --config qwen35_debugmodel_moe
--parallelism.spmd_backend spmd_types
--parallelism.data_parallel_shard_degree 2
--parallelism.pipeline_parallel_degree 2
--parallelism.tensor_parallel_degree 2
--parallelism.expert_parallel_degree 4
```

Because this includes PP, the existing integration test wrapper will not enable
SPMD typechecking. After the backend path runs, add a smaller non-PP variant for
typechecking if the model/config can fit:

```bash
--module qwen3_5 --config qwen35_debugmodel
--parallelism.spmd_backend spmd_types
--parallelism.data_parallel_shard_degree 2
--parallelism.tensor_parallel_degree 2
--debug.spmd_typechecking
activation-checkpoint:none
```

## 7. Validation Matrix

Aim for bitwise-identical loss and grad_norm between the default backend and
`spmd_types` for 10 training steps, using:

```bash
--training.steps 10
--debug.seed 42
--debug.deterministic
```

Do not use `--debug.deterministic_warn_only`.

Use `scripts/loss_compare.py` / TensorBoard values for loss and grad_norm; stdout
rounding is not enough.

Qwen3.5 has these coverage axes:

- Model family:
  - dense: `debugmodel`, `0.8B`, `2B`, `4B`, `9B`, `27B`
  - MoE: `debugmodel_moe`, `35B-A3B`, `122B-A10B`, `397B-A17B`
- Full-attention backend, via `model_registry(attn_backend=...)`:
  - `flex`
  - `flex_flash` on Hopper/Blackwell only
  - `varlen`
- GatedDeltaNet backend:
  - `fla_chunked` (current default)
  - `fla_fused_recurrent`
  - `torch_native` for reference/numerical testing
  - Note: `model_registry` does not currently expose `fla_backend`, so covering
    non-default GatedDeltaNet backends may require a small config-registry knob
    or a test-only converter.
- MoE token dispatcher backend, for MoE flavors:
  - local dispatch when EP is disabled
  - standard all-to-all (`moe_comm_backend="standard"`) when EP is enabled
  - `deepep`, `hybridep`, and `minimal_async_ep` are out of scope for this
    enablement and should be handled in a separate PR
- Parallelism shape:
  - FSDP only
  - FSDP + TP
  - FSDP + TP + PP
  - MoE only: FSDP + TP + EP
  - MoE only: FSDP + TP + EP + PP
  - HSDP variants if `data_parallel_replicate_degree > 1` is in scope
  - CP remains unsupported for Qwen3.5 and should keep failing clearly

Validation order:

1. Fake backend config construction with `spmd_types`.
2. GPU smoke for the existing Qwen3.5 MoE PP+FSDP+TP+EP integration shape.
3. Non-PP TP smoke with `--debug.spmd_typechecking`.
4. Default vs `spmd_types` 10-step bitwise loss/grad_norm compare on
   `qwen35_debugmodel`.
5. Repeat the bitwise compare on `qwen35_debugmodel_moe`.
6. Expand across the backend axes above. For large real-size configs, at minimum
   run the same flavor/parallelism shape with a feasible short deterministic
   smoke and record any configs that cannot be run due to hardware, dependency,
   or memory limits.

### Current Smoke Status

Environment used:

- Conda env: `full_spmd_tt`
- `PYTHONPATH` begins with local `spmd_types`, then
  `/data/users/pianpwk/flash-linear-attention`,
  `/data/users/pianpwk/vision`,
  `/data/users/pianpwk/full_spmd_types_torchtitan/pytorch`, and this repo.
- `FLA_TILELANG=0`
- Typechecking smokes use
  `--debug.spmd_typechecking activation-checkpoint:none`, because SPMD
  typechecking is not compatible with selective AC.

Passing one-step typechecking smokes:

- Dense FSDP only: `qwen35_debugmodel`, `dp_shard=2`.
- Dense TP only: `qwen35_debugmodel`, `tp=2`.
- Dense FSDP + TP: `qwen35_debugmodel`, `dp_shard=2`, `tp=2`.
- Dense HSDP: `qwen35_debugmodel`, `dp_replicate=2`, `dp_shard=4`.
- MoE EP off: `qwen35_debugmodel_moe`, `dp_shard=8`, `ep=1`.
- MoE EP on: `qwen35_debugmodel_moe`, `dp_shard=8`, `ep=2`.
- MoE FSDP + TP + EP: `qwen35_debugmodel_moe`, `dp_shard=4`,
  `tp=2`, `ep=2`.
- Full-attention `varlen`: `qwen35_debugmodel`, `tp=2`.

Passing one-step non-typechecking smoke:

- Dense FSDP + TP with default selective AC:
  `qwen35_debugmodel`, `dp_shard=2`, `tp=2`.

Known failing or blocked axes:

- GatedDeltaNet `torch_native` with `tp=2` fails the local-map output
  boundary for `GatedDeltaKernel`: the kernel output is typed `R`, while the
  current sharding contract expects `V`.
- GatedDeltaNet `fla_fused_recurrent` reaches backward and then fails in FLA
  itself because backward is not implemented for that backend.
- Full-attention `flex_flash` is blocked in this environment because PyTorch's
  flex flash lowering requires the CUTE flash attention package
  (`flash-attn-4`), which is not installed.

Still unvalidated:

- PP shapes. These were intentionally deferred.
- 10-step bitwise default-backend vs `spmd_types` comparisons for loss and
  grad norm.
- Larger dense and MoE model flavors.
- Out-of-scope MoE dispatchers: `deepep`, `hybridep`, and
  `minimal_async_ep`.

### Current 10-Step Bitwise Status

All comparisons below used:

```bash
--training.steps 10
--debug.seed 42
--debug.deterministic
--metrics.enable_tensorboard
--metrics.log_freq 1
```

Loss and grad norm were compared from full-precision TensorBoard scalar values:

- `loss_metrics/global_avg_loss`
- `grad_norm`

Passing default-backend vs `spmd_types` comparisons:

- Dense FSDP 1D:
  - config: `qwen35_debugmodel`
  - shape: `dp_shard=2`
  - result: bitwise equal loss and grad norm for all 10 steps
  - output: `/tmp/qwen35_bitwise_20260709_175248_fsdp`
- Dense HSDP ND data mesh:
  - config: `qwen35_debugmodel`
  - shape: `dp_replicate=2`, `dp_shard=4`
  - result: bitwise equal loss and grad norm for all 10 steps
  - output: `/tmp/qwen35_bitwise_20260709_175501_hsdp`
- MoE FSDP with local dispatcher:
  - config: `qwen35_debugmodel_moe`
  - shape: `dp_shard=8`, `ep=1`, `tp=1`
  - result: bitwise equal loss and grad norm for all 10 steps
  - output: `/tmp/qwen35_bitwise_20260709_175625_moe_epoff`
- MoE FSDP with standard all-to-all EP:
  - config: `qwen35_debugmodel_moe`
  - shape: `dp_shard=8`, `ep=2`, `tp=1`
  - result: bitwise equal loss and grad norm for all 10 steps
  - output: `/tmp/qwen35_bitwise_20260709_175740_moe_ep2`

Blocked comparisons:

- Dense TP 1D:
  - config: `qwen35_debugmodel`
  - shape: `tp=2`, `dp_shard=1`
  - default backend fails before the `spmd_types` run starts
  - failure: mixed `torch.Tensor` and `DTensor` in Qwen3.5 vision rotary
    position embedding, at `torch.outer(seq, self.inv_freq)`
  - output: `/tmp/qwen35_bitwise_20260709_175349_tp`
- Dense FSDP + TP ND:
  - config: `qwen35_debugmodel`
  - shape: `dp_shard=2`, `tp=2`
  - default backend fails with the same mixed `torch.Tensor` / `DTensor`
    vision rotary issue
  - output: `/tmp/qwen35_bitwise_20260709_175428_fsdp_tp_default_probe`
- Dense HSDP + TP ND:
  - config: `qwen35_debugmodel`
  - shape: `dp_replicate=2`, `dp_shard=2`, `tp=2`
  - default backend fails with the same mixed `torch.Tensor` / `DTensor`
    vision rotary issue
  - output: `/tmp/qwen35_bitwise_20260709_175859_tp_nd_probes`
- MoE FSDP + TP + EP ND:
  - config: `qwen35_debugmodel_moe`
  - shape: `dp_shard=4`, `tp=2`, `ep=2`
  - default backend fails with the same mixed `torch.Tensor` / `DTensor`
    vision rotary issue
  - output: `/tmp/qwen35_bitwise_20260709_175859_tp_nd_probes`

Interpretation:

- The `spmd_types` path has already passed TP and FSDP+TP one-step
  typechecking smokes.
- Full 10-step bitwise validation for any TP-including shape is currently
  blocked by the default backend failing on this WIP checkout before comparison
  can be made.
