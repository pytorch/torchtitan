# DeepSeek-V3 16B CODA benchmark shapes

## Capture

The source is an unfused GraphTrainer joint graph captured with:

- model: `graph_trainer_deepseek_v3_16b` (15.706B parameters, 2.661B active)
- fake world: FSDP4, EP4, TP1, standard EP
- batch: local batch 4, sequence length 4096 (`M=16384`)
- activation checkpointing: full
- dataset: `c4_test`
- graph passes: `coda_patterns: []`

The analyzed graph is:

```text
outputs/profiling/dsv3_fake/graph/dsv3-16b-unfused-shapes-20260812/
  tlparse/-_-_-_-/after_joint_transformer_block_bucketing_reordering_pass_142.txt
```

It contains 858 `aten.mm` nodes and 312 `aten._grouped_mm` nodes. It contains
no FlexGEMM or CODA nodes. Routed expert token dimensions remain symbolic after
EP all-to-all, so the grouped GEMMs are excluded from this dense FlexGEMM suite
and remain owned by DistMoE.

## Benchmark inventory

| Case | Pattern | Real GEMM shape | Occurrence |
| --- | --- | --- | --- |
| `b1_lm_head_input_grad_cast` | B1 | `(2048, 102400) @ (102400, 2048)` | 8 loss chunks |
| `b2_shared_expert_swiglu_backward` | B2 | `M=16384, D=2048, P=2816`, 3 GEMMs | 26 MoE layers |
| `b2_dense_ffn_swiglu_backward` | B2 | `M=16384, D=2048, P=10944`, 3 GEMMs | dense layer 0 |
| `b4_router_input_grad_add` | B4 | `(16384, 64) @ (64, 2048)` | 26 MoE layers |
| `b5_mla_kv_rmsnorm_backward` | B5 | `(16384, 4096) @ (4096, 512)` | 27 layers |
| `b6_shared_expert_weight_grad_cast` | B6 | `(2816, 16384) @ (16384, 2048)` | shared W1/W3 |
| `b7_attention_input_grad_merge` | B7 | `(16384, 576) @ (576, 2048)` + `(16384, 3072) @ (3072, 2048)` | 27 layers |
| `f2_kv_rmsnorm` | F2-KV | `2048 -> 576`, RMSNorm(512), `512 -> 4096`, `M=16384` | 27 layers |
| `f3_attention_output` | F3-A | `(16384, 2048) @ (2048, 2048)` | 27 layers |
| `f3_moe_output` | F3-B | `(16384, 2816) @ (2816, 2048)` | 26 MoE layers |
| `f3_dense_ffn_output` | F3-B | `(16384, 10944) @ (10944, 2048)` | dense layer 0 |
| `f4_shared_expert_swiglu` | F4 | two `(16384, 2048) @ (2048, 2816)` | 26 MoE layers |
| `f4_dense_ffn_swiglu` | F4 | two `(16384, 2048) @ (2048, 10944)` | dense layer 0 |

F2-Q and F6 are intentionally absent. This 16B configuration has a direct Q
projection (`2048 -> 3072`) rather than Q LoRA plus RMSNorm, and its router is
softmax without the sigmoid-plus-expert-bias epilogue.

## Extracted forward samples

These are real nodes copied from the post-bucketing FX graph.

### F2-KV and direct Q

```python
mm_7: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_29, t_7)

mm_8: "bf16[16384, 576]" = torch.ops.aten.mm.default(view_32, t_8)
split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(
    _unsafe_view_8, [512, 64], -1
)
_fused_rms_norm_4 = torch.ops.aten._fused_rms_norm.default(
    getitem_19, [512], _unsafe_view_538, 1e-05
)
mm_9: "bf16[16384, 4096]" = torch.ops.aten.mm.default(view_40, t_9)
```

The `3072` direct-Q GEMM feeds B7 backward but does not form F2-Q.

### F3-A attention output

```python
mm_3: "bf16[16384, 2048]" = torch.ops.aten.mm.default(view_19, t_3)
_unsafe_view_3 = torch.ops.aten.reshape.default(mm_3, [4, 4096, 2048])
add = torch.ops.aten.add.Tensor(embedding, _unsafe_view_3)
_fused_rms_norm_2 = torch.ops.aten._fused_rms_norm.default(
    add, [2048], _unsafe_view_522, 1e-05
)
```

### Dense-layer F4 and F3-B

```python
mm_4: "bf16[16384, 10944]" = torch.ops.aten.mm.default(view_22, t_4)
silu = torch.ops.aten.silu.default(_unsafe_view_4)
mm_5: "bf16[16384, 10944]" = torch.ops.aten.mm.default(view_24, t_5)
mul_2 = torch.ops.aten.mul.Tensor(silu, _unsafe_view_5)

mm_6: "bf16[16384, 2048]" = torch.ops.aten.mm.default(view_26, t_6)
add_1 = torch.ops.aten.add.Tensor(add, _unsafe_view_6)
_fused_rms_norm_3 = torch.ops.aten._fused_rms_norm.default(
    add_1, [2048], _unsafe_view_541, 1e-05
)
```

### Shared-expert F4 and F3-B

```python
mm_12: "bf16[16384, 2816]" = torch.ops.aten.mm.default(view_91, t_14)
silu_2 = torch.ops.aten.silu.default(_unsafe_view_14)
mm_13: "bf16[16384, 2816]" = torch.ops.aten.mm.default(view_93, t_15)
mul_8 = torch.ops.aten.mul.Tensor(silu_2, _unsafe_view_15)

mm_14: "bf16[16384, 2048]" = torch.ops.aten.mm.default(view_95, t_16)
add_5 = torch.ops.aten.add.Tensor(view_83, _unsafe_view_14)
add_6 = torch.ops.aten.add.Tensor(add_2, add_5)
```

## Extracted backward samples

### B1 chunked LM head

```python
view_1827: "bf16[2048, 102400]" = torch.ops.aten.reshape.default(
    view_1826, [2048, 102400]
)
mm_217: "bf16[2048, 2048]" = torch.ops.aten.mm.default(
    view_1827, _unsafe_view_1109
)
_to_copy_723: "f32[4, 512, 2048]" = torch.ops.aten._to_copy.default(
    alias_173, dtype=torch.float32
)
```

The graph repeats this sequence eight times because the loss is chunked into
512 tokens per local batch element.

### B2 and B6 shared expert

```python
mm_240: "bf16[16384, 2816]" = torch.ops.aten.mm.default(
    view_1887, _unsafe_view_1106
)
mul_159 = torch.ops.aten.mul.Tensor(view_1888, silu_52_recomputed)
mul_160 = torch.ops.aten.mul.Tensor(view_1888, _unsafe_view_265_recomputed)

mm_241: "bf16[2816, 2048]" = torch.ops.aten.mm.default(
    t_311, view_1818_recomputed
)
mm_242: "bf16[16384, 2048]" = torch.ops.aten.mm.default(
    view_1890, _unsafe_view_1107
)
silu_backward = torch.ops.aten.silu_backward.default(
    mul_160, _unsafe_view_264_recomputed
)
mm_243: "bf16[2816, 2048]" = torch.ops.aten.mm.default(
    t_315, view_1816_recomputed
)
mm_244: "bf16[16384, 2048]" = torch.ops.aten.mm.default(
    view_1893, _unsafe_view_1105
)
add_140 = torch.ops.aten.add.Tensor(view_1891, view_1894)
_to_copy_763: "f32[2816, 2048]" = torch.ops.aten._to_copy.default(
    mm_243, dtype=torch.float32
)
```

The layer-0 dense backward has the same topology with `P=10944`; it is kept as
a separate B2 case because its GEMM aspect ratios and tuning choices differ.

### B4 router input gradient

```python
mm_246: "f32[16384, 2048]" = torch.ops.aten.mm.default(
    view_1910, _to_copy_704_recomputed
)
_to_copy_770: "bf16[4, 4096, 2048]" = torch.ops.aten._to_copy.default(
    view_1911, dtype=torch.bfloat16
)
add_143 = torch.ops.aten.add.Tensor(add_142, _to_copy_770)
```

### B5 KV projection plus RMSNorm backward

```python
mm_250: "bf16[16384, 512]" = torch.ops.aten.mm.default(
    view_1919, _unsafe_view_1100
)
view_1920 = torch.ops.aten.reshape.default(mm_250, [4, 4096, 512])
_fused_rms_norm_backward_2 = torch.ops.aten._fused_rms_norm_backward.default(
    view_1920,
    getitem_694_recomputed,
    [512],
    alias_210,
    _unsafe_view_1099,
    [True, True],
)
```

### B7 direct-Q and KV input-gradient merge

```python
mm_252: "bf16[16384, 2048]" = torch.ops.aten.mm.default(
    view_1927, _unsafe_view_1098
)
mm_254: "bf16[16384, 2048]" = torch.ops.aten.mm.default(
    view_1931, _unsafe_view_1097
)
add_145 = torch.ops.aten.add.Tensor(view_1928, view_1932)
```

## Running the suite

For the CUDA 13.2 nightly used during validation, pin the toolkit libraries to
the venv. Loading this wheel against the host CUDA 13.0 libraries can produce
`cudaErrorNoKernelImageForDevice` on SM103.

```bash
CUDA_ROOT=/home/bahuang/local/venvs/torch-cu132-nightly/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib"

python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench_16b --list

python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench_16b \
  --case f4_shared_expert_swiglu \
  --warmup 5 \
  --rounds 5 \
  --iterations 20
```

Run different cases in fresh processes on SM103. QuACK CUDA dialect
initialization can make a later case fail if several FlexGEMM programs run in
one process. The tuner already provides this isolation:

```bash
python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_autotune \
  --suite 16b \
  --case f4_shared_expert_swiglu \
  --devices 0,1,2,3 \
  --search full
```

Every FlexGEMM uses `tuned: true`; F4's SiLU-producing GEMM also uses
`fast_math: true`. Pass `--config` once per FlexGEMM to force explicit QuACK
configurations during tuning.
