# Muse Spark MXFP8 CP1 benchmark

The opt-in recipe `muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark_cp1`
runs DP64, EP64, CP1, TP1, PP1 with one 32K sequence per rank per step. It
inherits forced balanced routing and random initialization from the existing
benchmark; it does not train a usable model.

| Setting | Value |
| --- | --- |
| Persistent parameters | FP32 |
| FSDP parameter/reduction dtype | BF16 / BF16 |
| Fused AdamW moments | BF16, on device |
| Reshard after forward | Always |
| Outer gradient accumulation | One microbatch per step |
| Dist-MoE precision/pipeline | MXFP8, staged |
| Saved expert activations | 40 GiB total per live slot, including required state |
| Head projection | MXFP8 operands, native FP32 GEMM output |
| Loss | 8 chunks, FP32 scale/softcap and existing cross-entropy |
| Added local compile regions | `muse_softcap_ce`, `moe_router_topk` |

Use the base 64-GPU benchmark's dependencies, tokenizer assets, pinned-memory
allocator setting, NUMA placement, and single NVLink domain. Set
`NCCL_NVLS_ENABLE=1`, `NCCL_MIN_CTAS=1`, and `NCCL_MAX_CTAS=8` in the launch
environment. The CTA bound controls NCCL kernels, not the Dist-MoE barriers.

```bash
python -m torch.distributed.run <your-64-rank-launch-options> \
  -m torchtitan.train \
  --module torchtitan_recipes.models.muse_spark \
  --config muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark_cp1
```

For compute testing on one sufficiently large GPU, use the companion
`muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark_cp1_fake` recipe with
`NGPU=64` and `--comm-backend fake`. FakePG omits real transport and barriers;
its throughput and maximum activation budget cannot be treated as real-run
results. Use at least ten steps and exclude startup/profiling from measurements.

The FP32 output change avoids BF16 epilogue rounding; MXFP8 operands and
compiled postprocessing still change arithmetic. Learned-router convergence
and the base symmetric-memory teardown limitation need separate validation.
