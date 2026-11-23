# Muse Spark MXFP8 head

The Muse HiMidLo head already emits FP32 logits. This opt-in override changes
its projection operands to MXFP8 and asks the native GEMM to write FP32 outputs.
It preserves parameter names and uses the same Muse scaling and smooth cap.
It requires the MXFP8 PyTorch/TorchAO runtime and an aligned head vocabulary.

```bash
python -m torchtitan.train \
  --module torchtitan_recipes.models.muse_spark \
  --config muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark \
  --override torchtitan_recipes.overrides.muse_spark_mxfp8_head.mxfp8_lm_head
```

By default the override moves scaling and softcap into the chunked loss's
`muse_softcap_ce` local compile region. It wraps the existing cross-entropy,
including token-count normalization, z-loss, and CP/TP vocabulary parallelism.
The head remains outside the compiled loss and keeps its native FSDP lifecycle.
To retain separate postprocessing, pass
`--override 'torchtitan_recipes.overrides.muse_spark_mxfp8_head.mxfp8_lm_head={"fuse_softcap_loss":false}'`.

The override does not change the default Muse head. MXFP8 operands and compiled
FP32 postprocessing can change numerical results. Forced balanced routing in
the benchmark recipe does not establish learned-router training convergence.
