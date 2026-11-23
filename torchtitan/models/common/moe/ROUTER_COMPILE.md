# Post-gate router compilation

The opt-in `moe_router_topk` region reuses the implementation proposed in
[PR #5343](https://github.com/pytorch/torchtitan/pull/5343), authored by Felipe
Mello. The focused port keeps other models' default compile regions and
numerical reference files unchanged. It preserves the existing `moe_aux_loss`
region through a shared local-statistics helper.

The flow is:

```text
gate projection -> score / select / gather / normalize / scale /
                   routing map / counts / probability sums
                -> buffer updates / auxiliary-gradient injection
```

The gate projection and state updates remain outside the region. Recompute
therefore does not repeat buffer updates. The compiled selection uses fused
argmax rounds where applicable; ties select the lower candidate index. Forced
balanced benchmark routers bypass selection but still compile post-gate scores,
gathers, normalization, routing maps, and statistics.

The new Muse CP1 benchmark enables this region. Other recipes can opt in by
adding `moe_router_topk` to their `model.local_compile.regions`. Compiled FP32
arithmetic can differ from eager rounding; numerical and convergence checks
remain necessary for a training recipe using learned routing.
