# Router-score absorption

Router-score absorption for DeepEP routed experts, implemented in
[`RoutedExperts`](moe.py) and [`GroupedLinear`](../linear/linear.py).
Local, AllToAll, and HybridEP keep post-expert scoring; Dist-MoE owns its
compute and communication optimizations. Remat policies are described in
[REMAT.md](../../../distributed/REMAT.md).

## Overview

Applying a router score after W2 makes the score gradient depend on W2's
unscaled routed output. Keeping that tensor costs memory; discarding it
requires another W2 forward GEMM during backward. Absorption moves the score
multiplication before the linear projection, so the score gradient instead
reuses W2's already-required input gradient. A policy can then save W2 without
retaining its routed output for scoring.

The diagram compares post-W2 scoring (top) with absorption (bottom):

![Post-W2 scoring retains the routed output for backward; absorption moves scoring before W2.](https://github.com/user-attachments/assets/6f4e5155-6cfb-455f-925f-8d9165899666)

[Diagram source: #5543](https://github.com/pytorch/torchtitan/pull/5543)

The diagram shows the bias-free case. `Gelu` represents the configured gated
activation, and its smaller-input annotation assumes `F < D`. Avoiding W2
forward replay requires saving W2 and combine; FullAC still replays W2.

`T` is the number of input tokens on this rank, `R` the number of routed rows,
`D` the model width, `F` the expert intermediate width, and `E` the number of
local experts. Dispatch groups rows by expert. The routed output `[R,D]` can
be much larger than the combined output `[T,D]` because each token visits
several experts.

For one bias-free routed row, let `W` be W2, `h` its input, `s` its score,
and `g` the gradient of the weighted output:

```text
Score after W2                    Score before W2
z = W @ h; y = s * z               a = s * h; y = W @ a
ds = dot(g, z)                    da = W.T @ g; ds = dot(da, h)
     needs W2's output                 reuses input and input gradient
```

Both forms still need W2's ordinary input-gradient and weight-gradient GEMMs.
The optimization removes a dependency on its forward output, not those
backward GEMMs.

## Configurations

DeepEP's score placement depends on the expert configuration. `B` below is
the selected expert bias expanded over routed rows; omit it for bias-free
experts. Each score broadcasts over the row's features.

| Expert configuration | Score placement | Weighted expert output |
| --- | --- | --- |
| Bias-free, no `output_postprocess` | Before W2 | `W2(s * h)` |
| GPT-OSS bias, no `output_postprocess` | Before W2 and on expanded bias | `W2(s * h) + s * B` |
| Any `output_postprocess` | After postprocessing; absorption disabled | `s * output_postprocess(W2(h) + B)` |

### Remat policies

The table isolates the W2/score dependency for bias-free BF16 experts without
a postprocess or a real dtype conversion after W2. Other activations,
parameters, communication buffers, and temporary workspaces are excluded.
"Post-W2" is a comparison computation, not a DeepEP configuration switch.
Scoring is replayed in that comparison; putting it inside a SAVE region would
retain its `[R,D]` operand even when W2 itself is configured to replay.

| Policy | Score after W2 | Absorption with saved DeepEP combine |
| --- | --- | --- |
| No remat | Autograd retains the unscaled `[R,D]` output for the score gradient and `[R,F]` input for the weight gradient. | No unscaled W2 output is needed, but both original and scaled `[R,F]` inputs are retained. The extra scaled input offsets some or all of the removed output storage. |
| Selective remat with W2 saved | Scoring needs the unscaled `[R,D]` output, so saving W2 keeps that output resident. | The W2 output need not stay resident. Replay rebuilds the scaled input for backward and skips the W2 forward GEMM. |
| Full remat (`FullActivationRematConfig`) | W2 output can be discarded, then rebuilt during backward. | W2 output need not remain resident, but its `grouped_mm` still replays. Absorption does not change this policy. |

With absorption, saving W2 obtains the low W2-output retention of full remat
while avoiding its W2 forward replay. `DefaultActivationRematConfig` already
saves W2. To save it while otherwise recomputing policy-controlled regions:

```python
from torchtitan.distributed.activation_remat import ActivationRematConfig

config.activation_remat = ActivationRematConfig(
    save_regions=["*routed_experts.w2.grouped_mm"],
)
```

This replays dispatch and W13; DeepEP combine is always saved. Adding
`"*token_dispatcher.ep_communication"` also saves dispatch. Under
`torch.compile`, the graph partitioner also determines actual replay work;
inspect the graph or profile rather than inferring GEMM counts from a policy
name alone.

### Memory and compute

- **Routed-output storage:** one BF16 `[R,D]` output occupies `2 * R * D`
  bytes. At `R = 32768` and `D = 7168`, that is 448 MiB per MoE layer, or
  25.4 GiB across 58 layers of potentially avoided output retention with saved
  W2. Actual `R` varies with routing, EP layout, and padding.
- **Combined-output storage:** removing an unnecessary pin on `[T,D]` is a
  saving separate from absorption, enabled by skipping replay of trailing
  consumers. At `T = 4096` and the same `D`, it is 56 MiB per layer, or
  3.17 GiB across 58 layers.
- **Compute:** each avoided bias-free W2 forward replay costs approximately
  `2RFD` FLOPs. Rebuilding its scaled input costs `RF` multiplications;
  gradient scaling and score-gradient reductions move from width `D` to `F`.
  GPT-OSS also scores its expanded bias at width `D`, adding pointwise work
  and reductions rather than another GEMM.

These are analytical storage and compute examples, not measured peak-memory
or step-time improvements. They exclude allocator reuse and transient peak
memory, and do not measure communication overlap.

## Submodule behavior

- **GroupedLinear:** absorbs optional per-row scores before W2. Products
  compute in FP32 through type promotion and are cast back before GEMM:
  `(h_RF * s_R.float().reshape(-1, 1)).to(h_RF.dtype)`. Keeping the activation
  operand in its original dtype avoids a materialized FP32 copy that the
  multiplication would save for backward. Moving score multiplication before
  W2 changes BF16 rounding; exact-arithmetic equivalence does not imply
  identical outputs, losses, or gradients.
- **GPT-OSS bias:** `GptOssGroupedLinear._expand_grouped_bias` selects the
  bias by expert for each routed row, using the grouped-GEMM offsets:
  `B_RD[r, :] = b_ED[expert_for_row(r), :]`. Thus `s` scales the row-expanded
  bias, not the expert axis. Unused tail rows receive zero bias. W13's bias
  is added before activation and is not scaled separately. The W2 bias add
  follows the `grouped_mm` remat decision: replaying even this cheap add
  after a saved matmul would retain its full output. Without remat, the
  score gradient retains expanded bias `[R,D]`; under remat it is cheaply
  rebuilt from per-expert bias and offsets.
- **RoutedExperts postprocessing:** a configured `output_postprocess` runs
  before FP32 score multiplication and disables absorption. It still needs
  W2's output, as does a real dtype conversion during replay: saved W2 keeps
  that output resident, while full remat rebuilds it. The common BF16
  `type_as` is a no-op and does not impose this dependency.
- **DeepEP dispatch:** returns expert-row-aligned scores. Scores are rebuilt
  by replayed dispatch/permutation or retained with saved dispatch outputs.
  Dispatch replay follows `ep_communication` and uses deterministic receive
  ordering, adding a sort per dispatch.
- **DeepEP combine:** receives already-weighted rows and always saves the
  entire region, including unpermute. Replaying unpermute would require
  retaining or rebuilding W2's output. Its backward retains routing indices:
  `R` int64 indices cost `8R` bytes, or 256 KiB at `R = 32768`, excluding the
  opaque handle and other retained tensors. Only forward replay is skipped;
  original forward and backward communication still run.
- **Diagnostics:** use `torch_remat.collect_trace()` for the effective
  regions and `torch_remat.format_saved_tensors_report(loss)` for resident
  retained storage. Measure peak memory and steady-state step time
  separately on the target model and topology.
