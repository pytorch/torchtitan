# Shared-expert stream overlap

`MoE.Config.overlap_shared_experts=True` runs shared experts on a CUDA side
stream. The default is serial execution. The opt-in requires shared experts,
CUDA inputs and TP=1; unsupported configurations raise. Ordinary MoE and Kimi's
latent MoE use the same stream boundary.

```python
from dataclasses import replace

moe_config = replace(moe_config, overlap_shared_experts=True)
```

The host still constructs router, routed experts, shared experts and output
addition in their original order. An input-readiness event recorded before
routed work lets the shared stream start without waiting for routed completion:

```text
caller stream:  input-ready -> router / routed experts -> wait shared -> add
                    |                                      ^
shared stream:      +-------> shared experts ---------------+
```

This does not guarantee simultaneous execution or an end-to-end speedup.
Host launch timing and available GPU resources determine actual overlap;
measure it rather than treating a configured stream as performance evidence.

The enclosing FSDP boundary establishes parameter readiness before the input
event. Explicit event/stream waits establish shared input readiness and output
completion. `record_stream` protects allocator lifetime for inputs and outputs,
including no-grad forwards. Recompute placeholders must not receive allocator
operations; during checkpoint replay the explicit join and checkpoint lifetime
keep recomputed tensors live. Autograd executes a branch's backward on its
forward stream.

Warm up on the final CUDA device before capture to create the module's stream.
Capture retains the fork/join dependencies; replay does not reconstruct them
from Python. The policy does not alter routing, activation arithmetic, parameter
layout or the expert backend's activation-slot contract. It may increase peak
activation memory because the two branches can be live simultaneously.

The recipe override `torchtitan_recipes.overrides.shared_expert_overlap` enables
the policy on supported MoE configs and leaves no-shared-expert configs unchanged.
Whole-model compilation and GraphTrainer parallel-branch lowering are not
provided by this policy.
