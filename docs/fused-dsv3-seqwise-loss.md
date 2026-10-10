# Fused DeepSeek V3 sequence-wise loss

Current TorchTitan calls this objective `MicrobatchWiseLoadBalanceLoss`. This
experimental override replaces its raw loss calculation with
`FusedDSv3SeqwiseLossFunction`, a `torch.autograd.Function` with explicit forward
and backward kernels, one CUDA launch in each direction. The model/operator
layer and CUDA launcher layer are separate.

```bash
--override torchtitan_recipes.overrides.fused_dsv3_seqwise_loss.fused_dsv3_seqwise_loss
```

The Function computes the same raw token-sum loss as current `main`:

```python
counts_E = routing_map_TE.to(scores_TE.dtype).sum(0)
frequencies_E = torch.nn.functional.normalize(counts_E, p=1, dim=0) * E
probabilities_TE = torch.nn.functional.normalize(scores_TE, p=1, dim=-1)
raw_sum = (frequencies_E * probabilities_TE.sum(0)).sum()
```

It consumes the router's actual routing map, including biased/group-limited
assignments. It does not perform a second top-k or assume every row has eight
assignments. Backward differentiates the normalized scores. `AuxLoss.inject`
retains the native coefficient, dynamic denominator, metric name/accumulator,
and checkpoint behavior. The router, gate, expert weights, and state-dict keys
are unchanged.

The fast path requires GB300, contiguous FP32 scores `[4096,256]` with 16-byte
alignment, a matching contiguous boolean routing map, no padding mask, and
CP=TP=1. Other shapes, dtypes, layouts, or parallelism use the native loss,
including its CP/TP reductions. No collective is removed or simulated.

Each loss module owns a nonpersistent integer arrival counter. Calls on that
module must be stream-ordered, just like its mutable metric accumulator.
Independent modules can execute concurrently. The CUDA sources are packaged
in the wheel and JIT-built through `load_inline`, requiring a matching CUDA
toolkit and C++ compiler. Warm up both directions before capture, or call
`kernels.prepare()` from the override module before compilation/capture.

This loss-only override is an alternative to the joint router/loss override.
The override framework rejects applying both to the same router, preventing
duplicate auxiliary-loss injection.

## Validation and component timings

On PyTorch `2.16.0.dev20261007+cu130`, GB300, tests compare bit patterns for
positive/signed/zero/tiny scores, the epsilon boundary, sparse/full/empty routing
maps, backward, dynamic loss scaling, metric state under remat, full-graph
Function compilation, CUDA replay, and independent streams. CPU tests cover
model config replacement, native fallback, and state-dict compatibility.

```bash
python -m pytest --import-mode=importlib tests/unit_tests/{cpu,gpu}/test_fused_dsv3_seqwise_loss.py
python -m benchmarks.fused_dsv3_seqwise --buffers 96 --samples 30
```

Median microseconds per raw-loss component call, with 96 independent FP32
`[4096,256]` score tensors (384 MiB), interleaved CUDA-graph samples:

| Phase | Native | Fused |
| --- | ---: | ---: |
| Forward | 47.46 | 7.23 |
| Backward | 31.99 | 3.57 |
| Forward + backward | 76.91 | 10.11 |

Backward timing retains the forward autograd graph. These measurements exclude
native injection/metric bookkeeping and are not full-model TPS. Independent
roofline acceptance and scaling validation remain open; this is an opt-in
experimental implementation.
