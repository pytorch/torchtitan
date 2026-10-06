# GraphTrainer SPMD and PP reproduction stack

This candidate contains 10 functional commits followed by one `[not-for-land]`
commit with the
[performance run instructions](graph_trainer_perf_runs_instructions.md),
recipes, launch scripts, and numerical comparison helpers. SPMD is the
single-stage workload. The 256-GPU PP runs use PP2/VPP8 or PP4/VPP4, with
16 logical stages in either case.

## Functional commit order

1. Keep full-step CUDA graph capture client-owned (#5023).
2. Accumulate PP gradients in backward graphs, including WGrad fusion (#5208).
3. Preserve asynchronous pipeline FSDP unshard lookahead (#5175).
4. Canonically order communication-only reductions.
5. Coalesce repeated FSDP gradient reductions (#5121).
6. Defer FSDP persistent-gradient casts until final reduction, preserving gradient provenance through cleanup (#5118).
7. Preserve eager PP loss order for numerics match.
8. Support MTP in PP for DSv3, including shared-parameter lifecycle hooks (`[mtp-only]`).
9. Preserve nonzero FSDP shard order for quantized weights (eager correctness).
10. Finalize deferred FSDP reductions explicitly (eager correctness).

The three `[perf]` commits follow client-owned CUDA graph capture. Published-PR
commits come first; the unnumbered communication-ordering commit follows them.
The MTP-only feature follows the general GraphTrainer changes, immediately
before the two `[eager]` correctness commits.

The MTP+PP showcase commit retains runtime enablement, two focused CPU numerics
regressions, and small updates to existing test fixtures. Extended ownership
and checkpoint test matrices, integration recipes, and the expanded MTP guide
live in the reproduction-only commit. Their relocation does not change the
runtime implementation or remove verification coverage from the complete stack.

## MTP-only scope

The only MTP-only functional commit in this candidate is
`[graph_trainer][mtp-only] Support MTP in PP for DSv3`. It enables the
combination of MTP and multiple pipeline stages. SPMD MTP support is already
in the base; no-MTP SPMD and PP workloads do not need this feature.

The commit also contains generic pipeline result/lifecycle plumbing and
callsite migrations needed by that feature. Non-MTP pipelines use no-op
lifecycle hooks and count every parameter in the gradient norm as before.
The label describes the feature's need in these workloads, not a requirement
that every helper be model-specific. If omitting the commit from a no-MTP
stack, keep the pipeline API and its consumers consistent and omit its
MTP-specific reproduction tests and recipes as well.

The other functional changes are not MTP-only:

| Change | Non-MTP application |
| --- | --- |
| Client-owned CUDA graph capture (#5023) | Full-step capture for SPMD and PP. |
| Repeated FSDP gradient reduction coalescing (#5121) | Matching contributions to shared/tied parameters, independently of MTP. |
| In-graph PP accumulation and WGrad fusion (#5208) | Multiple microbatches in ordinary PP schedules. |
| Deferred persistent-gradient casts and provenance-preserving cleanup (#5118) | FSDP1 gradient accumulation and matching its reduction epilogue. |
| PP asynchronous unshard lookahead (#5175) | Communication/compute overlap in ordinary PP schedules. |
| Communication-only reduction ordering | Extracted FSDP reductions and nested process groups. |
| Eager PP reporting-loss order | Reporting losses from multiple PP microbatches. |
| Quantized FSDP shard order | Nonzero-dimension sharding of quantized weights. |
| Eager deferred FSDP reduction finalization | Non-PP accumulation with deferred synchronization. |

These changes may be no-ops when their graph/configuration pattern is absent;
that does not make them MTP-specific. The reproduction-only commit mixes
MTP-specific recipes/tests with general launch, profiling, and comparison
tools, so it is not labelled MTP-only as a whole.

Upstream MTP parent bucketing, scoped FSDP buckets, symmetric-memory validation,
and WGrad add-chain implementation (#5119) are reused. The newer PP accumulation
PR (#5208) includes the PP WGrad wiring, so that wiring is not a separate commit.
No additional precompile, CODA, or full/regional Inductor enablement fixes are
included. The client-owned capture PR updates all its consumers, including
existing precompile and Inductor pass selection.

## Published PR patch identity

Four PR-labelled commits preserve the added and removed lines and file-mode
changes of their pinned source revisions. Only unchanged context and hunk
positions may differ after rebasing. The local #5118 revision deliberately
extends its published patch with the provenance-preserving correctness fix
needed on this base. The remote PR has not been updated by this local squash.
The revisions below are cached PR snapshots, not a claim that the remote PR
heads have been checked more recently.

| PR | Pinned patch source | Local patch status |
| --- | --- | --- |
| [#5121](https://github.com/pytorch/torchtitan/pull/5121) | [ff2aa7a55](https://github.com/pytorch/torchtitan/commit/ff2aa7a550b1c5b8c3ded412f824f480872667da) | Exact |
| [#5208](https://github.com/pytorch/torchtitan/pull/5208) | [d14347ba7](https://github.com/pytorch/torchtitan/commit/d14347ba724492654073ea4d48331b4035021e9c) | Exact |
| [#5118](https://github.com/pytorch/torchtitan/pull/5118) | [381aa47ca](https://github.com/pytorch/torchtitan/commit/381aa47cadea49643a24e5303a0edce6e9418c68) | Extended with provenance correctness |
| [#5023](https://github.com/pytorch/torchtitan/pull/5023) | [52eb67b69](https://github.com/pytorch/torchtitan/commit/52eb67b69476c352345b9563acd3f8386ebc9e97) | Exact |
| [#5175](https://github.com/pytorch/torchtitan/pull/5175) | [c6e0ee4f7](https://github.com/pytorch/torchtitan/commit/c6e0ee4f7d3d25c848ae61150bc18a6cf99db0cf) | Exact |

Before accepting a newer PR revision, fetch its current head, verify the patch
again, and repeat the numerical and performance gates. Record any local patch
extension explicitly and update its PR before claiming exact patch identity.

The local #5118 commit now includes its provenance compatibility change. The upstream
SimpleFSDP refactor left the terminal gradient cast outside its annotation scope
and added identity views after it. Retain the cast's FSDP annotations and carry
parameter-gradient identities through view cleanup. Without these fixes,
the #5118 root/nested SimpleFSDP tests fail on the newer base. There is no
separate provenance commit in this candidate.

The setup commit includes a temporary compatibility revert for PyTorch runtimes
without FSDP `MixedPrecisionPolicy.param_dtype_override_fn`. It is not a
GraphTrainer fix and must not land. Remove that revert when validating a runtime
that implements the API.

Eager reproduction and numerical-comparison recipes use the existing in-place
DistMoE WGrad policy. Gradient output dtype comes from the live parameter-owned
destination; GraphTrainer retains its existing explicit functional-output dtype
argument. No additional functional-WGrad dtype API or engine wiring is included.
The reproduction commit retains the GraphTrainer test-fixture state needed by
client-owned capture and pipeline cleanup. Changing the local SPMD numerical
control from functional to in-place WGrad requires a fresh GPU numerical gate;
previous functional-control results do not establish parity for this candidate.

## Validation scope

Historical performance does not establish acceptance of a rewritten stack.
Validate the exact candidate commit and runtime using the four-GPU gates, then
matched 256-GPU eager/GraphTrainer runs. Record source and dependency revisions,
all full-precision loss/grad-norm samples, clean throughput intervals, memory
measurements, and profiler artifacts. The run guide describes dependency-aware
ablation and separates numerical acceptance from performance acceptance.
