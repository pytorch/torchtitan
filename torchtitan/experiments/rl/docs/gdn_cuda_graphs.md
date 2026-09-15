# GDN CUDA graph execution

TorchTitan's Attention Gym GDN path supports breakable PIECEWISE CUDA graphs
without an eager boundary at every GDN layer. This requires the TorchTitan V1
worker (`TORCHTITAN_WORKER_CLS`) and `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, which the
RL generator sets for `FULL_AND_PIECEWISE` execution. TorchInductor compilation
is not required.

## Metadata and capture contract

`TorchTitanGDNAttentionBackend` is selected only by TorchTitan GDN layers; it
does not replace native vLLM backend registrations. Its builder retains native
metadata construction and FULL-decode support, and owns an additional set of
fixed-address sequence offsets, cache slots, and initial-state flags.

`TorchTitanGDNGraphWrapper` prepares those buffers once per attention group,
before calling vLLM's capture/replay wrapper. Each token bucket uses fixed-shape
prefix views. Unused slots are null, and an extra null-slot sequence covers the
physical token padding, so padded outputs are zero and unused cache slots are
untouched. Freshly assigned slots ignore old state; continuation chunks reload
and advance their existing state.

Metadata-free dummy warmups/captures receive inert packed GDN metadata. Other
attention layers still receive `None`, preserving their existing eager-break
behavior. At least one CUDA graph warmup is required to compile kernel variants
before capture. Pre-cache profiling without initialized builders is unchanged.

The captured GDN path uses packed paged convolution and chunk GDN for both
prefill-only and mixed prefill/decode batches. Batch-invariant execution instead
keeps the existing recurrent scan with autotuning disabled. Captured Python
control flow never depends on the live number of requests or prefill tokens.

## Preserved paths and limitations

- FULL decode keeps native metadata and the existing fused decode kernels.
- Real NONE/eager execution keeps the original implementation.
- A pure-decode batch dispatched to PIECEWISE runs the original model eagerly.
  Packed prefill and fused decode have different rounding; silently substituting
  one for the other would change outputs/state. Prefer `FULL_AND_PIECEWISE` to
  retain captured decode.
- The new path currently requires data-parallel size 1 and no microbatching/DBO.
  Other configurations retain the existing eager GDN boundaries. Tensor
  parallelism is not disabled by this restriction.
- Speculative decoding remains unsupported by the GDN adapter.

Do not remove the legacy eager decorator globally: replay does not rerun Python
metadata lookups, and metadata-free capture would record a no-op.

## Validation

A real Qwen3.5-0.8B TP1/DP1 run on GB200 matched legacy eager execution exactly
for 96 generated token IDs, 480 top-5 logprob values/ranks, and cumulative
logprobs. The workload used a 128-token scheduler budget, prompt lengths
127/129/257/63, eight output tokens, and repeated/reordered request batches.
It exercised graph-memory profiling, cache reinitialization, PIECEWISE prefill,
and padded FULL decode. Prefill had seven graph segments with only the six
standard-attention breaks; decode had one graph and no breaks.

That run used PyTorch `2.15.0.dev20260830+cu130`, the August 31 vLLM source build,
Attention Gym `26222eb`, and FA4 `383cbcba` (`4.0.0b31.dev5`). FA4 b19 cannot run
Qwen3.5's paged head-dimension-256 attention. In an older vLLM environment, also
check optional TileLang's TVM-FFI requirements: TileLang 0.1.12 requires FFI
<=0.1.11, conflicting with current FA4. Its JIT-monitor import can abort even
when the model does not use TileLang kernels. Use compatible dependency builds
or omit the unused TileLang package in a Qwen-only environment.

CPU metadata check (run in the RL environment):

```sh
pytest torchtitan/experiments/rl/tests/test_gdn_graph_metadata.py -v
```

GPU capture/replay (requires vLLM and Attention Gym's linear dependencies):

```sh
pytest torchtitan/experiments/rl/tests/test_gdn_cudagraph.py -v
```

The GPU tests compare outputs and full backing cache storage bitwise against
legacy eager execution. They cover changed sequence boundaries and request
counts, fresh/resumed slots, slot reuse, padded tails, mixed batches, the pure
decode fallback, and both chunk and batch-invariant recurrent GDN. The integration
test uses the real metadata builder and outer capture/replay wrapper, including
metadata-free capture; successful kernel capture alone is not sufficient.
