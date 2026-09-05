# MoE token dispatcher runbook

This runbook covers the token dispatchers in
`torchtitan/models/common/token_dispatcher.py`: what each one does, when to
use it, how router scores flow through them, and how to diagnose common
failures.

## Where the dispatcher sits

Every MoE layer builds a `RoutedExperts` module that owns two siblings:

```text
MoE.forward
  -> router                (topk_scores_TK, topk_expert_ids_TK, scores_TE)
  -> RoutedExperts.forward
       -> token_dispatcher.dispatch(x_TD, topk_scores_TK, topk_expert_ids_TK, counts_E)
       -> inner_experts(routed_input_RD, counts_e, routed_scores_R=...)
       -> token_dispatcher.combine(routed_output_RD, metadata, x_TD)
```

Shape suffixes (scoped to this file): `T` = local tokens, `D` = model dim,
`K` = top-k, `E` = global experts, `e` = local experts, `N = T*K` routed
slots in expert-sorted order, `R` = routed rows owned by the local experts
(`R != N` when EP > 1, or when token groups are padded).

`dispatch` returns `(routed_input_RD, num_tokens_per_local_expert_e,
metadata)`; `combine` consumes the same `metadata`. Treat the metadata as an
opaque token: never construct or mutate it outside the dispatcher pair.

## Where router scores are applied

Every dispatcher carries an `absorb_router_scores` config flag (default
`False`) that selects one of two equivalent placements for the routing
scores:

- **Post-combine scoring (default).** The dispatcher keeps
  `topk_scores_experts_sorted_N` in the metadata. `combine` upcasts the
  `(R, D)` expert outputs to float32, multiplies by the scores, casts back,
  and scatter-adds into the `(T, D)` output. Autograd has to save the float32
  copy of the post-W2 outputs.

- **Pre-W2 absorption (`absorb_router_scores=True`).** `dispatch` instead
  carries `routed_scores_R` in the metadata. `RoutedExperts.forward` passes
  it to the expert module, which multiplies its `(R, F)` hidden activations
  by the scores (in float32) before the W2 grouped GEMM. `combine` then
  scatter-adds the outputs as-is. The float32 upcast moves from `(R, D)` to
  the smaller `(R, F)`, and the post-W2 output no longer needs a saved
  float32 copy for the score multiply.

Both placements compute `score * expert(x)`; they differ only in bfloat16
rounding order, so losses match to rounding but are not bit-identical. Use
`--debug.seed=42 --debug.deterministic` comparisons when validating a swap.

The two metadata fields are mutually exclusive: exactly one of
`topk_scores_experts_sorted_N` / `routed_scores_R` is set. The
`LocalDispatchMetadata` dataclass documents this contract; `RoutedExperts`
reads `metadata.routed_scores_R` and falls back to `None` for dispatchers
whose metadata predates the field.

### Expert module contract

`GroupedExperts.forward(x_RD, num_tokens_per_expert_E, *,
routed_scores_R=None)` applies the scores when they are not `None`.
Implementations in tree:

| Expert module | Absorption | Notes |
|---|---|---|
| `GroupedExperts` (common) | yes | SwiGLU activations scaled before W2 |
| `KimiGroupedExperts` | yes | SiTU-glu activations scaled before W2 |
| `FusedGroupedExperts` (fused_swiglu override) | yes | Same pattern after fused silu_and_mul |
| `GptOssGroupedExperts` | yes | Also scales the interleaved post-W2 bias rows so the output equals `score * (h @ W2 + b2)` |
| Float8 / MXFP8 quantized subclasses | yes | Inherit `forward`; only `_grouped_mm` is swapped. Scaled activations are quantized dynamically, so per-block relative error is unchanged |

## Dispatcher support matrix

| Dispatcher | EP | Padding | `absorb_router_scores` | Scores applied |
|---|---|---|---|---|
| `LocalTokenDispatcher` | 1 | no | yes | combine (default) or pre-W2 |
| `AllToAllTokenDispatcher` | any | no | yes | combine or pre-W2 |
| `TorchAOTokenDispatcher` | any | yes (`pad_multiple`) | yes | combine or pre-W2, pad rows carry a zero sentinel score |
| `DeepEPTokenDispatcher` | >1 | no | yes (mock-validated; see caveat) | backend combine (default) or pre-W2 via popped tracked scores |
| `HybridEPTokenDispatcher` | >1 | yes | yes (mock-validated; see caveat) | backend combine (default) or pre-W2 via popped tracked scores |
| `MinimalAsyncEPTokenDispatcher` | >1 | no | no (raises / warns) | fused inside the combine kernel |

## Dispatchers

### LocalTokenDispatcher (EP=1)

Reorders tokens into expert-sorted order with a stable argsort; no
communication. Selected automatically when `comm_backend="standard"` and
`ep_mesh` is `None` (the `AllToAllTokenDispatcher` falls back to this path).
Metadata: `LocalDispatchMetadata` with `N == R`.

### AllToAllTokenDispatcher (EP>1)

The default `"standard"` backend. Lifecycle:

1. `_local_reorder`: stable argsort of `topk_expert_ids_TK` into
   expert-sorted `(N, D)` order.
2. `_token_count_exchange`: all-to-all of per-expert token counts, then a
   device-to-host sync to materialize `input_splits` / `output_splits`.
3. `_dispatch_token_exchange`: all-to-all of the routed tokens. With
   `absorb_router_scores=True` the scores ride along as an extra `(R, 1)`
   all-to-all (the only additional communication absorption costs).
4. `_permute`: rank-major to expert-major reorder.
5. `combine`: `_unpermute`, reverse all-to-all, score multiply (unless
   absorbed), `deterministic_scatter_add` back to `(T, D)`.

Under the `spmd_types` backend the exchanges go through `spmd.all_to_all`
and the routed scores get `spmd.reinterpret_mesh` treatment like the tokens
(see the `is_type_checking` blocks in `dispatch`).

Gotchas:

- The count exchange forces a D2H sync; this is inherent to variable-size
  all-to-all, not a bug.
- `EP=1` silently takes the local path (no all-to-all), which is what makes
  single-GPU numerics debugging possible.

### TorchAOTokenDispatcher (quantized grouped GEMMs)

Subclass of `AllToAllTokenDispatcher` that replaces `_permute` with torchao's
`permute_and_pad`: each expert's token group is padded to a multiple of
`pad_multiple` (16 for FP8, 32 for MXFP8) as required by quantized grouped
GEMM kernels. Installed automatically by the MoE quantization converters via
`swap_token_dispatcher` (`torchtitan/components/quantization/utils.py`),
which also propagates `absorb_router_scores`.

Padding mechanics that matter for scores:

- `permute_and_pad` appends one zero **sentinel row** to the rank-major
  input; `permuted_indices` maps pad positions to that row using index `-1`
  (Python negative indexing wraps to the appended row).
- `_gather_routed_scores` therefore concatenates a single zero score before
  indexing, so every pad row carries score 0 and contributes nothing to the
  experts or the combine. The base class override is a plain gather because
  the base `_permute` never pads.
- Rows past the last expert group (up to the padded buffer length) are dead:
  they may contain garbage from `_grouped_mm`, but they only ever map to the
  sentinel row, which `_unpermute` strips.

Works with EP>1 (all-to-all + padded permute) and EP=1 (padded permute only).

Gotchas:

- Requires a triton-capable device: `permute_and_pad` launches a triton
  index kernel. On CPU-only environments it raises (use the base dispatcher
  for numerics debugging instead).
- `metadata.input_splits` / `output_splits` are empty lists in the EP=1
  path; there is no all-to-all to reverse.

### DeepEPTokenDispatcher

DeepEP v2 `ElasticBuffer` dispatch/combine for H100/NVLink Switch. Requires
`expert_parallel_degree > 1`, `num_max_tokens_per_rank` (validated against
the training shape by `update_ep_token_dispatcher_config`), and the DeepEP
installation. `topk_scores_TK` is passed **into** `dispatch_tokens`; the
backend exchanges the scores along with the tokens and tracks them aligned
with the dispatched rows (`state.permuted_scores` on the compact path,
`state.recv_scores` on the expand path).

Supports `absorb_router_scores` with no extra communication: dispatch pops
the tracked scores into the metadata via
`deepep.extract_routed_scores(state)`, leaving the state score-free so
`combine_tokens` skips its score multiply; the experts apply the scores
before W2 instead. `cudagraphable=True` selects the static expand layout
(inference only; training falls back to compact) and works with absorption
the same way.

Caveat (prototype status): the wiring is validated against mocked transport
only. Still to validate on real H100/NVLink hardware:

- Compact-path loss parity (absorbed vs post-combine) and score gradients
  through the dispatch custom op's autograd on a multi-rank EP run.
- SAC / activation-checkpointing interactions: the score multiply moves from
  the combine region (post-custom-op) into the expert region, which changes
  what the DeepEP custom ops save versus recompute.
- Expand (cudagraphable) layout under CUDA graph capture with absorption.

### HybridEPTokenDispatcher

HybridEP kernels for GB200/NVLink72. Same EP requirement; `pad_multiple`
pads token groups for the fused-permute MXFP8 path. `torch.ops.hybridep.dispatch`
returns `permuted_scores` covering every dispatched row, padding included.
Supports `absorb_router_scores` the same way as DeepEP
(`hybridep.extract_routed_scores`); pad rows keep whatever score the
dispatch kernel assigned them, and their outputs are discarded by the
combine either way. Same prototype caveat: mock-validated wiring, not yet
run on GB200.

`non_blocking_capacity_factor` in `(0, 1]` enables CPU-free dispatch with a
reduced fused-permute capacity -- tokens past the capacity are silently
dropped (an overflow flag is set on device); keep it at 1.0 unless load
balancing keeps the distribution uniform.

### MinimalAsyncEPTokenDispatcher

Fixed-size symmetric-memory buffers for constrained DP >= EP topologies.
Requires an EP mesh and fully-specified `hidden_dim`,
`num_max_tokens_per_rank`, `dtype`, `device` (filled from the runtime config
by `maybe_update_minimal_async_ep_config`). Scores never travel to the
expert ranks: the dispatch op takes no scores, and the **fused** combine
kernel applies slot-ordered scores on the origin rank (with a dedicated
`topk_scores_grad_kernel` for their gradient). Absorption would require
kernel-level changes to carry `(R,)` scores through the symmetric-memory
buffer, so `absorb_router_scores=True` is rejected in
`make_token_dispatcher_config` and warns when the dispatcher is constructed
directly.

## Config plumbing

`make_token_dispatcher_config` (`torchtitan/models/common/config_utils.py`)
maps `comm_backend` to a Config subclass. It **raises** `ValueError` if
`absorb_router_scores=True` is requested with `minimal_async_ep`, because
that backend applies scores inside its fused combine kernel.

`make_routed_experts_config` builds the inner experts plus dispatcher. Its
`absorb_router_scores` default is `None` = automatic: enabled for
`comm_backend="standard"`, disabled otherwise. DeepEP/HybridEP support
absorption but stay post-combine by default until validated on their target
hardware; pass `True` explicitly to opt in. Current defaults per model:

- qwen3 / qwen3.5 / deepseek_v3 / transformers modeling backend: automatic
  (absorbed on the standard backend).
- gpt_oss and kimi_k3: build their dispatcher configs directly and stay on
  post-combine scoring. Their expert modules support absorption, so flipping
  the flag is a numerics-only change if you want to experiment.
- MoE quantization (FP8 / MXFP8) swaps the dispatcher via
  `swap_token_dispatcher`, which propagates `absorb_router_scores` into the
  `TorchAOTokenDispatcher.Config` (or the HybridEP config).

To force a mode from a config registry function:

```python
routed_experts=make_routed_experts_config(
    ...,
    comm_backend="standard",   # or "deepep" / "hybridep"
    absorb_router_scores=False,  # or True
),
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: absorb_router_scores=True is not supported with comm_backend='minimal_async_ep'` | MinimalAsyncEP applies scores inside its fused combine kernel, and its dispatch op carries no scores to the expert ranks. Use another backend or drop the flag. |
| `IndexError` / out-of-bounds gather on routed scores under TorchAO | A `_permute` override that pads must also override `_gather_routed_scores` to append the zero sentinel score (see `TorchAOTokenDispatcher._gather_routed_scores`). |
| Warning `MinimalAsyncEPTokenDispatcher applies router scores inside its combine; absorb_router_scores=True is ignored` | The flag was set on a MinimalAsyncEP dispatcher. Results stay correct; remove the flag to silence it. |
| DeepEP/HybridEP absorbed outputs are unscored | The backend dispatch returned no tracked scores, so `extract_routed_scores` returned `None` and the experts skipped the multiply. Check the backend installation/version actually returns `recv_scores` / `permuted_scores`. |
| `TypeError: forward() got an unexpected keyword argument 'routed_scores_R'` | A custom `GroupedExperts` subclass overrode `forward` with the old positional signature. Add the keyword-only `routed_scores_R: torch.Tensor | None = None` parameter and apply it before W2 (see the expert module contract above). |
| Loss differs slightly after enabling absorption | Expected: the score multiply moves before W2, changing bfloat16 rounding order. Compare with tolerance (`scripts/loss_compare.py`), not bit-equality. A large divergence means a misaligned `routed_scores_R` -- check that every dispatcher `_permute` override keeps scores aligned (identity-expert round trip is a good scratch test). |
| `permute_and_pad` fails with a triton error on CPU | TorchAO's index kernel needs a triton-capable device. Use the base dispatcher for CPU numerics work. |
| Silent score drop (experts receive `routed_scores_R=None` unexpectedly) | The dispatcher config was rebuilt without propagating `absorb_router_scores` (e.g. a converter constructing a new dispatcher Config). `swap_token_dispatcher` propagates it; new swap sites must do the same. |
