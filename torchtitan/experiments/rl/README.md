# `num_actual_tokens` and Padding-Aware MoE

## How `vllm_rl`'s MetaShufflingMoE Handles Padding

The following describes the `vllm_rl` reference design, not a TorchTitan or
upstream vLLM API contract. In this design, the padding-aware implementation is
specifically `MetaShufflingMoE`. It uses `num_actual_tokens` to distinguish real
tokens from padding while keeping the fixed tensor shapes required by CUDA
graphs.

Suppose a batch contains 5 real tokens and is padded to 8 tokens for TP=4:

```text
num_tokens_unpadded = 5
num_tokens_padded   = 8
```

The flow is:

1. The GPU model runner writes `5` into a persistent, one-element GPU tensor.
   It preprocesses and forwards inputs with the padded length of 8, but passes
   the actual count through the forward context.
2. The vLLM `MOEWrapper` reads the count from the forward context only when its
   wrapped module is a `MetaShufflingMoE`. If its input is sequence-parallel, it
   converts the global count into one count per TP rank. With 8 padded tokens
   and TP=4, each rank receives 2 rows, so the real token counts are
   `[2, 2, 1, 0]`.
3. If MoE starts with a replicated input and internally splits it across TP
   ranks, it derives each local count from the rank's offset:

   ```python
   local_num_actual_tokens = (
       num_actual_tokens - offset
   ).clamp(min=0, max=local_size)
   ```

4. In eager mode, MoE slices away padded rows before routing and pads its
   output back to the original extent before the residual connection.
5. In static CUDA graph mode, MoE cannot change tensor shapes. Its norm,
   router, and dispatch operations use the count to ignore rows at indices
   greater than or equal to `num_actual_tokens`. The output therefore retains
   the padded extent.
6. After the model returns, the runner indexes the hidden states with the
   scheduled-token indices. Padded rows are not used to compute sampled
   logits.

The count is stored in a persistent GPU tensor instead of a Python integer so
its value can change between CUDA graph replays without changing a captured
tensor address or graph shape. This scheme assumes that real tokens form a
contiguous prefix and padding occupies the tail.

This forward-context value is also separate from an attention backend's
similarly named actual-token count. Attention metadata is constructed from the
actual and padded batch lengths independently; the forward-context tensor
carries padding awareness into model components such as MoE.

### Do Padding Tokens Participate in Computation?

The answer depends on the execution mode:

| Mode | Padding-token computation |
| --- | --- |
| Eager/dynamic | `_moe()` trims `hidden_states` to the real prefix before pooling, shared-expert computation, routing, dispatch, and routed-expert computation. It restores the padded output extent with zeros afterward. The normalization before `_moe()` receives the local actual-token count through its padding-aware interface. |
| Static CUDA graph | Tensor shapes remain padded. The router and dispatcher use `num_actual_tokens` so padded rows do not enter routed-expert computation. However, fixed-shape preprocessing may still operate over the allocated extent, the shared expert receives the full padded tensor, and batch-invariant router scoring may compute logits for padded rows. These results do not become valid routed-token outputs. |

Therefore, padded tokens do not consume routed-expert computation in either
mode. Eager mode also removes them from downstream shared-expert and routing
work after normalization. Static mode preserves fixed shapes, so it may still
perform non-routed computation on padded rows even though those rows are
excluded from routed dispatch and ignored by the final model output.

### Code Pointers for Eager and Static Modes

In eager mode, `MetaShufflingMoE._moe()` records the padded token extent, slices
`hidden_states` to the real prefix, and clears `num_actual_tokens` because every
downstream tensor is now compact:

[Eager trimming in `meta_shuffling_moe.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/meta_shuffling_moe.py:648)

```python
original_T = hidden_states.shape[0]
if not static_shape and num_actual_tokens is not None:
    hidden_states = hidden_states[:num_actual_tokens]
    num_actual_tokens = None
```

After routing and expert computation, it creates a zero-filled tensor with the
original extent and copies the compact result into its prefix. This restores
the shape required by the transformer residual connection:

[Eager output restoration in `meta_shuffling_moe.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/meta_shuffling_moe.py:765)

```python
if final_output.local_tensor.shape[0] < original_T:
    padded = torch.zeros(original_T, hidden_dim, ...)
    padded[: final_output.local_tensor.shape[0]] = final_output.local_tensor
```

In static CUDA graph mode, the trimming condition is false, so the padded token
extent remains fixed. The local actual-token count is first passed to the
padding-aware normalization:

[Local count and normalization in `meta_shuffling_moe.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/meta_shuffling_moe.py:549)

The count is then passed into the router:

[Router invocation in `meta_shuffling_moe.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/meta_shuffling_moe.py:678)

The router masks positions greater than or equal to `num_actual_tokens`, passes
the count to its score and index kernels, and returns `num_routed_tokens`:

[Padding-aware router in `router.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/moe/router.py:68)

That routed count is passed to the all-to-all dispatcher so only valid routed
rows participate in expert dispatch:

[Valid count passed to dispatch in `meta_shuffling_moe.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/meta_shuffling_moe.py:695)

Finally, the padding-aware scatter-add writes routed results into a fixed
`(T, D)` output while ignoring invalid routed positions:

[Fixed-shape scatter-add in `meta_shuffling_moe.py`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/inference/models/ffn/meta_shuffling_moe.py:1419)

Static mode does not guarantee that every operation avoids padded rows. The
shared-expert path still receives the full padded tensor, and batch-invariant
router scoring may compute logits for every row. The important guarantee is
that padded rows are excluded from routed-expert dispatch and that the MoE
output retains its fixed padded shape.

### Backend Distinction

The names `TokenShufflingMoE` and `MetaShufflingMoE` refer to different
implementations. The padding behavior described above is not shared by every
token-shuffling backend.

| MoE backend | Handling of vLLM runner padding |
| --- | --- |
| `MetaShufflingMoE` | `MOEWrapper` passes `num_actual_tokens`. Eager execution trims padded rows; static execution masks them from routing while retaining the padded tensor extent. |
| `TokenShufflingMoE` | Its `__call__()` signature contains `num_actual_tokens`, but the current implementation does not consume it. `MOEWrapper` also does not retrieve the forward-context count for this backend, so runner-added rows are still processed. |
| `TokenChoiceMoE` | Does not consume vLLM's `num_actual_tokens`. It has a separate `router_valid_mask` path that excludes padding from router statistics, but deliberately round-robin routes padded rows for balanced computation instead of skipping them. |

The relevant `MOEWrapper` logic is conceptually:

```python
num_actual_tokens = None
if isinstance(self.moe_module, MetaShufflingMoE):
    num_actual_tokens = get_num_actual_tokens_from_forward_context()

out = self.moe_module(
    hidden_states,
    use_static_shape=use_static_shape,
    num_actual_tokens=num_actual_tokens,
)
```

Therefore, saying that `vllm_rl` uses "token shuffling" to handle padding is
too broad. More precisely, `vllm_rl` uses the `MetaShufflingMoE` backend and its
padding-aware router and dispatch path. Selecting the older
`TokenShufflingMoE` backend does not provide the same behavior.

## TorchTitan Padding-Aware MoE

TorchTitan uses the same count-based public contract, while adapting the
implementation to its DTensor MoE boundary.

### Contract

`TorchTitanGPUModelRunner._pad_for_sequence_parallelism()` rounds an EP+TP
batch's token count up to a multiple of TP. The runner also stores the original
scheduled-token count in a persistent, one-element GPU tensor and passes it as
`num_actual_tokens` through the model and transformer blocks to MoE.

```python
def forward(
    self,
    x_BLD: torch.Tensor,
    *,
    num_actual_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

The count contains the number of real tokens in this DP rank's logical,
unsharded `B * L` order. `None` means that every row is real. This contract
assumes that real tokens form a contiguous prefix and vLLM padding is appended
at the tail.

For 5 actual tokens padded to 8 tokens:

```text
num_actual_tokens = [5]
is_valid_BL       = [[True, True, True, True, True, False, False, False]]
```

When dense SP is disabled, the MoE input is TP-replicated, so
`MoE.forward()` derives its input/shared-expert mask directly from the logical
token order without reading the GPU scalar into Python:

```python
B, L, _ = x_BLD.shape
token_indices_BL = torch.arange(B * L, device=x_BLD.device).reshape(B, L)
is_valid_BL = token_indices_BL < num_actual_tokens
```

When dense SP is enabled, the MoE input is already TP sequence-sharded. The
input/shared-expert mask therefore uses the TP rank and local sequence length.
The EP router and routed-expert path independently use the same TP metadata
because EP internally sequence-shards routed tokens even when dense SP is off.
For 5 real tokens padded to 8 over four TP ranks, the local masks are `[TT]`,
`[TT]`, `[TF]`, and `[FF]`.

### Does Padding Participate in TorchTitan MoE Computation?

Partially. Padding behavior differs by component and dispatcher mode:

| Component or mode | Padding-token computation |
| --- | --- |
| Normalization and router | These operate on the padded input extent. The validity mask is applied after the router, so padded rows may produce router logits. |
| Shared experts | Shared experts process the full padded tensor. Their padded output rows are masked to zero afterward. |
| Dynamic routed-expert dispatch | `RoutedExperts` removes padded rows from `x_TD`, top-k scores, and top-k IDs before dispatch. Padding therefore does not participate in routing communication or routed-expert computation. The compact output is scattered back into the original padded extent with zeros. |
| Static DeepEP CUDA graph dispatch | Tensor shapes remain fixed. Padded routing scores are zero, and DeepEP converts their expert IDs to `-1`, so they are not valid routed tokens and cannot affect expert counts or real-token results. Fixed-capacity kernels may still execute over allocated buffer slots, so this does not guarantee that every GPU operation associated with the padded extent is eliminated. |

Thus, TorchTitan excludes padding from meaningful routed-expert work, but it
still incurs router and shared-expert computation for padded rows. Padded rows
remain in the model tensor only to preserve the fixed extent required by vLLM
and CUDA graphs.

### Data Path

1. **Runner:** Allocate `num_actual_tokens` once in
   `TorchTitanGPUModelRunner`. Before each real forward, fill it from
   `scheduler_output.total_num_scheduled_tokens`. Initialize it to the runner's
   maximum token count so CUDA graph dummy runs treat every captured input row
   as valid.
2. **Runner-to-model handoff:** Add the persistent tensor to model kwargs from
   `_init_model_kwargs()`. Override `_preprocess()` only to update its value
   before calling the upstream implementation. This covers real forwards and
   dummy/capture forwards without copying upstream `execute_model()`.
3. **vLLM wrapper:** Add an explicit `num_actual_tokens` tensor argument to
   `VLLMModelWrapper.forward()` and keep the padded hidden-state extent through
   the model.
4. **Transformer blocks:** Add an optional keyword-only `num_actual_tokens`
   tensor argument to the decoder and transformer-block interfaces. Pass it only
   to MoE; dense FFNs may ignore it. Audit every supported block, including
   Qwen3, Qwen3.5, DeepSeek V3, GPT-OSS, and Llama 3.
5. **MoE boundary:** `MoE.forward()` derives a validity mask with the same
   placement as its input for shared-expert masking. It derives a second mask
   from the router output, whose placement represents the internal expert
   sequence sharding, and uses that mask for routing counts and scores.
6. **Routed-expert local region:** The replicated scalar passes through the
   keyword-only `local_map` boundary unchanged. `RoutedExperts.forward()` then
   rebuilds the rank-local mask from the local `x_BLD` shape and the sequence
   TP mesh coordinate. This avoids representing ragged local shards as a
   DTensor.
7. **Routing:** Force all top-k scores for padded rows to zero and exclude those
   rows from the routing map used for `num_local_tokens_per_expert_E` and
   `tokens_per_expert_E`. Padded tokens must not affect load-balancing state.
8. **Dispatch:** Give token dispatchers an explicit padding-aware contract.
   Each backend must either remove invalid rows or support a no-route sentinel;
   its expert counts, dispatched rows, and combine metadata must agree.
9. **Combine:** Return zeros at padded token positions while retaining the
   original padded `(B, L, D)` extent. This keeps residual connections and the
   final hidden-state shape stable. vLLM already selects only scheduled token
   positions before computing logits.

Passing a tensor model kwarg uses small runner hooks and does not depend on a
downstream modification to upstream vLLM. The installed vLLM `ForwardContext`
also has an `is_padding` field, but its stock runner does not populate it. If
upstream starts populating that field, the wrapper can consume it and remove
the custom model kwarg.

### Why Compaction Happens After DTensor Redistribution

TorchTitan does not copy MetaShufflingMoE's eager slice at the beginning of
`MoE.forward()`. At that point, `x_BLD` still has a global DTensor shape. TP
ranks can have different numbers of real local tokens, and dynamically
shortening the global DTensor cannot represent those ragged local shards.

Instead, keep the global padded extent until the existing `RoutedExperts`
local-map boundary:

```text
global num_actual_tokens
    -> token-axis DTensor redistribution of x_BLD
    -> local-map boundary keeps the scalar unchanged
    -> rebuild rank-local is_valid_T from local x_BLD
    -> compact or mask local routed-expert inputs
```

This preserves TorchTitan's global DTensor contract while giving each local
dispatcher validity information aligned with its local token rows.

### Dispatcher Behavior

Padding cannot be implemented by changing only the expert-count tensor. The
current standard dispatcher still permutes every top-k assignment, so excluding
a row from the counts without excluding it from the dispatched data would make
split sizes disagree and can fail or hang.

The implementation handles these paths as follows:

- **Local and standard all-to-all:** Compact `x_TD`, top-k scores, and top-k IDs
  to valid local rows before dispatch. After combine, scatter the compact output
  into a zero-initialized tensor with the original local token extent. These
  paths already use dynamic shapes and host-visible split sizes, so compaction
  fits their execution model.
- **DeepEP compact mode:** Uses the same compaction behavior.
- **DeepEP CUDA graph mode:** Preserve the static token extent. Set padded rows'
  scores to zero; the existing DeepEP wrapper converts zero-score selections to
  expert ID `-1`, meaning no expert selection. Dispatch and combine retain the
  fixed input extent, and the zero scores leave padded output rows zero.
- **HybridEP:** Uses the same pre-dispatch compaction and post-combine shape
  restoration as the other dynamic dispatchers.
- **MinimalAsyncEP:** Its existing contract excludes TP/SP and padding. The
  vLLM EP+TP runner path therefore does not select this backend.

The eager local path is conceptually:

```python
compact_x_TD = x_TD[is_valid_T]
compact_scores_TK = topk_scores_TK[is_valid_T]
compact_ids_TK = topk_expert_ids_TK[is_valid_T]

compact_out_TD = dispatch_compute_combine(
    compact_x_TD,
    compact_scores_TK,
    compact_ids_TK,
)

out_TD = torch.zeros_like(x_TD)
out_TD[is_valid_T] = compact_out_TD
```

The static DeepEP path keeps every tensor shape fixed and uses zero routing
scores as the no-route signal:

```python
topk_scores_TK = topk_scores_TK.masked_fill(
    ~is_valid_T[:, None],
    0,
)
```

The existing DeepEP wrapper converts those zero-score selections to expert ID
`-1`. This is TorchTitan's equivalent of MetaShufflingMoE's static
`num_actual_tokens` handling.

Shared experts require separate treatment. For correctness, mask their output
to zero at padded positions. Actually avoiding shared-expert computation for
padding would require either eager compaction or padding-aware dense kernels;
it should not block skipping padded rows in the routed-expert path.

### Attention Shape Prerequisite

The attention adapter already knows its actual token count. It writes attention
results only to the real prefix, zeros the padded tail, and returns the full
output tensor. This is required because `VLLMAttentionWrapper` expects the
runner-padded length; returning only `output[:num_actual_tokens]` would make an
eager prefill fail when it reshapes the attention result to the padded model
shape.

This is shape handling, not MoE padding awareness: attention should continue to
build KV metadata from real tokens, while MoE receives `num_actual_tokens`
through the model interface and derives its internal validity mask.

### Validation Plan

1. Unit-test runner padding and propagation: 5 scheduled tokens become 8 model
   rows while `num_actual_tokens` remains 5. Verify that dummy CUDA graph runs
   use the same persistent tensor.
2. Unit-test local MoE against an unpadded reference. Valid outputs must match,
   padded outputs must be zero, and expert/load-balancing counts must exclude
   padding.
3. Add distributed standard all-to-all cases for TP=4 with actual token counts
   1, 3, 7, and 8. The first two cover TP ranks with no real local tokens; 7
   covers uneven non-empty shards; 8 is the no-padding control.
4. Add a DeepEP CUDA graph replay test that captures one padded extent and
   replays it with different `num_actual_tokens` values. Confirm that padded
   rows are not dispatched and valid outputs match eager execution.
5. Test both `default` and `spmd_types` sharding backends. Confirm that the mask
   has the same token-axis layout as the MoE activation at the local-map
   boundary.
6. Run the vLLM parity regression with one active decode request under TP>1 and
   EP>1. Compare decode log probabilities with a second-pass prefill and cover
   eager, `FULL_DECODE_ONLY`, and the supported DeepEP CUDA graph mode.
7. Verify that a batch requiring no sequence padding remains bitwise identical
   before and after the change.

The distributed standard all-to-all, DeepEP graph replay, and end-to-end vLLM
parity cases remain required GPU validation before landing the change.
