# Llama4x and TorchTitan input and activation shape flow

This note first traces a language-model input from the Llama4x Sixlib data
pipeline through the Transformer, attention, TokenChoiceMoE, and output
projection. It then compares that flow with TorchTitan's current common decoder
and DeepSeek V3 MoE implementation. The main question is when the batch and
sequence dimensions remain separate and when they are folded into a single
token dimension.

## Notation

| Symbol | Meaning |
| --- | --- |
| `B` | Local batch size |
| `S` | Sequence length presented to this model invocation |
| `D` | Transformer residual/hidden dimension |
| `Hq` | Number of local query heads |
| `Hkv` | Number of local key/value heads |
| `Dh` | Attention head dimension |
| `E` | Number of routed experts visible to the router |
| `K` | Number of experts selected per token |
| `V` | Vocabulary size, or the local vocabulary shard when vocab parallelism is used |
| `TP` | Tensor-parallel world size |

`S` can be a context-parallel-local sequence length. Llama4x gathers the token
IDs when it needs global attention metadata, while the activation layout stays
consistent with the local model invocation.

## Shape overview

```text
Sixlib batch
  input IDs                                      [B, S]
       |
       v
token embedding                                 [B, S, D]
       |
       | fold B and S
       v
Transformer residual stream                     [B*S, D]
       |
       | sequence parallel, when enabled
       v
local residual stream                           [B*S/TP, D]
       |
       +---------------- Attention -------------------------------+
       | qkv projection gathers SP tokens as needed               |
       | [B*S, ...] -> [B, S, heads, Dh] -> [B*S, ...]            |
       +----------------------------------------------------------+
       |
       +---------------- Dense FFN / TokenChoiceMoE --------------+
       | remains a 2-D token matrix: [num_local_tokens, D]        |
       | router logits: [num_local_tokens, E]                     |
       | top-k IDs/scores: [num_local_tokens, K]                  |
       +----------------------------------------------------------+
       |
       v
final residual stream                            [B*S, D] or SP shard
       |
       v
output projection                                [B*S, V]
       |
       | restore input shape
       v
logits                                           [B, S, V]
```

The important conclusion is that Llama4x deliberately folds `B` and `S` for
the persistent Transformer residual stream. Attention temporarily reconstructs
the logical batch and sequence axes, while dense FFNs and MoE operate directly
on the flattened token matrix.

## 1. Sixlib produces a two-dimensional token batch

Sixlib converts its NumPy token arrays into `x_tensor` and `y_tensor`. Both have
shape `[B, S]`. Internal Sixlib padding (`PADDING_ID`, normally `-1`) is also
converted to model token ID `0` before tensor construction.

Source:

- [`_replace_placeholder_ids`: internal padding to token ID 0](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/input/sixlib/iterator.py:282)
- [Batch IDs and targets converted to tensors](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/input/sixlib/iterator.py:492)
- [Padding/loss mask construction](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/input/sixlib/iterator.py:545)

```python
ids = _replace_placeholder_ids(batch.ids, ...)
targets = _replace_placeholder_ids(batch.targets, ...)
x_tensor = torch.from_numpy(ids).to(torch.int32)       # [B, S]
y_tensor = torch.from_numpy(targets).to(torch.int32)   # [B, S]
mask = (x_tensor != 0) & (loss_weight_tensor > 0.0)    # [B, S]
```

## 2. The Transformer accepts `[B, S]` token IDs

`Transformer.forward()` explicitly expects a two-dimensional token tensor and
extracts the batch and sequence lengths:

```python
bs, N = tokens.shape[0:2]
```

It calls preprocessing with `output_shape=(bs * N,)`. This is the instruction
that later folds batch and sequence into one dimension.

Source:

- [`Transformer.forward` input contract](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2156)
- [Pass `output_shape=(bs * N,)` to preprocessing](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2253)

```python
h = self.pre_transformer_layers_processing(
    tokens=tokens,              # [B, S]
    output_shape=(bs * N,),
    ...,
)
```

## 3. Embedding starts as `[B, S, D]`, then folds to `[B*S, D]`

The token embedding preserves the two token-ID axes:

```python
h = self.tok_embeddings(tokens)  # [B, S, D]
```

Multimodal perception/audio injection also occurs before flattening, so these
paths can still address batch and sequence positions directly.

Llama4x then always folds the first two dimensions:

```python
# Flatten the first two dims ... It's required for sequence parallelism,
# but we just always do it.
h = h.view(*output_shape, self.args.residual_dim)  # [B*S, D]
```

Source:

- [Embedding and preprocessing](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2497)
- [Flatten batch and sequence](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2632)

The flattening is row-major. For flattened token index `r`:

```text
batch index    = r // S
sequence index = r % S
```

## 4. Sequence parallelism shards the folded token dimension

If sequence parallelism is enabled, Llama4x scatters the already-flattened
tensor across TP ranks:

```python
h = scatter_to_sequence_parallel_region(h)
```

Ignoring uneven splits and temporary padding, the local shape changes as
follows:

```text
SP off: [B*S, D]
SP on:  [B*S/TP, D] on each TP rank
```

This does not create a semantic third layout. The first axis is still the
row-major flattened token axis; each TP rank simply owns a shard of it.

Source:

- [Sequence-parallel scatter](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2648)

For evaluation inputs whose flattened token count is not divisible by TP,
Llama4x can append zero activation rows before the SP scatter. These rows are
removed again before restoring the output shape.

Source:

- [Compute TP padding count](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2243)
- [Append flattened activation padding](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2638)
- [Remove TP padding before output reshape](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:3330)

## 5. The Transformer blocks retain the 2-D residual layout

`transformer_layers_forward()` passes `h` through every Transformer block as
the `residual_stream`. The attention branch and FFN branch both receive this
2-D representation.

Source:

- [Transformer layer loop](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2428)
- [`TransformerBlock.forward`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:745)
- [Residual stream passed to the FFN/MoE](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:803)

Conceptually:

```text
residual_stream: [num_local_tokens, D]

num_local_tokens = B*S       when SP is off
num_local_tokens = B*S/TP    when SP is on, ignoring uneven splits
```

## 6. Attention temporarily restores `[B, S, ...]`

Attention receives the flattened residual stream. Its QKV projection gathers
the sequence-parallel token shards when required by the parallel linear layer.
After projection, attention infers the batch size from the total number of
projected token rows:

```python
batch_size = xqkv.shape[0] // slen
```

It then reconstructs explicit batch and sequence axes:

```python
xq = xq.view(batch_size, slen, self.n_local_heads, self.head_dim)
xk = xk.view(batch_size, slen, self.n_local_kv_heads, self.head_dim)
xv = xv.view(batch_size, slen, self.n_local_kv_heads, self.head_dim)
```

Therefore the attention kernel operates on logical shapes such as:

```text
Q: [B, S, Hq,  Dh]
K: [B, S, Hkv, Dh]
V: [B, S, Hkv, Dh]
```

After attention, Llama4x folds the output again:

```python
output = output.reshape(batch_size * slen, -1)
```

Source:

- [Attention receives the flattened residual stream](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/layers/attention_module.py:924)
- [Infer batch size and restore Q/K/V axes](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/layers/attention_module.py:982)
- [Flatten attention output again](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/layers/attention_module.py:1047)

## 7. TokenChoiceMoE remains flattened

Unlike attention, TokenChoiceMoE does not restore separate batch and sequence
axes. It treats the first dimension as a collection of token rows:

```text
hidden_states: [num_local_tokens, D]
router logits: [num_local_tokens, E]
top-k scores:  [num_local_tokens, K]
top-k IDs:     [num_local_tokens, K]
routing map:   [num_local_tokens, E]
```

The same hidden-state matrix goes to the shared expert and routed-expert path.
The dispatcher may reorder and exchange token rows for expert computation, but
the combined MoE result returns to the same flattened residual-stream layout.

Source:

- [`TokenChoiceMoE.default_forward`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/moe/moe_layers.py:774)
- [Router entry point](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/moe/router.py:389)
- [Router matrix multiplication](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/moe/router.py:535)
- [Router epilogue and top-k routing](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/moe/router.py:574)

### Padding mask alignment

The MoE padding-validity mask follows exactly the same folding and SP sharding
as the hidden states:

```python
router_valid_mask = (tokens != 0).reshape(-1)  # [B*S]
if self.sequence_parallel:
    router_valid_mask = scatter_to_sequence_parallel_region(router_valid_mask)
```

Thus row `r` of `router_valid_mask` describes row `r` of `hidden_states` on the
same rank. Padding rows are assigned to experts round-robin for compute balance
and removed from the routing map used for expert-load statistics.

Source:

- [Build and shard `router_valid_mask`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2364)
- [Round-robin padding assignments](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/moe/router.py:495)
- [Exclude padding from expert statistics](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/moe/router.py:653)

## 8. The output projection restores `[B, S, V]`

After the final Transformer layer, the output norm and vocabulary projection
still consume the flattened residual stream. Once TP-only activation padding
has been removed, Llama4x restores the original input axes:

```python
output = self.output(h, ...)           # [B*S, V]
output = output[:-tp_padding_num_tokens]  # when TP padding was added
output = output.view(*input_shape, -1) # [B, S, V]
```

Source:

- [`post_transformer_layers_processing`](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:3270)
- [Restore the input batch and sequence axes](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:3333)

## 9. TorchTitan keeps `[B, L, D]` through the model

TorchTitan's current common decoder and DeepSeek V3 model make a different
layout choice from Llama4x. The persistent Transformer activation keeps
separate batch and sequence dimensions:

```text
input IDs                  [B, L]
    |
    v
token embedding            [B, L, D]
    |
    +-- attention           [B, L, D] -> [B, L, heads, Dh] -> [B, L, D]
    |
    +-- router              [B, L, D] -> [B, L, E] -> [B, L, K]
    |
    +-- shared expert       [B, L, D] -> [B, L, D]
    |
    +-- routed experts      [B, L, D]
            | flatten only inside the local dispatch region
            v
                           [T=B*L, D] -> [R, D] -> [T, D]
            | restore local B and L
            v
                           [B, L, D]
    |
    v
LM head                    [B, L, V]
```

Here, `T` means the number of token rows entering a local dispatcher and `R`
means the number of routed rows assigned to the rank's local experts. Under
sequence or expert parallelism, `L` in the local dispatch region can be a local
sequence shard rather than the global sequence length.

### 9.1 Decoder, embedding, and LM head

The common decoder applies the embedding, every Transformer layer, the final
norm, and the LM head without reshaping the activation:

```python
h = self.tok_embeddings(tokens)  # [B, L, D]
for layer in self.layers.values():
    h = layer(h, attention_masks, positions)
h = self.norm(h)
output = self.lm_head(h)          # [B, L, V]
```

The vocab-parallel embedding also preserves all leading input dimensions.
`mask.unsqueeze(-1)` shows that it adds only the embedding dimension to the
input token shape.

Source:

- [TorchTitan common decoder forward](/home/jessicazhong/torchtitan/torchtitan/models/common/decoder.py:262)
- [Vocab-parallel embedding](/home/jessicazhong/torchtitan/torchtitan/models/common/embedding.py:47)

### 9.2 DeepSeek V3 attention keeps batch and sequence explicit

DeepSeek V3 attention directly reads a three-dimensional input:

```python
bsz, seqlen, _ = x.size()  # x: [B, L, D]
```

Its projected Q/K/V tensors add a head axis while preserving `B` and `L`:

```text
Q: [B, L, Hq,  Dh]
K: [B, L, Hkv, Dh]
V: [B, L, Hkv, Dh]
```

After the attention kernel, the head dimensions are merged back into
`[B,L,*]`, and the output projection returns `[B,L,D]`.

Source:

- [DeepSeek V3 attention input and Q shape](/home/jessicazhong/torchtitan/torchtitan/models/deepseek_v3/model.py:93)
- [Restore the attention output to `[B,L,*]`](/home/jessicazhong/torchtitan/torchtitan/models/deepseek_v3/model.py:143)
- [DeepSeek V3 block retains `[B,L,D]`](/home/jessicazhong/torchtitan/torchtitan/models/deepseek_v3/model.py:173)

### 9.3 TorchTitan's router also remains three-dimensional

The TorchTitan MoE wrapper and router explicitly use shape-suffixed names:

```text
x_BLD:                [B, L, D]
scores_BLE:           [B, L, E]
topk_scores_BLK:      [B, L, K]
topk_expert_ids_BLK:  [B, L, K]
routing_map_BLE:      [B, L, E]
```

Expert counts sum over both the batch and sequence axes:

```python
num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))
```

The routed and shared branches both receive `x_BLD`. Consequently, the MoE
boundary and the routed-plus-shared join remain `[B,L,D]`.

Source:

- [TorchTitan TokenChoiceTopKRouter shape contract](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:273)
- [TorchTitan MoE `[B,L,D]` contract and routing map](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:397)
- [Routed and shared expert branches](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:449)

### 9.4 Only routed-expert dispatch folds `B` and `L`

TorchTitan delays flattening until `RoutedExperts.forward()`, after routing has
already produced `[B,L,K]` expert IDs and scores:

```python
B, L, D = x_BLD.shape
K = topk_scores_BLK.size(-1)
T = B * L
x_TD = x_BLD.view(T, D)
topk_scores_TK = topk_scores_BLK.view(T, K)
topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
```

The dispatcher then changes `[T,D]` into expert-major `[R,D]`. After expert
computation and combine, TorchTitan restores the three-dimensional layout:

```python
return out_TD.view(B, -1, D)
```

This flattening is an implementation detail of dispatch/grouped GEMM rather
than the model-wide residual layout.

Source:

- [Flatten at the routed-expert boundary](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:125)
- [Token dispatcher `[T,D]` contract](/home/jessicazhong/torchtitan/torchtitan/models/common/token_dispatcher.py:71)
- [Restore `[B,L,D]` after combine](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:160)

### 9.5 vLLM inference uses a synthetic singleton batch dimension

The `vllm-torchtitan` inference path has an additional adapter boundary. vLLM
provides a packed one-dimensional token stream rather than a rectangular
training batch:

```text
vLLM input IDs: [Ttotal]
positions:      [Ttotal]
```

`VLLMModelWrapper` inserts a singleton batch dimension before calling the
TorchTitan embedding and layers:

```python
tokens_2d = input_ids.unsqueeze(0)  # [1, Ttotal]
h = self.model.tok_embeddings(tokens_2d)  # [1, Ttotal, D]
positions = positions.unsqueeze(0)        # [1, Ttotal]
```

Therefore the core TorchTitan model still sees a three-dimensional activation,
but its logical dimensions in this inference path are:

```text
B = 1
L = Ttotal, the total packed tokens scheduled by vLLM
h = [1, Ttotal, D]
```

The wrapper removes the synthetic batch axis after the final norm and returns
`[Ttotal,D]` to vLLM.

The vLLM attention adapter performs another local layout conversion. It
receives TorchTitan's `[B,L,N,H]`, reshapes it to vLLM's `[T,N,H]`, runs paged
attention, and restores `[B,L,N,H]` before returning to the TorchTitan layer:

```text
[1,Ttotal,N,H] -> [Ttotal,N,H] -> [1,Ttotal,N,H]
```

Source:

- [vLLM flattened input contract and singleton batch conversion](/home/jessicazhong/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py:349)
- [Return flattened hidden states to vLLM](/home/jessicazhong/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py:390)
- [vLLM attention layout adapter](/home/jessicazhong/torchtitan/torchtitan/experiments/rl/models/attention.py:257)

## 10. Logical shape versus physical sharding in TorchTitan

TorchTitan preserves the logical DTensor shape `[B,L,D]` even when the
underlying local tensor is sequence-sharded. The placement, rather than a
model-wide flatten, describes ownership:

| Configuration | Logical activation | TP placement at block boundaries | Typical local shape |
| --- | --- | --- | --- |
| SP off | `[B,L,D]` | replicated/identity | `[B,L,D]` |
| SP on | `[B,L,D]` | `Shard(1)` | `[B,L/TP,D]` |
| SP off, EP on, inside MoE router | `[B,L,D]` | converted to `Shard(1)` internally | `[B,L/TP,D]` |

The root decoder sharding configuration states that SP shards activation
dimension 1, the explicit sequence dimension. DeepSeek attention gathers this
sequence shard to replicated input for its projections and returns to the
configured SP layout through its rowwise output projection.

Source:

- [Decoder SP placement is `Shard(1)`](/home/jessicazhong/torchtitan/torchtitan/models/common/decoder_sharding.py:253)
- [DeepSeek V3 attention input redistribution](/home/jessicazhong/torchtitan/torchtitan/models/deepseek_v3/sharding.py:83)
- [MoE router sequence sharding when EP is enabled](/home/jessicazhong/torchtitan/torchtitan/models/common/moe_sharding.py:89)
- [MoE boundary input/output placement](/home/jessicazhong/torchtitan/torchtitan/models/common/moe_sharding.py:269)

This is the most important terminology distinction:

```text
Llama4x SP local residual:      [B*S/TP, D]  (batch and sequence folded)
TorchTitan SP local residual:   [B, L/TP, D] (sequence axis remains explicit)
```

Both contain the same number of local token rows, but they expose different
rank and axis semantics to model code.

## 11. Llama4x versus TorchTitan summary

| Stage | Llama4x | TorchTitan DeepSeek V3 |
| --- | --- | --- |
| Model input IDs | `[B,S]` | `[B,L]` |
| Embedding output | Initially `[B,S,D]` | `[B,L,D]` |
| Persistent residual stream | `[B*S,D]` | `[B,L,D]` |
| SP local activation | `[B*S/TP,D]` | Logical `[B,L,D]`, local `[B,L/TP,D]` |
| Attention input | Flattened `[B*S,D]` | Explicit `[B,L,D]` |
| Attention kernel | Temporarily reconstructs `[B,S,H,Dh]` | Naturally forms `[B,L,H,Dh]` |
| Router input | `[num_local_tokens,D]` | `[B,L,D]` DTensor, possibly `Shard(1)` |
| Router output | 2-D token/expert maps | `[B,L,E]` and `[B,L,K]` |
| Routed dispatch | Already flat, then expert-major | Flattens locally to `[T,D]`, then expert-major `[R,D]` |
| Shared expert | Flattened token matrix | `[B,L,D]` |
| MoE output | Flattened residual layout | Restored `[B,L,D]` |
| Final logits | Reshaped to `[B,S,V]` | Already `[B,L,V]` |

For the `vllm-torchtitan` inference path, interpret the TorchTitan column with
`B=1` and `L=Ttotal`. vLLM owns the real request boundaries and paged-attention
metadata; the TorchTitan model receives the packed tokens as one synthetic
sequence-shaped axis.

The two models are mathematically compatible: a token-wise linear layer or MoE
router produces the same values whether it sees `[B,L,D]` or `[B*L,D]`, as long
as flattening preserves row order. The practical differences are where shape
metadata lives, which dimension SP shards, and where masks or routing metadata
must be transformed.

## 12. Padding behavior differs at the MoE boundary

Llama4x constructs a flattened `router_valid_mask` from token IDs, shards it in
the same way as the flattened residual stream, round-robin routes padding for
compute balance, and excludes padding from expert-load statistics.

TorchTitan currently has no equivalent `router_valid_mask` in the common MoE.
For MoE-internal sequence divisibility, it can pad the explicit `L` activation
axis with zero hidden vectors before routing:

```python
seq_pad = (-original_L) % self.expert_sequence_parallel_size
x_BLD = F.pad(x_BLD, (0, 0, 0, seq_pad))
```

Those added rows pass through the ordinary router and are included in
`routing_map_BLE.sum(dim=(0,1))`; the resulting output rows are trimmed only
after the routed and shared branches are combined. Therefore TorchTitan's
current MoE-internal activation padding is not recognized as padding by the
router for load-balancing purposes.

Source:

- [TorchTitan MoE sequence padding](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:412)
- [TorchTitan routing counts include all rows](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:430)
- [Trim padded sequence rows from MoE output](/home/jessicazhong/torchtitan/torchtitan/models/common/moe.py:472)
- [Llama4x router-valid mask](/data/users/jessicazhong/fbsource/genai/llama4x/llama4x/model/transformer.py:2364)

## Practical implications

1. Both routers make independent decisions per token. Retaining `[B,L]` in
   TorchTitan does not make routing sequence-level; it only retains shape
   metadata longer.
2. Llama4x masks aligned with MoE activations must be flattened in row-major
   order and SP-sharded like `[B*S,D]`. TorchTitan masks can remain `[B,L]`
   until the local routed-expert dispatch boundary.
3. Batch boundaries matter to attention in both systems. Llama4x reconstructs
   them inside attention; TorchTitan never removes them from the model-level
   activation.
4. TorchTitan's explicit `L` axis makes its SP placement directly expressible
   as `Shard(1)`. Llama4x SP shards the combined token axis.
5. Any change to flattening, sharding, or token order must apply the identical
   transformation to routing scores, expert IDs, padding masks, and other
   token metadata.
