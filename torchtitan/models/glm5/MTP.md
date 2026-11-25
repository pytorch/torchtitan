# GLM-5 multi-token prediction

GLM reuses DeepSeek-V3's MTP construction, token shifting, auxiliary loss,
shared output head and distributed lifecycle. See the
[shared MTP documentation](../deepseek_v3/MTP.md) for those mechanics and the
[GLM README](README.md) for recipes and validation commands. This note covers
the GLM-specific checkpoint and DSA contracts.

## Checkpoint layout

The released GLM-5.2 configuration has 78 decoder layers and one NextN block.
Its checkpoint stores that block under `model.layers.78`, including its DSA
indexer, MLA, routed/shared MoE and norms, plus `enorm`, `hnorm`, `eh_proj`
and `shared_head.norm`. The vocabulary projection uses the shared top-level
`lm_head`; there is no additional `shared_head.head.weight`.

`Glm5StateDictAdapter` maps this block to native `mtp_layers.0.*`.
Adapter tests cover native/HF round trips and the released key layout.
GLM-5.3 uses the same architecture configurations rather than separate aliases.

References: [released configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
and [checkpoint index](https://huggingface.co/zai-org/GLM-5.2/blob/main/model.safetensors.index.json).

## DSA training differences

For each shifted objective, GLM constructs linear-size document and validity
metadata so DSA selections respect packed-document and padding boundaries.
Under CP this metadata identifies local queries against the global key pool.
`Glm5MTPTransformerBlock` specializes the attention metadata contract while
reusing the upstream block construction. Frozen indexer work is included in
the model's forward-only indexer FLOP correction, including MTP indexers.

The recipe's default `mtp_scale=0.3` follows TorchTitan's DeepSeek-V3
convention; the released GLM configuration does not publish a training loss
scale. Hugging Face target-decoder parity does not validate auxiliary MTP
logits because its GLM implementation omits the NextN block.

## Supported boundary

MTP uses the shared FSDP and TP/SP lifecycle and GLM's CP metadata handling.
PP+MTP is rejected at model configuration time; it is not silently reduced to
base-model training.

The released `index_share_for_mtp_iteration` describes iterative index reuse
in speculative decoding. This implementation trains the NextN objective;
it does not provide a serving scheduler, KV cache or acceptance/rejection loop.
