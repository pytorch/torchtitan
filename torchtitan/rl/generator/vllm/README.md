# Custom Flex attention masks

Define `get_mask_mod()` on a model's `FlexInnerAttention.Config` subclass to
apply the same additional mask in training and vLLM generation:

```python
from dataclasses import dataclass

from torchtitan.models.common.attention import FlexInnerAttention


class StridedFlexInnerAttention(FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        def get_mask_mod(self):
            def strided_mask(b, h, q_idx, kv_idx):
                return (kv_idx % 3 == 0) | (q_idx == kv_idx)

            return strided_mask
```

Use this inner-attention config in both the trainer and generator model configs.
Titan adds causal masking and prevents attention across documents. `q_idx` and
`kv_idx` are positions within a document: the trainer translates packed indices,
and vLLM translates paged KV indices and includes the decode offset. The mask
must be FX-traceable and depend only on query/key positions and scalar model
configuration constants. Config validation rejects batch/head-dependent masks
and captured tensors or modules. Non-causal attention is unsupported.

Training shares metadata by inner-attention class, so layers of the same class
must use the same traced mask expression; differing mask parameters raise
`ValueError`. The generator also builds one callable per class and
sets vLLM's `Attention.logical_mask_mod` before metadata construction. Full CUDA
graphs require the same callable across all layers sharing a vLLM metadata
builder; vLLM rejects differing masks in such a group. Native sliding windows
continue to use `per_layer_sliding_window` and compose with the custom mask.
Custom masks with vLLM's compressed MLA cache raise `ValueError` because its
attention kernels only support causal masking.
