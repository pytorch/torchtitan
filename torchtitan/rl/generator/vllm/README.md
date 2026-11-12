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
must depend only on its arguments and fixed model configuration. Batch-dependent
masks, captured training tensors, and non-causal attention are unsupported.

Training shares metadata by inner-attention class, so layers of the same class
must use the same mask. The generator also builds one callable per class and
sets vLLM's `Attention.logical_mask_mod` before metadata construction. Full CUDA
graphs require the same callable across all layers sharing a vLLM metadata
builder; vLLM rejects differing masks in such a group. Native sliding windows
continue to use `per_layer_sliding_window` and compose with the custom mask.
