# Q5: Attention Kernel Dispatch

Locate the core attention module. Trace the logic that selects between an
optimised kernel (FlashAttention, PyTorch SDPA flash backend, ring attention,
etc.) and any fallback math implementation. Identify the exact file, class, line
number, and configuration flag/condition controlling this dispatch. If the
dispatch is delegated to the PyTorch scaled_dot_product_attention, say so and
cite the call site.
