# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The MoE counting-sort dispatch (one-hot cumsum ranks -> offsets -> row of each route -> scatter -> gathers) at
decode sizes. (a) Static: 5 kernels at every size, each latency bound. (b) Dynamic (compiled at 2048 then 1024
tokens, largest first as in vLLM's generator): the one-hot cumsum becomes a looped reduction (triton_red) instead
of a persistent one (triton_per). FlashInfer's routing does sort + permutation in 1 kernel (2.7-3.9 us).

Expected on 1x GB300 (see ../README.md, ask 10): static 5 kernels, ~9 us of kernels, ~18 us graph at T=34;
dynamic: cumsum kernel ~4.9 vs ~2.1 us static at T=34.
"""
import torch
from _common import (
    counting_sort_dispatch,
    graph_kernels,
    graph_us,
    header,
    router_inputs,
)

header()
static = torch.compile(counting_sort_dispatch, fullgraph=True, dynamic=False)
dynamic = torch.compile(counting_sort_dispatch, fullgraph=True)
with torch.no_grad():
    for num_tokens in (2048, 1024):
        dynamic(*router_inputs(num_tokens))
    for name, compiled in (("static", static), ("dynamic", dynamic)):
        print(f"--- {name}")
        for num_tokens in (1, 8, 34, 64, 256):
            inputs = router_inputs(num_tokens)
            wall = graph_us(lambda: compiled(*inputs))
            kernels = graph_kernels(lambda: compiled(*inputs))
            print(
                f"T={num_tokens:4d}: graph {wall:5.1f} us, {len(kernels)} kernels, kernel sum "
                f"{sum(us for us, _ in kernels):5.1f} us: "
                + "  ".join(f"{us:.1f}" for us, _ in kernels)
            )
            if num_tokens == 34:
                for us, kernel_name in kernels:
                    print(f"      {us:5.2f} us  {kernel_name[:100]}")
