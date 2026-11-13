# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Programmatic dependent launch (PDL) for Inductor's Triton kernels: the MoE dispatch region (5 kernels) in a CUDA
graph, run once with TORCHINDUCTOR_ENABLE_PDL=0 and once with =1 (see ../run_all.sh). Outputs are compared bitwise.

Expected on 1x GB300 (see ../README.md, ask 4): T=1 ~12 us without PDL, ~8 us with; T=34 ~12 vs ~11 us;
bitwise equal.
"""
import os

import torch
from _common import counting_sort_dispatch, graph_us, header, router_inputs

header()
print(f"TORCHINDUCTOR_ENABLE_PDL={os.environ.get('TORCHINDUCTOR_ENABLE_PDL', '0')}")
dispatch = torch.compile(counting_sort_dispatch, fullgraph=True, dynamic=False)
with torch.no_grad():
    for num_tokens in (1, 8, 34, 64, 256):
        inputs = router_inputs(num_tokens)
        reference = counting_sort_dispatch(*inputs)
        bitwise = all(torch.equal(a, b) for a, b in zip(dispatch(*inputs), reference))
        print(
            f"T={num_tokens:4d}: dispatch {graph_us(lambda: dispatch(*inputs)):5.1f} us  bitwise vs eager {bitwise}"
        )
