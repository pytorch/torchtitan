# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for the G-005 index_add problems: run(src [R, O] bf16, index [R] int64 in [0, 9)).

Even seeds: sorted indices with random per-destination counts, including the padding row 8,
as the dispatcher's repeat_interleave produces them. Odd seeds: unsorted random indices, so
a kernel cannot rely on sorting. R defaults to the training value 65536.
"""

import torch


def make_inputs(spec, g, sizes):
    from harness import concrete

    o = spec["inputs"][0]["shape"][1]
    r = concrete(spec["inputs"][0]["shape"][:1], {"__default__": 65536, **sizes})[0]
    src = torch.randn(r, o, generator=g, device="cuda").to(torch.bfloat16)
    if (
        int(torch.randint(0, 2, (), generator=g, device="cuda")) == 0
        or g.initial_seed() % 2 == 0
    ):
        counts = torch.rand(9, generator=g, device="cuda")
        counts = (counts / counts.sum() * r).floor().long()
        counts[0] += r - counts.sum()
        index = torch.repeat_interleave(torch.arange(9, device="cuda"), counts)
    else:
        index = torch.randint(0, 9, (r,), generator=g, device="cuda")
    return [src, index]
