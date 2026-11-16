# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""With a symbolic token count, the MoE combine (per-token score-weighted sum of its 8 expert rows) keeps bounds
masks on every load and runs ~1.6x slower than the static kernel. Compiled large-first, so this is not ask 8.

numel = T * 2048 with symbolic T is not a known multiple of the max XBLOCK, so the 1D kernel keeps
`xindex < xnumel`. `triton.prefer_nd_tiling` (2D T x D tiles) recovers most of it.

Expected on 1x GB300 (see ../README.md, ask 9), us at T = 4096 / 16384 / 65536:
dynamic ~45 / ~159 / ~637, dynamic + prefer_nd_tiling ~29 / ~94 / ~397, static ~29 / ~92 / ~351.
"""
import torch
import torch._inductor.config as inductor_config
from _common import DIM, header, TOP_K

inductor_config.fx_graph_cache = False
inductor_config.autotune_local_cache = False


def make_combine(scale):
    """Gather-sum combine; ``scale`` makes each compiled copy a distinct kernel (no shared tuned launcher)."""

    def combine(rows_ND, row_of_route_N, scores_TK):
        num_tokens = scores_TK.shape[0]
        rows_TKD = rows_ND[row_of_route_N].view(num_tokens, TOP_K, -1)
        scores_TK = scores_TK.float()
        out_TD = rows_TKD[:, 0].float() * scores_TK[:, 0:1]
        for k in range(1, TOP_K):
            out_TD = out_TD + rows_TKD[:, k].float() * scores_TK[:, k : k + 1]
        return (out_TD * scale).to(rows_ND.dtype)

    return combine


def combine_inputs(num_tokens):
    rows_ND = torch.randn(num_tokens * TOP_K, DIM, device="cuda", dtype=torch.bfloat16)
    return (
        rows_ND,
        torch.randperm(num_tokens * TOP_K, device="cuda"),
        torch.rand(num_tokens, TOP_K, device="cuda"),
    )


if __name__ == "__main__":
    from _common import graph_us

    header()
    token_counts = [4096, 16384, 65536]
    results = {}
    with torch.no_grad():
        dynamic = {}
        for i, (name, options) in enumerate(
            [
                ("dynamic", {}),
                ("dynamic + prefer_nd_tiling", {"triton.prefer_nd_tiling": True}),
            ]
        ):
            torch._dynamo.reset()
            compiled = torch.compile(
                make_combine(1.0 + (i + 1) / 1024), fullgraph=True, options=options
            )
            for num_tokens in (
                16384,
                14336,
            ):  # large-first warmup: the second call compiles dynamic
                compiled(*combine_inputs(num_tokens))
            dynamic[name] = compiled
        for num_tokens in token_counts:
            inputs = combine_inputs(num_tokens)
            for name, compiled in dynamic.items():
                results.setdefault(name, []).append(graph_us(lambda: compiled(*inputs)))
            torch._dynamo.reset()
            static = torch.compile(make_combine(1.0), fullgraph=True, dynamic=False)
            results.setdefault("static", []).append(graph_us(lambda: static(*inputs)))
    print(f"us per call at T = {token_counts} (CUDA graph):")
    for name, times in results.items():
        print(f"  {name:28s} " + " ".join(f"{us:7.1f}" for us in times))
