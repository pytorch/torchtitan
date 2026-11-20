# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
import torch.nn as nn

from torchtitan.config.transform.base import convert_config_type
from torchtitan.distributed.local_compile import (
    INFERENCE_BATCH_INVARIANT_REGION,
    LocalCompileConfig,
)
from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.config_utils import make_router_config
from torchtitan.models.common.dist_moe.routed_experts import _mask_padded_routes
from torchtitan.rl.generator.moe_router import FusedSoftmaxTopKRouter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _FixedLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits_TE: torch.Tensor | None = None

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        del x_TD
        assert self.logits_TE is not None
        return self.logits_TE


def _routers(
    *,
    dim: int,
    num_experts: int,
    top_k: int,
    route_norm: bool,
    route_scale: float,
):
    config = make_router_config(
        dim=dim,
        num_experts=num_experts,
        gate_param_init={"weight": partial(nn.init.normal_, std=0.05)},
        score_func=Softmax.Config(),
        top_k=top_k,
        route_norm=route_norm,
        route_scale=route_scale,
    )
    with torch.device("cuda"):
        stock = config.build()
        fused = convert_config_type(config, FusedSoftmaxTopKRouter).build()
    stock.init_states()
    fused.load_state_dict(stock.state_dict())
    return stock.eval(), fused.eval()


def _logits(kind: str, num_tokens: int, num_experts: int) -> torch.Tensor:
    shape = (num_tokens, num_experts)
    if kind == "normal":
        return torch.randn(shape, device="cuda") * 4
    if kind == "ties":
        return torch.randint(-2, 3, shape, device="cuda").float()
    if kind == "underflow":
        return torch.randn(shape, device="cuda") * 80
    assert kind == "nonfinite"
    logits_TE = torch.randn(shape, device="cuda")
    rows = torch.arange(0, num_tokens, 3, device="cuda")
    cols = torch.randint(0, num_experts, rows.shape, device="cuda")
    logits_TE[rows, cols] = torch.tensor(
        [float("inf"), float("nan")], device="cuda"
    ).repeat(rows.numel())[: rows.numel()]
    return logits_TE


def _assert_bitwise(actual, expected) -> None:
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == e.dtype and a.shape == e.shape
        if a.dtype == torch.float32:
            assert torch.equal(a.isnan(), e.isnan())
            a, e = a.nan_to_num().view(torch.int32), e.nan_to_num().view(torch.int32)
        assert torch.equal(a, e)


@pytest.mark.parametrize(
    "num_experts, top_k",
    [(8, 2), (64, 1), (160, 6), (512, 10), (2048, 32)],
)
@pytest.mark.parametrize("kind", ["normal", "ties", "underflow", "nonfinite"])
@torch.inference_mode()
def test_fused_router_matches_stock_router_bitwise(num_experts, top_k, kind):
    torch.manual_seed(num_experts * 100 + top_k)
    for route_norm, route_scale, with_bias in (
        (True, 1.0, False),
        (False, 2.5, False),
        (True, 2.5, True),
    ):
        stock, fused = _routers(
            dim=16,
            num_experts=num_experts,
            top_k=top_k,
            route_norm=route_norm,
            route_scale=route_scale,
        )
        stock.gate = _FixedLogits()
        fused.gate = _FixedLogits()
        expert_bias_E = (
            torch.randint(-1, 2, (num_experts,), device="cuda").float() * 1e-3
            if with_bias
            else None
        )
        for num_tokens in (1, 7, 128, 2049) + ((5000,) if num_experts == 512 else ()):
            logits_TE = _logits(kind, num_tokens, num_experts)
            stock.gate.logits_TE = logits_TE
            fused.gate.logits_TE = logits_TE
            x_TD = torch.empty(num_tokens, 16, device="cuda")
            _assert_bitwise(fused(x_TD, expert_bias_E), stock(x_TD, expert_bias_E))


@torch.inference_mode()
def test_fused_router_and_padding_mask_match_eager_under_cuda_graph_replay():
    torch.manual_seed(0)
    num_tokens, dim = 64, 256
    stock, fused = _routers(
        dim=dim, num_experts=512, top_k=10, route_norm=True, route_scale=1.0
    )
    x_TD = torch.empty(num_tokens, dim, device="cuda", dtype=torch.bfloat16)
    padding_mask_T = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")

    def routed(router):
        scores_TK, expert_ids_TK, routing_map_TE = router(x_TD)
        expert_ids_TK, scores_TK = _mask_padded_routes(
            expert_ids_TK, scores_TK, padding_mask_T
        )
        return scores_TK, expert_ids_TK, routing_map_TE

    def expected():
        scores_TK, expert_ids_TK, routing_map_TE = stock(x_TD)
        padding_mask_TK = padding_mask_T.unsqueeze(-1)
        return (
            scores_TK.masked_fill(padding_mask_TK, 0.0),
            expert_ids_TK.masked_fill(padding_mask_TK, -1),
            routing_map_TE,
        )

    LocalCompileConfig(regions=[INFERENCE_BATCH_INVARIANT_REGION]).apply()
    try:
        x_TD.normal_()
        routed(fused)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = routed(fused)
        for num_valid_tokens in (num_tokens, 48, 1):
            x_TD.normal_()
            x_TD[num_valid_tokens:] = 1e4
            padding_mask_T.copy_(
                torch.arange(num_tokens, device="cuda") >= num_valid_tokens
            )
            graph.replay()
            _assert_bitwise(graph_outputs, expected())
            if num_valid_tokens < num_tokens:
                assert (graph_outputs[1][num_valid_tokens:] == -1).all()
                assert (graph_outputs[0][num_valid_tokens:] == 0).all()
    finally:
        LocalCompileConfig().apply()
