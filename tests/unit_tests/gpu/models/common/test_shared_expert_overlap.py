# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact numerical checks for shared-expert CUDA stream scheduling."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.activation import Sigmoid, SiTUGLU
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.moe import KimiLatentMoE


@pytest.fixture(autouse=True)
def deterministic_numerics():
    """Use strict deterministic execution for eager and graph comparisons."""
    torch.manual_seed(42)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _moe_config(*, latent: bool) -> MoE.Config:
    """Build shared and learned-router branches with a reduced SiTU-GLU FFN."""
    dim, expert_dim, hidden_dim, num_experts = 128, 64 if latent else 128, 128, 4
    values = dict(
        num_experts=num_experts,
        load_balance_coeff=None,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=HiMidLoLinear.Config(in_features=dim, out_features=num_experts),
            score_func=Sigmoid.Config(),
            top_k=2,
            route_norm=True,
        ),
        routed_experts=RoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=num_experts,
                in_features=expert_dim,
                out_features=hidden_dim,
                num_linears=2,
            ),
            w2=GroupedLinear.Config(
                group_size=num_experts,
                in_features=hidden_dim,
                out_features=expert_dim,
            ),
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=num_experts, top_k=2
            ),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
        shared_experts=DenseFeedForward.Config(
            w13=Linear.Config(in_features=dim, out_features=hidden_dim, num_linears=2),
            w2=Linear.Config(in_features=hidden_dim, out_features=dim),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
    )
    if latent:
        return KimiLatentMoE.Config(
            **values,
            routed_down=Linear.Config(in_features=dim, out_features=expert_dim),
            routed_norm=RMSNorm.Config(normalized_shape=expert_dim),
            routed_up=Linear.Config(in_features=expert_dim, out_features=dim),
        )
    return MoE.Config(**values)


class _Block(Module):
    def __init__(self, config: MoE.Config):
        super().__init__()
        self.moe = config.build()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.moe(x_TD)


class _Model(Module):
    def __init__(self, config: MoE.Config):
        super().__init__()
        self.layers = ModuleDict({"0": _Block(config)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_overlap_preserves_exact_training(
    latent: bool, checkpoint: bool
) -> None:
    """Twelve deterministic updates preserve exact outputs, norms and gradients."""
    torch.manual_seed(42)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        config = _moe_config(latent=latent)
        serial = _Model(config).cuda().bfloat16()
        overlapped = (
            _Model(replace(config, overlap_shared_experts=True)).cuda().bfloat16()
        )
        # GroupedLinear and RMSNorm parameters are initialized via init_states.
        with torch.no_grad():
            for parameter in serial.parameters():
                parameter.uniform_(-0.05, 0.05)
        overlapped.load_state_dict(deepcopy(serial.state_dict()))
        if checkpoint:
            FullActivationRematConfig().apply(serial)
            FullActivationRematConfig().apply(overlapped)
        streams = {}

        def shared_forward(module, args):
            streams["shared"] = torch.cuda.current_stream().cuda_stream

        def routed_forward(module, args):
            streams["routed"] = torch.cuda.current_stream().cuda_stream

        handles = [
            overlapped.layers["0"].moe.shared_experts.register_forward_pre_hook(
                shared_forward
            ),
            overlapped.layers["0"].moe.routed_experts.register_forward_pre_hook(
                routed_forward
            ),
        ]
        for _ in range(12):
            input_TD = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
            results = []
            for model in (serial, overlapped):
                model.zero_grad(set_to_none=True)
                x_TD = input_TD.detach().clone().requires_grad_(True)
                output_TD = model(x_TD)
                loss = output_TD.float().square().mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                results.append((output_TD.detach(), loss.detach(), norm, x_TD.grad))
                del output_TD, loss
            torch.cuda.synchronize()
            assert streams["shared"] != streams["routed"]
            assert streams["routed"] == torch.cuda.current_stream().cuda_stream
            for actual, expected in zip(results[1], results[0], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(
                overlapped.parameters(), serial.parameters(), strict=True
            ):
                assert actual.grad is not None and expected.grad is not None
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
                with torch.no_grad():
                    actual.add_(actual.grad, alpha=-0.01)
                    expected.add_(expected.grad, alpha=-0.01)
        for handle in handles:
            handle.remove()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_overlap_cuda_graph_replay(checkpoint: bool) -> None:
    """Five changing-input updates match serial outputs and every gradient."""
    torch.manual_seed(42)
    model = _Model(replace(_moe_config(latent=True), overlap_shared_experts=True))
    model = model.cuda().bfloat16()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    if checkpoint:
        FullActivationRematConfig().apply(model)
    serial = _Model(_moe_config(latent=True)).cuda().bfloat16()
    serial.load_state_dict(deepcopy(model.state_dict()))
    if checkpoint:
        FullActivationRematConfig().apply(serial)
    x_TD = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    parameters = tuple(model.parameters())

    def forward_backward():
        output_TD = model(x_TD)
        loss = output_TD.float().square().mean()
        gradients = torch.autograd.grad(loss, (x_TD, *parameters))
        return output_TD, loss, *gradients

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            forward_backward()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward_backward()
    for _ in range(5):
        with torch.no_grad():
            x_TD.copy_(torch.randn_like(x_TD))
        graph.replay()
        reference_TD = x_TD.detach().clone().requires_grad_(True)
        expected_output = serial(reference_TD)
        expected_loss = expected_output.float().square().mean()
        expected_gradients = torch.autograd.grad(
            expected_loss, (reference_TD, *serial.parameters())
        )
        torch.cuda.synchronize()
        for candidate, expected in zip(
            actual, (expected_output, expected_loss, *expected_gradients), strict=True
        ):
            torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
        with torch.no_grad():
            for parameter, reference, gradient, reference_gradient in zip(
                parameters,
                serial.parameters(),
                actual[3:],
                expected_gradients[1:],
                strict=True,
            ):
                parameter.add_(gradient, alpha=-0.01)
                reference.add_(reference_gradient, alpha=-0.01)
                torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
