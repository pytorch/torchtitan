# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_utils import DeterministicGuard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils._pytree import tree_map

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    SlidingWindowVarlenInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _metadata():
    boundaries = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    return VarlenAttentionMetadata(boundaries, boundaries, 32, 32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("upcast_qk", [False, True])
@pytest.mark.parametrize("window_size", [(-1, -1), (-1, 0), (7, 0)])
def test_forward_backward_precision(dtype, upcast_qk, window_size):
    torch.manual_seed(42)
    left, right = window_size
    attention = VarlenInnerAttention.Config().build()
    inputs = [
        torch.randn(
            64,
            4,
            64,
            device="cuda",
            dtype=torch.float32 if upcast_qk and index < 2 else dtype,
            requires_grad=True,
        )
        for index in range(3)
    ]
    reference_inputs = [x.detach().double().requires_grad_() for x in inputs]
    metadata = _metadata()._replace(window_size=window_size)
    actual = attention(*inputs, attention_metadata=metadata)
    reference_q, reference_k, reference_v = (
        x.reshape(2, 32, 4, 64).transpose(1, 2) for x in reference_inputs
    )
    with sdpa_kernel(SDPBackend.MATH):
        indices = torch.arange(32, device="cuda")
        mask = None
        if left >= 0:
            mask = (indices[:, None] >= indices[None, :]) & (
                indices[:, None] - indices[None, :] <= left
            )
        expected = (
            F.scaled_dot_product_attention(
                reference_q,
                reference_k,
                reference_v,
                is_causal=left == -1 and right == 0,
                attn_mask=mask,
            )
            .transpose(1, 2)
            .reshape(64, 4, 64)
        )
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    expected_grads = torch.autograd.grad(
        expected, reference_inputs, grad_output.double()
    )

    assert actual.dtype == inputs[0].dtype
    tolerance = 0.001 if dtype == torch.float16 else 0.01
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        relative_error = (result.double() - reference).norm() / reference.norm()
        assert relative_error.item() < tolerance


def test_fp32_fallback_matches_legacy_forward_backward():
    torch.manual_seed(42)
    attention = VarlenInnerAttention.Config().build()
    inputs = [
        torch.randn(64, 4, 64, device="cuda", requires_grad=True) for _ in range(3)
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    metadata = _metadata()
    actual = attention(*inputs, attention_metadata=metadata)
    expected = attention(
        *(x.bfloat16() for x in reference_inputs), attention_metadata=metadata
    ).float()
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    torch.testing.assert_close(
        (actual, actual_grads), (expected, expected_grads), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["compile", "trace", "graph"])
@pytest.mark.parametrize("sliding_window_size", [None, 8])
def test_execution_modes_match_eager(dtype, mode, sliding_window_size):
    torch.manual_seed(42)
    config = (
        VarlenInnerAttention.Config()
        if sliding_window_size is None
        else SlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=sliding_window_size
        )
    )
    attention = config.build()
    inputs = [
        torch.randn(64, 4, 64, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    metadata = config.build_attention_metadata(
        torch.arange(32, device="cuda").repeat(2),
        max_num_documents=2,
        max_context_length=32,
    )

    def step(q, k, v, *, attention_metadata):
        output = attention(q, k, v, attention_metadata=attention_metadata)
        loss = output.float().square().mean()
        return output, loss, torch.autograd.grad(loss, (q, k, v))

    expected = tree_map(torch.Tensor.detach, step(*inputs, attention_metadata=metadata))
    if mode == "trace":
        traced = minimal_fx_tracer(step, module=attention)(
            *inputs, attention_metadata=metadata
        )
        actual = run_traced(traced, module=attention)(
            *inputs, attention_metadata=metadata
        )
    elif mode == "compile":
        compiled = torch.compile(attention, fullgraph=True)
        output = compiled(*inputs, attention_metadata=metadata)
        loss = output.float().square().mean()
        actual = output, loss, torch.autograd.grad(loss, inputs)
    else:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step(*inputs, attention_metadata=metadata)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = step(*inputs, attention_metadata=metadata)
        with torch.no_grad():
            for tensor in inputs:
                tensor.normal_()
        reference_inputs = [
            tensor.detach().clone().requires_grad_() for tensor in inputs
        ]
        expected = tree_map(
            torch.Tensor.detach, step(*reference_inputs, attention_metadata=metadata)
        )
        graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestVarlenUlyssesNumerics(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_packed_forward_backward_matches_unsharded(self):
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("cp",))
        device = torch.device("cuda", self.rank)
        # The middle document crosses the CP boundary at token 64.
        positions = torch.cat([torch.arange(n) for n in (17, 79, 24, 8)]).to(device)
        padding_mask = torch.arange(128, device=device) >= 120
        metadata = create_varlen_metadata_for_document(
            positions,
            padding_mask=padding_mask,
            max_num_documents=6,
            max_context_length=96,
        )

        for dtype in (torch.float16, torch.bfloat16):
            for kv_heads in (8, 4):
                with (
                    self.subTest(dtype=dtype, kv_heads=kv_heads),
                    DeterministicGuard(True),
                ):
                    torch.manual_seed(42)
                    inputs = [
                        torch.randn(
                            128,
                            heads,
                            64,
                            device=device,
                            dtype=dtype,
                            requires_grad=True,
                        )
                        for heads in (8, kv_heads, kv_heads)
                    ]
                    local_inputs = [
                        x.detach()
                        .chunk(self.world_size)[self.rank]
                        .clone()
                        .requires_grad_()
                        for x in inputs
                    ]
                    expected = VarlenInnerAttention.Config().build()(
                        *inputs, attention_metadata=metadata, enable_gqa=kv_heads != 8
                    )
                    expected_loss = expected[~padding_mask].float().square().sum()
                    expected_loss.backward()

                    with set_current_spmd_mesh(mesh):
                        actual = UlyssesVarlenInnerAttention.Config().build()(
                            *local_inputs,
                            attention_metadata=metadata,
                            enable_gqa=kv_heads != 8,
                        )
                        local_padding = padding_mask.chunk(self.world_size)[self.rank]
                        actual_loss = actual[~local_padding].float().square().sum()
                        actual_loss.backward()

                    loss = actual_loss.detach().clone()
                    dist.all_reduce(loss)
                    torch.testing.assert_close(loss, expected_loss, atol=0, rtol=1e-5)
                    tolerance = 0.001 if dtype == torch.float16 else 0.01
                    for result, reference in zip(
                        (actual, *(x.grad for x in local_inputs)),
                        (expected, *(x.grad for x in inputs)),
                    ):
                        reference = reference.chunk(self.world_size)[self.rank]
                        relative_error = (
                            result.float() - reference.float()
                        ).norm() / reference.float().norm()
                        self.assertLess(relative_error.item(), tolerance)
                    for x in local_inputs:
                        self.assertEqual(
                            torch.count_nonzero(x.grad[local_padding]).item(), 0
                        )
