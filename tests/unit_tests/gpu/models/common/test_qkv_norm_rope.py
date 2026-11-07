# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import unittest
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# Qwen3: q/k norms + CosSinRoPE. Llama 3: no norms, ComplexRoPE. Muse Glimmer overrides the
# method with a q scale; its debug layer 0 is sliding window with RoPE, layer 3 global with none.
@pytest.mark.parametrize(
    "model, layer_id",
    [("qwen3", 0), ("llama3", 0), ("muse_glimmer", 0), ("muse_glimmer", 3)],
)
def test_qkv_norm_rope_region_matches_eager(model, layer_id):
    build_model_config = importlib.import_module(
        f"torchtitan.models.{model}"
    ).build_model_config
    torch.manual_seed(0)
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    attention = config.layers[layer_id].attention.build().cuda()
    qkv_linear = attention.qkv_linear
    fp32_params = {name: p.detach().clone() for name, p in attention.named_parameters()}
    token_counts = (64, 37, 1)
    inputs = []
    for num_tokens in token_counts:
        # The wqkv output: [T, n_kv_heads * R * head_dim].
        qkv = torch.randn(
            num_tokens,
            (attention.n_heads + 2 * attention.n_kv_heads) * qkv_linear.head_dim,
            device="cuda",
        )
        positions = torch.randint(0, 1024, (num_tokens,), device="cuda")
        inputs.append((qkv, positions))

    def run(dtype, regions):
        LocalCompileConfig(regions=regions).apply()
        # Parameters only: the RoPE cache stays fp32, as in production.
        for name, param in attention.named_parameters():
            param.data = fp32_params[name].to(dtype)
        with torch.no_grad():
            return [
                attention._prepare_query_key_value(qkv.to(dtype), positions)
                for qkv, positions in inputs
            ]

    try:
        reference = run(torch.float32, [])
        expected = run(torch.bfloat16, [])
        # One binding, as in the generator: the token counts compile a static graph, a
        # dynamic one and a T=1 one.
        actual = run(torch.bfloat16, ["rope", "qkv_norm_rope"])
    finally:
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    # The compiled norm reduces in another order than the eager kernel, so a few elements
    # differ by a bf16 ulp; without emulate_precision_casts ~31% of q would. Check both:
    # few differing elements vs eager, and as close as eager to an fp32 run of the same math.
    for num_tokens, actual_qkv, expected_qkv, reference_qkv in zip(
        token_counts, actual, expected, reference, strict=True
    ):
        for name, actual_tensor, expected_tensor, reference_tensor in zip(
            "qkv", actual_qkv, expected_qkv, reference_qkv, strict=True
        ):
            assert actual_tensor.is_contiguous()
            if name == "v":  # a plain copy
                assert torch.equal(actual_tensor, expected_tensor), num_tokens
            else:
                differing = (actual_tensor != expected_tensor).float().mean()
                assert differing < 1e-3, (num_tokens, name, differing)
            actual_error = (actual_tensor.float() - reference_tensor).norm()
            eager_error = (expected_tensor.float() - reference_tensor).norm()
            assert actual_error <= 1.05 * eager_error, (
                num_tokens,
                name,
                actual_error,
                eager_error,
            )


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestQKVNormRopeTensorParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_tp2_matches_eager(self):
        """TP2 generator setting (no SP, no grad): the region's local q/k/v heads match eager,
        and the decode batch sizes compile at most 3 graphs: static, dynamic and T=1."""
        from torchtitan.models.qwen3 import build_model_config

        torch.manual_seed(0)
        config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(enable_sequence_parallel=False),
        )
        attention = config.layers[0].attention.build().to(self.device_type)
        # Parameters only: the RoPE cache stays fp32, as in production.
        for param in attention.parameters():
            param.data = param.data.to(torch.bfloat16)
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        with patch(
            "torchtitan.distributed.parallelism_context.device_type", self.device_type
        ):
            parallelism_context.build_mesh()
        attention._parallelize(parallelism_context)

        token_counts = (64, 37, 1, 2, 16, 128)
        inputs = []
        for num_tokens in token_counts:
            x_TD = torch.randn(
                num_tokens, config.dim, device=self.device_type, dtype=torch.bfloat16
            )
            positions = torch.randint(0, 1024, (num_tokens,), device=self.device_type)
            inputs.append((x_TD, positions))

        def run(regions):
            LocalCompileConfig(regions=regions).apply()
            with parallelism_context.activate_spmd(), torch.no_grad():
                return [
                    attention._prepare_query_key_value(
                        attention.qkv_linear(x_TD), positions
                    )
                    for x_TD, positions in inputs
                ]

        torch._dynamo.reset()
        try:
            expected = run([])
            unique_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            actual = run(["rope", "qkv_norm_rope"])
            unique_graphs = (
                torch._dynamo.utils.counters["stats"]["unique_graphs"] - unique_graphs
            )
        finally:
            LocalCompileConfig().apply()
            torch._dynamo.reset()

        self.assertLessEqual(unique_graphs, 3)
        for num_tokens, actual_qkv, expected_qkv in zip(
            token_counts, actual, expected, strict=True
        ):
            for name, actual_tensor, expected_tensor in zip(
                "qkv", actual_qkv, expected_qkv, strict=True
            ):
                self.assertEqual(actual_tensor.shape, expected_tensor.shape)
                if name == "v":  # a plain copy
                    self.assertTrue(
                        torch.equal(actual_tensor, expected_tensor), num_tokens
                    )
                else:
                    differing = (actual_tensor != expected_tensor).float().mean()
                    self.assertLess(differing, 1e-3, (num_tokens, name))
