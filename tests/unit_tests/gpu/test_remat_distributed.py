# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.config.transform import LoRAConverter, TensorParallelTransform

from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_sequence_parallel_placement,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_spmd,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.protocols.module import Module, ModuleDict


pytestmark = pytest.mark.multi_gpu


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


class _FeedForwardBlock(Module):
    def __init__(self, feed_forward: FeedForward):
        super().__init__()
        self.feed_forward = feed_forward

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x_TD).sum()


class _InnerAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del k_THK, v_THV, kwargs
        return q_THK


class _AttentionBlock(Module):
    def __init__(self, attention: GQAttention):
        super().__init__()
        self.attention = attention

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=None, positions=None).sum()


def _run_forward_backward(model: Module, x_TD: torch.Tensor):
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    output = model(input_TD)
    output.backward()
    assert input_TD.grad is not None
    parameter_grads = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return output.detach(), input_TD.grad.detach().clone(), parameter_grads


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestTpRematRegions(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _parallel_dims(self) -> ParallelDims:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims.build_mesh()
        return parallel_dims

    @staticmethod
    def _feed_forward_config(
        *, enable_sp: bool, use_lora: bool = False
    ) -> FeedForward.Config:
        init = {"weight": torch.nn.init.zeros_}
        config = make_ffn_config(
            dim=4,
            hidden_dim=8,
            w1_param_init=init,
            w2w3_param_init=init,
        )
        config = TensorParallelTransform().transform(config)
        assert isinstance(config, FeedForward.Config)
        if use_lora:
            config = LoRAConverter.Config(rank=2, alpha=4).build().convert(config)
            assert isinstance(config, FeedForward.Config)
        set_dense_ffn_sharding(
            config,
            attn_x_layout=(
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            ),
            enable_sp=enable_sp,
        )
        return config

    @staticmethod
    def _attention_config(*, enable_sp: bool) -> GQAttention.Config:
        init = {"weight": torch.nn.init.zeros_}
        config = make_gqa_config(
            dim=4,
            n_heads=2,
            n_kv_heads=2,
            wqkv_param_init=init,
            wo_param_init=init,
            inner_attention=_InnerAttention.Config(),
            rope=None,
        )
        config = TensorParallelTransform().transform(config)
        assert isinstance(config, GQAttention.Config)
        set_gqa_attention_sharding(config, enable_sp=enable_sp)
        set_gqa_inner_attention_local_spmd(config.inner_attention)
        return config

    @staticmethod
    def _assert_results_equal(expected, actual) -> None:
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def _assert_collectives_follow_regions(
        self,
        *,
        parallel_dims: ParallelDims,
        build_block,
        first_region: str,
        second_region: str,
        enable_sp: bool,
    ) -> None:
        mesh = parallel_dims.spmd_dense_mesh()
        for save_regions, expected_extra_collectives in (
            ([], 2 if enable_sp else 1),
            ([first_region], 1),
            ([second_region], 1 if enable_sp else 0),
            ([first_region, second_region], 0),
        ):
            with (
                self.subTest(save_regions=save_regions),
                # TorchTitan disables autograd multithreading so remat replay
                # can access the thread-local SPMD mesh used by collectives.
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                torch.manual_seed(42)
                baseline = _Model(build_block()).to(self.device_type)
                remat_model = _Model(build_block()).to(self.device_type)
                remat_model.load_state_dict(baseline.state_dict())

                baseline.parallelize(parallel_dims)
                remat_model.parallelize(parallel_dims)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                num_collectives = 0
                original_redistribute = spmd.redistribute

                def counted_redistribute(*args, **kwargs):
                    nonlocal num_collectives
                    src = kwargs["src"]
                    dst = kwargs["dst"]
                    if (isinstance(src, spmd.Shard) and dst in (spmd.I, spmd.R)) or (
                        src is spmd.P
                        and (isinstance(dst, spmd.Shard) or dst in (spmd.I, spmd.R))
                    ):
                        num_collectives += 1
                    return original_redistribute(*args, **kwargs)

                x_TD = torch.randn(4, 4, device=self.device_type)
                with patch.object(
                    spmd, "redistribute", side_effect=counted_redistribute
                ):
                    expected = _run_forward_backward(baseline, x_TD)
                    baseline_collectives = num_collectives
                    num_collectives = 0
                    actual = _run_forward_backward(remat_model, x_TD)

                self._assert_results_equal(expected, actual)
                self.assertEqual(baseline_collectives, 2 if enable_sp else 1)
                self.assertEqual(
                    num_collectives,
                    baseline_collectives + expected_extra_collectives,
                )

    @with_comms
    def test_tp_sp_collectives_follow_regions(self):
        parallel_dims = self._parallel_dims()
        self._assert_collectives_follow_regions(
            parallel_dims=parallel_dims,
            build_block=lambda: _FeedForwardBlock(
                self._feed_forward_config(enable_sp=True).build()
            ),
            first_region="feed_forward.w13",
            second_region="feed_forward.w2",
            enable_sp=True,
        )

    @with_comms
    def test_tp_collectives_follow_regions(self):
        parallel_dims = self._parallel_dims()
        self._assert_collectives_follow_regions(
            parallel_dims=parallel_dims,
            build_block=lambda: _FeedForwardBlock(
                self._feed_forward_config(enable_sp=False).build()
            ),
            first_region="feed_forward.w13",
            second_region="feed_forward.w2",
            enable_sp=False,
        )

    @with_comms
    def test_tp_sp_collectives_enclose_lora_projection(self):
        parallel_dims = self._parallel_dims()
        self._assert_collectives_follow_regions(
            parallel_dims=parallel_dims,
            build_block=lambda: _FeedForwardBlock(
                self._feed_forward_config(enable_sp=True, use_lora=True).build()
            ),
            first_region="feed_forward.w13",
            second_region="feed_forward.w2",
            enable_sp=True,
        )

    @with_comms
    def test_tp_sp_collectives_follow_attention_regions(self):
        parallel_dims = self._parallel_dims()
        self._assert_collectives_follow_regions(
            parallel_dims=parallel_dims,
            build_block=lambda: _AttentionBlock(
                self._attention_config(enable_sp=True).build()
            ),
            first_region="attention.qkv",
            second_region="attention.wo",
            enable_sp=True,
        )

    @with_comms
    def test_tp_collectives_follow_attention_regions(self):
        parallel_dims = self._parallel_dims()
        self._assert_collectives_follow_regions(
            parallel_dims=parallel_dims,
            build_block=lambda: _AttentionBlock(
                self._attention_config(enable_sp=False).build()
            ),
            first_region="attention.qkv",
            second_region="attention.wo",
            enable_sp=False,
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
