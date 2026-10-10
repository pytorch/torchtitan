# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.model.vllm_wrapper import (
    _replace_vllm_layer_configs,
    VLLMModelWrapper,
)


def test_vllm_replacement_preserves_attention_sharding() -> None:
    """Pass the trainer attention's sharding config through unchanged."""
    model_config = build_model_config("debugmodel", attn_backend="flex")
    model_config.set_sharding_(
        ParallelismConfig(tensor_parallel_degree=2, enable_sequence_parallel=True)
    )
    model_config.layers = [
        layer for layer in model_config.layers if layer.attention is not None
    ]

    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        assert model_layer.attention is not None
        assert vllm_layer.attention is not None
        model_sharding = model_layer.attention.inner_attention.sharding_config
        vllm_sharding = vllm_layer.attention.inner_attention.sharding_config
        assert model_sharding is not None
        assert vllm_sharding is not None
        assert vllm_sharding.in_src_shardings is model_sharding.in_src_shardings
        assert vllm_sharding.in_dst_shardings is model_sharding.in_dst_shardings
        assert vllm_sharding.out_src_shardings is model_sharding.out_src_shardings
        assert vllm_sharding.out_dst_shardings is model_sharding.out_dst_shardings
        assert vllm_sharding.local_spmd is model_sharding.local_spmd
        for name, layout in model_sharding.state_shardings.items():
            assert vllm_sharding.state_shardings[name] is layout


def test_routers_expose_routed_experts_to_vllm_capture():
    """vLLM binds ``capture_fn`` by attribute; every MoE router must call it with its ids."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from torchtitan.models.common.activation import Sigmoid
    from torchtitan.models.common.config_utils import make_router_config

    def build_router():
        router = make_router_config(
            dim=4,
            num_experts=4,
            score_func=Sigmoid.Config(),
            gate_param_init={"weight": torch.nn.init.zeros_},
            top_k=2,
        ).build()
        router.init_states()
        return router

    for tp_enabled, gathered_rows in ((False, 3), (True, 6)):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict(
            {"0": torch.nn.Module(), "1": torch.nn.Module()}
        )
        model.layers["0"].feed_forward = torch.nn.Linear(4, 4)
        model.layers["1"].router = build_router()
        wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.parallelism_context = SimpleNamespace(
            ep_enabled=True, tp_enabled=tp_enabled
        )
        tp_group = SimpleNamespace(
            all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim)
        )
        captured = []

        with patch(
            "torchtitan.rl.model.vllm_wrapper.get_tp_group", return_value=tp_group
        ):
            wrapper._expose_routed_experts_to_vllm()
            router = model.layers["1"].router
            assert router.layer_id == 1 and router.capture_fn is None
            router(torch.randn(3, 4))  # before vLLM binds capture_fn: no capture
            router.capture_fn = captured.append
            _, topk_expert_ids_TK, _ = router(torch.randn(3, 4))

        assert not hasattr(model.layers["0"].feed_forward, "layer_id")
        assert len(captured) == 1 and captured[0].shape == (gathered_rows, 2)
        assert torch.equal(captured[0][:3], topk_expert_ids_TK)
