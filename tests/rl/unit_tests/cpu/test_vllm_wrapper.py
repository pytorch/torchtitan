# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
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


class _Wrapper:
    """Stand-in `self` for VLLMModelWrapper's real weight-sync methods (no vLLM)."""

    load_state_dict = VLLMModelWrapper.load_state_dict
    prepare_for_forward = VLLMModelWrapper.prepare_for_forward

    def __init__(self, model: nn.Module):
        self.model = model


def test_load_state_dict_allocates_one_fsdp_group_at_a_time(tmp_path) -> None:
    """Load new weights with at most two FSDP groups' sharded buffers allocated at once.

    Guards that each group's sharded buffers are allocated before its copy, even when
    FSDP's own load pre-hook already resharded the group (tok_embeddings here).
    """
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        # The generator's layout: a tied [tok_embeddings, norm, lm_head] group, one per block.
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(32, 16)
        model.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)) for _ in range(3)
        )
        model.norm = nn.LayerNorm(16)
        model.lm_head = nn.Linear(16, 32, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        mesh = init_device_mesh("cpu", (1,))
        fully_shard([model.tok_embeddings, model.norm, model.lm_head], mesh=mesh)
        for block in model.layers:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model.set_keep_unsharded_storage(True)
        new_state_dict = {k: v.detach() + 1.0 for k, v in model.state_dict().items()}
        groups = [
            group
            for module in (model.tok_embeddings, *model.layers)
            for group in module._get_fsdp_state()._fsdp_param_groups
        ]

        def is_allocated(group) -> bool:
            return all(
                p._sharded_param_data.untyped_storage().size() > 0
                for p in group.fsdp_params
            )

        wrapper = _Wrapper(model)
        max_allocated_groups = 0

        def prepare_for_state_dict_load(module) -> None:
            nonlocal max_allocated_groups
            VLLMModelWrapper.prepare_for_state_dict_load(wrapper, module)
            # The load copies into these buffers next, so freed ones would be overrun.
            assert all(map(is_allocated, module._get_fsdp_state()._fsdp_param_groups))
            max_allocated_groups = max(
                max_allocated_groups, sum(map(is_allocated, groups))
            )

        wrapper.prepare_for_state_dict_load = prepare_for_state_dict_load
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                wrapper.prepare_for_forward(module)
        wrapper.load_state_dict(new_state_dict)

        assert max_allocated_groups <= 2
        assert all(
            p._sharded_param_data.untyped_storage().size() == 0
            for group in groups
            for p in group.fsdp_params
        )
        for name, param in model.named_parameters(remove_duplicate=False):
            assert param.equal(new_state_dict[name].to_local()), name
    finally:
        dist.destroy_process_group()
