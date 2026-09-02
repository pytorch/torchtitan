# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.deepseek_v3.model import Attention

from torchtitan.models.kimi_k2_7.qk_clip import (
    qk_clip,
    QKClipFlexAttention,
    register_qk_clip_hook,
)


class QKClipTest(unittest.TestCase):
    def test_attention_records_training_maxima_only(self) -> None:
        attention = QKClipFlexAttention.Config().build()
        q_THK = torch.randn(2, 2, 4)
        max_scores_1HT = torch.tensor([[[1.0, 3.0], [4.0, 2.0]]])
        block_mask = create_block_mask(
            lambda _b, _h, q_idx, kv_idx: q_idx >= kv_idx,
            1,
            2,
            2,
            2,
            device="cpu",
            _compile=False,
        )
        aux = SimpleNamespace(lse=None, max_scores=max_scores_1HT)

        attention.train()
        with patch(
            "torchtitan.models.common.attention.FlexAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            attention(
                q_THK,
                q_THK,
                q_THK,
                attention_masks=block_mask,
            )

        self.assertEqual(len(attention.max_attention_logits_H), 1)
        torch.testing.assert_close(
            attention.max_attention_logits_H[0],
            torch.tensor([3.0, 4.0]),
        )

        attention.max_attention_logits_H.clear()
        attention.eval()
        with patch(
            "torchtitan.models.common.attention.FlexAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            attention(
                q_THK,
                q_THK,
                q_THK,
                attention_masks=block_mask,
            )

        self.assertFalse(attention.max_attention_logits_H)

    def test_optimizer_hook_runs_qk_clip(self) -> None:
        model = nn.Linear(2, 2, bias=False)
        optimizers = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 0.0,
                        "weight_decay": 0.0,
                    },
                )
            ],
        ).build(model_parts=[model])
        reduction_mesh = cast(
            DeviceMesh,
            SimpleNamespace(size=lambda: 1),
        )

        def get_mesh(mesh_name: str) -> DeviceMesh:
            self.assertEqual(mesh_name, "loss")
            return reduction_mesh

        parallel_dims = cast(
            ParallelDims,
            SimpleNamespace(get_mesh=get_mesh),
        )
        with patch("torchtitan.models.kimi_k2_7.qk_clip.qk_clip") as mock_qk_clip:
            register_qk_clip_hook(optimizers, [model], parallel_dims)
            model.weight.grad = torch.zeros_like(model.weight)
            optimizers.step()

        mock_qk_clip.assert_called_once_with(
            [model],
            reduction_mesh=reduction_mesh,
        )


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class QKClipDistributedTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    @with_comms
    def test_reduces_head_maxima_and_clips_local_dtensor_weights(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        num_heads = self.world_size
        qk_nope_head_dim = 2
        qk_rope_head_dim = 1
        v_head_dim = 2
        in_features = 2

        def weight_module(num_rows: int) -> nn.Module:
            module = nn.Module()
            module.register_parameter(
                "weight",
                nn.Parameter(
                    distribute_tensor(
                        torch.ones(num_rows, in_features, device=device),
                        mesh,
                        (Shard(0),),
                    )
                ),
            )
            return module

        attention = Attention.__new__(Attention)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 1
        attention.qk_nope_head_dim = qk_nope_head_dim
        attention.qk_rope_head_dim = qk_rope_head_dim
        attention.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        attention.v_head_dim = v_head_dim
        attention.wq_b = weight_module(num_heads * attention.qk_head_dim)
        attention.wkv_b = weight_module(num_heads * (qk_nope_head_dim + v_head_dim))
        attention.inner_attention = QKClipFlexAttention.Config().build()
        rank_maxima = (
            torch.tensor([50.0, 400.0], device=device)
            if self.rank == 0
            else torch.tensor([200.0, 50.0], device=device)
        )
        attention.inner_attention.max_attention_logits_H.append(rank_maxima)
        model = nn.Module()
        model.add_module("attention", attention)

        qk_clip(
            [model],
            reduction_mesh=mesh,
        )

        local_scale = 0.5 if self.rank == 0 else 0.25
        q_weight_HDI = attention.wq_b.weight.to_local().view(
            1,
            attention.qk_head_dim,
            in_features,
        )
        kv_weight_HDI = attention.wkv_b.weight.to_local().view(
            1,
            qk_nope_head_dim + v_head_dim,
            in_features,
        )
        torch.testing.assert_close(
            q_weight_HDI[:, :qk_nope_head_dim],
            torch.full(
                (1, qk_nope_head_dim, in_features),
                local_scale**0.5,
                device=device,
            ),
        )
        torch.testing.assert_close(
            q_weight_HDI[:, qk_nope_head_dim:],
            torch.full(
                (1, qk_rope_head_dim, in_features),
                local_scale,
                device=device,
            ),
        )
        torch.testing.assert_close(
            kv_weight_HDI[:, :qk_nope_head_dim],
            torch.full(
                (1, qk_nope_head_dim, in_features),
                local_scale**0.5,
                device=device,
            ),
        )
        torch.testing.assert_close(
            kv_weight_HDI[:, qk_nope_head_dim:],
            torch.ones(1, v_head_dim, in_features, device=device),
        )
        self.assertFalse(attention.inner_attention.max_attention_logits_H)

    @with_comms
    def test_weight_scaling_is_communication_free(self) -> None:
        """Per-head scaling must stay local for every clipped weight.

        Building the replicated scales with ``distribute_tensor`` instead of
        ``from_local`` would broadcast once per weight per step, so this asserts
        the packed maxima all-reduce is the only collective ``qk_clip`` issues.
        """
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        num_heads = self.world_size
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, in_features = 2, 1, 2, 2
        num_layers = 4

        def weight_module(num_rows: int) -> nn.Module:
            module = nn.Module()
            module.register_parameter(
                "weight",
                nn.Parameter(
                    distribute_tensor(
                        torch.ones(num_rows, in_features, device=device),
                        mesh,
                        (Shard(0),),
                    )
                ),
            )
            return module

        model = nn.Module()
        for layer_id in range(num_layers):
            attention = Attention.__new__(Attention)
            nn.Module.__init__(attention)
            attention.q_lora_rank = 1
            attention.qk_nope_head_dim = qk_nope_head_dim
            attention.qk_rope_head_dim = qk_rope_head_dim
            attention.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            attention.v_head_dim = v_head_dim
            attention.wq_b = weight_module(num_heads * attention.qk_head_dim)
            attention.wkv_b = weight_module(num_heads * (qk_nope_head_dim + v_head_dim))
            attention.inner_attention = QKClipFlexAttention.Config().build()
            attention.inner_attention.max_attention_logits_H.append(
                torch.full((num_heads,), 400.0, device=device)
            )
            model.add_module(f"layer_{layer_id}", attention)

        with CommDebugMode() as comm_mode:
            qk_clip([model], reduction_mesh=mesh)

        collectives = {
            str(op): count for op, count in comm_mode.get_comm_counts().items() if count
        }
        total = sum(collectives.values())
        # One packed MAX all-reduce covers every layer and head; the 8 weights
        # this model clips must add nothing on top of it.
        self.assertEqual(total, 1, f"expected one collective, got {collectives}")
