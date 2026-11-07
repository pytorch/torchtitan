# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.distributed import ParallelismContext
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

from torchtitan.models.kimi_k2_7.qk_clip import (
    qk_clip,
    QKClipFlexInnerAttention,
    register_qk_clip_hook,
    register_qk_clip_shape_inference_cleanup,
)


def _qk_clip_config() -> QKClipFlexInnerAttention.Config:
    return QKClipFlexInnerAttention.Config()


def _qk_clip_mla_attention() -> MLAAttention:
    return MLAAttention.Config(
        wkv_b=Linear.Config(in_features=1, out_features=1),
        packed_kv_head_dim=1,
        inner_attention=_qk_clip_config(),
    ).build()


class QKClipTest(unittest.TestCase):
    def test_attention_records_training_maxima_only(self) -> None:
        attention = _qk_clip_config().build()
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
            "torchtitan.models.common.attention.FlexInnerAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            FlexInnerAttention.forward(
                attention,
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=block_mask,
            )

        self.assertEqual(len(attention.max_attention_logits_H), 1)
        torch.testing.assert_close(
            attention.max_attention_logits_H[0],
            torch.tensor([3.0, 4.0]),
        )

        attention.max_attention_logits_H.clear()
        attention.eval()
        with patch(
            "torchtitan.models.common.attention.FlexInnerAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            FlexInnerAttention.forward(
                attention,
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=block_mask,
            )

        self.assertFalse(attention.max_attention_logits_H)

    def test_remat_replay_does_not_record_duplicate_maxima(self) -> None:
        attention = _qk_clip_config().build()
        attention.train()
        q_THK = torch.randn(2, 2, 4, requires_grad=True)
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

        def forward(q_THK: torch.Tensor) -> torch.Tensor:
            return FlexInnerAttention.forward(
                attention,
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=block_mask,
            ).sum()

        with patch(
            "torchtitan.models.common.attention.FlexInnerAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            remat.checkpoint(region_name="attention")(forward)(q_THK).backward()

        self.assertEqual(len(attention.max_attention_logits_H), 1)
        torch.testing.assert_close(
            attention.max_attention_logits_H[0],
            torch.tensor([3.0, 4.0]),
        )


class _RecordingStage(nn.Module):
    """Pipeline stage stand-in that records its input's per-column max as head maxima."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.inner_attention = _qk_clip_config().build()

    def forward(self, x_TH: torch.Tensor) -> torch.Tensor:
        max_scores_1HT = x_TH.detach().amax(dim=0)[None, :, None]
        self.inner_attention._process_aux(SimpleNamespace(max_scores=max_scores_1HT))
        return self.proj(x_TH)


class QKClipSingleRankTest(unittest.TestCase):
    """One gloo rank: dp * cp == 1, as in PP-only or one-GPU runs."""

    def setUp(self) -> None:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_optimizer_hook_runs_without_loss_mesh(self) -> None:
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        # One QK-clip layer with one head. FSDP leaves its weights as DTensors
        # on the size-1 dp_shard mesh.
        attention = DeepSeekMLAAttention.__new__(DeepSeekMLAAttention)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 1
        attention.qk_nope_head_dim = 1
        attention.qk_head_dim = 2
        attention.v_head_dim = 1
        attention.wq_b = nn.Linear(1, 2, bias=False)
        attention.inner_attention = _qk_clip_mla_attention()
        attention.inner_attention.wkv_b = nn.Linear(1, 2, bias=False)
        mesh = parallelism_context.get_mesh("dp_shard")
        for projection in (attention.wq_b, attention.inner_attention.wkv_b):
            projection.weight = nn.Parameter(
                distribute_tensor(projection.weight.detach(), mesh, (Shard(0),))
            )
        max_logits_H = attention.inner_attention.inner_attention.max_attention_logits_H
        max_logits_H.append(torch.tensor([400.0]))
        optimizers = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=0.0,
                    weight_decay=0.0,
                    fused=False,
                )
            ],
        ).build(model_parts=[attention])

        # dp * cp == 1 leaves the "loss" mesh inactive: the hook must work without it.
        register_qk_clip_hook(optimizers, [attention], parallelism_context)
        for param in attention.parameters():
            param.grad = torch.zeros_like(param)
        optimizers.step()

        # qk_clip skipped the all-reduce and consumed the recorded maximum.
        self.assertFalse(max_logits_H)

    def test_pipeline_shape_inference_maxima_are_dropped(self) -> None:
        # Two stages on one rank. The first step runs one extra forward per stage
        # to learn shapes; stage 1 runs it on uninitialized memory.
        stage_modules = [_RecordingStage(), _RecordingStage()]
        schedule = ScheduleInterleaved1F1B(
            [
                PipelineStage(module, stage_index, 2, torch.device("cpu"))
                for stage_index, module in enumerate(stage_modules)
            ],
            n_microbatches=2,
            loss_fn=lambda output, target: output.sum(),
        )
        register_qk_clip_shape_inference_cleanup(schedule, stage_modules)

        schedule.step(torch.randn(4, 2), target=torch.zeros(4, 2))

        # One maximum per microbatch, none from the shape-inference forward.
        for module in stage_modules:
            self.assertEqual(len(module.inner_attention.max_attention_logits_H), 2)


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

        attention = DeepSeekMLAAttention.__new__(DeepSeekMLAAttention)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 1
        attention.qk_nope_head_dim = qk_nope_head_dim
        attention.qk_rope_head_dim = qk_rope_head_dim
        attention.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        attention.v_head_dim = v_head_dim
        attention.wq_b = weight_module(num_heads * attention.qk_head_dim)
        attention.inner_attention = _qk_clip_mla_attention()
        attention.inner_attention.wkv_b = weight_module(
            num_heads * (qk_nope_head_dim + v_head_dim)
        )
        rank_maxima = (
            torch.tensor([50.0, 400.0], device=device)
            if self.rank == 0
            else torch.tensor([200.0, 50.0], device=device)
        )
        attention.inner_attention.inner_attention.max_attention_logits_H.append(
            rank_maxima
        )
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
        kv_weight_HDI = attention.inner_attention.wkv_b.weight.to_local().view(
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
        self.assertFalse(
            attention.inner_attention.inner_attention.max_attention_logits_H
        )

    @with_comms
    def test_weight_scaling_is_communication_free(self) -> None:
        """Per-head scaling must stay local for every clipped weight.

        Building the replicated scales with ``distribute_tensor`` instead of
        ``from_local`` would broadcast once per weight per step, so this asserts
        the packed maxima all-reduce is the only collective ``qk_clip`` issues.
        The head count is not a multiple of the shard count, so ``Shard(0)``
        splits a head across ranks.
        """
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        num_heads = self.world_size + 1
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
            attention = DeepSeekMLAAttention.__new__(DeepSeekMLAAttention)
            nn.Module.__init__(attention)
            attention.q_lora_rank = 1
            attention.qk_nope_head_dim = qk_nope_head_dim
            attention.qk_rope_head_dim = qk_rope_head_dim
            attention.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            attention.v_head_dim = v_head_dim
            attention.wq_b = weight_module(num_heads * attention.qk_head_dim)
            attention.inner_attention = _qk_clip_mla_attention()
            attention.inner_attention.wkv_b = weight_module(
                num_heads * (qk_nope_head_dim + v_head_dim)
            )
            # Head h has max logit 400 * 4**h, so a head given another head's
            # scale fails the value check below.
            attention.inner_attention.inner_attention.max_attention_logits_H.append(
                400.0 * 4.0 ** torch.arange(num_heads, device=device)
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
        # Head h gets the scale 100 / (400 * 4**h) = 4**-(h + 1), and
        # alpha = 0.5 makes its NoPE scale 2**-(h + 1).
        q_row_scales, kv_row_scales = [], []
        for h in range(num_heads):
            q_row_scales += [2.0 ** -(h + 1)] * qk_nope_head_dim
            q_row_scales += [4.0 ** -(h + 1)] * qk_rope_head_dim
            kv_row_scales += [2.0 ** -(h + 1)] * qk_nope_head_dim + [1.0] * v_head_dim
        for weight, row_scales in (
            (attention.wq_b.weight, q_row_scales),
            (attention.inner_attention.wkv_b.weight, kv_row_scales),
        ):
            torch.testing.assert_close(
                weight.full_tensor(),
                torch.tensor(row_scales, device=device)[:, None].expand(
                    -1, in_features
                ),
            )
