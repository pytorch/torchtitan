# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.config import DebugConfig, ParallelismConfig
from torchtitan.distributed.context_parallel import (
    _HeadTailLoadBalancer,
    HeadTailCPLoadBalancer,
)
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    LinearAttentionMetadata,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
)
from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.attn_gym import (
    causal_conv1d,
    context_parallel_conv_history,
    context_parallel_gdn,
    context_parallel_kda,
    ContextParallelRouting,
    is_missing_kernel,
    l2norm,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding

pytestmark = pytest.mark.multi_gpu


@unittest.skipUnless(
    torch.cuda.device_count() >= 2
    and torch.cuda.get_device_capability() >= (9, 0)
    and not is_missing_kernel(ContextParallelRouting),
    "requires two CUDA devices and Attention Gym",
)
class TestLinearAttentionCPRouting(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_routing_preserves_loss_gradients_and_updates(self):
        config = DebugConfig(seed=42, deterministic=True)
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(config.deterministic)
        try:
            for family in ("kda", "gdn"):
                for headtail in (False, True):
                    with self.subTest(family=family, headtail=headtail):
                        self._compare_routes(family, headtail)
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

    def _compare_routes(self, family, headtail):
        torch.manual_seed(42)
        device = torch.device(self.device_type, self.rank)
        # Packed documents include fragment boundaries inside a 64-token chunk.
        positions = torch.cat([torch.arange(n) for n in (73, 119, 64)])
        permutation = (
            _HeadTailLoadBalancer(256, self.world_size, "cpu")._generate_indices()
            if headtail
            else None
        )
        fragments = (
            [[(0, 64), (192, 256)], [(64, 192)]]
            if headtail
            else [[(0, 128)], [(128, 256)]]
        )
        if permutation is not None:
            with patch(
                "torchtitan.distributed.context_parallel.spmd_mesh_group",
                return_value=dist.group.WORLD,
            ), patch(
                "torchtitan.distributed.context_parallel.current_spmd_mesh",
                return_value=SimpleNamespace(device_type=self.device_type),
            ):
                balancer = HeadTailCPLoadBalancer.Config().build(
                    seq_len=256, attention_metadata=None
                )
            layout_fragments = balancer.get_token_fragments(
                permutation.to(device), cp_size=self.world_size
            )
            self.assertEqual(layout_fragments, fragments)
        microbatch = TokenizedTrainingMicrobatch(
            input=torch.arange(256),
            labels=torch.arange(256),
            positions=positions,
            padding_mask=torch.zeros(256, dtype=torch.bool),
            loss_token_counts=torch.tensor(256),
            routing_token_counts=torch.tensor([256]),
        )
        batch = microbatch.to_input_dict(
            device, non_blocking=True, include_cu_seqlens=True
        )
        varlen = create_varlen_metadata_for_document(
            batch["positions"], max_num_documents=8, max_context_length=256
        )
        reference = ContextParallelRouting.from_fragments(
            varlen.cu_seq_q.tolist(), fragments, self.rank, device, conv_history=3
        )
        backend = (
            ContextParallelInnerKDA if family == "kda" else ContextParallelInnerGDN
        )
        batch["attention_metadata"] = {
            backend: LinearAttentionMetadata(varlen=varlen, num_conv_history_tokens=3)
        }
        model = SimpleNamespace(
            config=SimpleNamespace(
                first_full_attention=SimpleNamespace(
                    inner_attention=AllGatherKVFlexInnerAttention.Config()
                )
            )
        )
        with patch(
            "torchtitan.models.common.attention.cp_linear_attention.spmd_mesh_group",
            return_value=dist.group.WORLD,
        ), patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=dist.group.WORLD,
        ), patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=SimpleNamespace(device_type=self.device_type),
        ), patch(
            "torchtitan.distributed.context_parallel.shard_tensors",
            side_effect=lambda x, **_: x,
        ):
            prepared = Decoder._cp_shard(
                model,
                batch,
                input_shardings=decoder_input_sharding(),
                parallelism_context=SimpleNamespace(cp=self.world_size),
                parallelism=ParallelismConfig(
                    context_parallel_degree=self.world_size,
                    context_parallel_load_balancer=(
                        HeadTailCPLoadBalancer.Config() if headtail else None
                    ),
                ),
            )
        optimized = prepared["attention_metadata"][backend].cp_routing

        gate_shape = (1, 128, 2, 128) if family == "kda" else (1, 128, 2)
        parameters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
                )
                for shape in ((1, 128, 768), (768, 4), gate_shape, (1, 128, 2))
            ]
        )
        parameters_new = copy.deepcopy(parameters)
        optimizers = [
            torch.optim.AdamW(p, lr=1e-3) for p in (parameters, parameters_new)
        ]
        kernel = context_parallel_kda if family == "kda" else context_parallel_gdn
        for _ in range(3):
            results = []
            for params, routing, optimizer in zip(
                (parameters, parameters_new), (reference, optimized), optimizers
            ):
                optimizer.zero_grad(set_to_none=True)
                qkv, conv_weight, raw_gate, raw_beta = params
                history = context_parallel_conv_history(qkv, routing, dist.group.WORLD)
                convolved = causal_conv1d(
                    qkv,
                    conv_weight,
                    activation="silu",
                    initial_state=history,
                    cu_seqlens=routing.cu_seqlens,
                )
                q, k, v = [
                    t.reshape(1, 128, 2, 128) for t in convolved.chunk(3, dim=-1)
                ]
                out, _ = kernel(
                    l2norm(q, autotune=False),
                    l2norm(k, autotune=False),
                    v,
                    -F.softplus(raw_gate.float()),
                    raw_beta.float().sigmoid(),
                    routing=routing,
                    group=dist.group.WORLD,
                )
                loss = out.float().square().mean()
                loss.backward()
                grads = [p.grad.detach().clone() for p in params]
                grad_norm = (
                    torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
                )
                results.append((out.detach(), loss.detach(), grad_norm, grads))
                optimizer.step()
            for before, after in zip(results[0][:3], results[1][:3]):
                self.assertTrue(torch.equal(before, after))
            for before, after in zip(results[0][3], results[1][3]):
                self.assertTrue(torch.equal(before, after))
            for before, after in zip(parameters, parameters_new):
                self.assertTrue(torch.equal(before, after))
