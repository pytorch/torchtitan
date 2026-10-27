# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import pytest
import spmd_types as spmd
import torch
import torch_remat as remat
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import DeterministicGuard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.common.attention.cp_mla import (
    AllGatherKVMLAFlexAttention,
    UlyssesMLAAttention,
)
from torchtitan.models.common.attention.mla import materialize_mla_kv
from torchtitan.models.common.decoder_sharding import token_id_placement
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan_recipes.overrides.fused_mla import _materialize_cp_mla_kv, fused_mla


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
def test_cp_materializer_traces_forward_and_backward():
    with DeterministicGuard(True):
        torch.manual_seed(42)
        q_THK = torch.randn(33, 8, 32, device="cuda", dtype=torch.bfloat16)
        kv_THP = torch.randn_like(q_THK).requires_grad_()
        k_shared_TR = torch.randn(
            33, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad_k_THK = torch.randn_like(q_THK)
        grad_v_THV = torch.randn(33, 8, 16, device="cuda", dtype=torch.bfloat16)

        def run(q_THK, kv_THP, k_shared_TR, grad_k_THK, grad_v_THV):
            k_THK, v_THV = _materialize_cp_mla_kv(q_THK, kv_THP, k_shared_TR)
            gradients = torch.autograd.grad(
                (k_THK, v_THV), (kv_THP, k_shared_TR), (grad_k_THK, grad_v_THV)
            )
            return k_THK, v_THV, *gradients

        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        expected = (
            k_THK,
            v_THV,
            *torch.autograd.grad(
                (k_THK, v_THV), (kv_THP, k_shared_TR), (grad_k_THK, grad_v_THV)
            ),
        )
        inputs = (q_THK, kv_THP, k_shared_TR, grad_k_THK, grad_v_THV)
        graph = make_fx(run, tracing_mode="fake")(*inputs)
        # The joint graph already contains backward, as in GraphTrainer.
        with torch.no_grad():
            actual = torch.compile(graph, fullgraph=True)(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        targets = {
            node.target for node in graph.graph.nodes if node.op == "call_function"
        }
        assert torch.ops.torchtitan.fused_mla_k_rope.default in targets
        assert torch.ops.torchtitan.fused_mla_kv_backward.default in targets


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestFusedMLACP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_cp_outputs_and_gradients_match_stock(self):
        with (
            DeterministicGuard(True),
            torch.autograd.set_multithreading_enabled(False),
        ):
            context = ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=2,
                tp=1,
                pp=1,
                ep=1,
                world_size=self.world_size,
                enable_sequence_parallel=False,
            )
            context.build_mesh()
            for backend, load_balance in (
                (AllGatherKVMLAFlexAttention, False),
                (AllGatherKVMLAFlexAttention, True),
                (UlyssesMLAAttention, False),
            ):
                with self.subTest(backend=backend.__name__, load_balance=load_balance):
                    self._check_backend(context, backend, load_balance)

    def _check_backend(self, context, backend, load_balance):
        torch.manual_seed(42)
        device = torch.device("cuda", self.rank)
        parallelism = ParallelismConfig(
            context_parallel_degree=2, enable_sequence_parallel=False
        )
        model_config = build_model_config(
            "debugmodel", parallelism=parallelism, seq_len=512, attn_backend="flex"
        )
        ContextParallelTransform(inner_attention_map={MLAAttention: backend}).transform(
            model_config
        )
        stock_config = model_config.layers[0].attention
        stock = stock_config.build().to(device)
        fused = fused_mla(copy.deepcopy(stock_config)).build().to(device)
        stock.init_states(buffer_device=device)
        fused.init_states(buffer_device=device)
        for module in (stock, fused):
            for parameter in module.parameters():
                parameter.data = parameter.data.to(torch.bfloat16)
        fused.load_state_dict(stock.state_dict(), strict=True)
        for module in (stock, fused):
            module._parallelize(context)
            module.configure_remat_regions((), ())

        # A document crosses the CP boundary; positions restart in the next one.
        positions_T = torch.cat(
            (torch.arange(320, device=device), torch.arange(192, device=device))
        )
        inputs_TD = torch.randn(
            512, stock_config.dim, device=device, dtype=torch.bfloat16
        )
        grad_TD = torch.randn_like(inputs_TD)
        with context.activate_spmd():
            metadata = stock_config.inner_attention.build_attention_metadata(
                positions_T
            )
            permutation = None
            if load_balance:
                permutation = (
                    HeadTailCPLoadBalancer.Config()
                    .build(seq_len=512, attention_metadata=metadata)
                    .generate_permutation()
                )
                inputs_TD = inputs_TD[permutation[0]]
                positions_T = positions_T[permutation[0]]
                grad_TD = grad_TD[permutation[0]]
            metadata = backend.prepare_cp_metadata(metadata, permutation=permutation)

        local_positions_T = positions_T.chunk(2)[self.rank].contiguous()
        local_inputs_TD = inputs_TD.chunk(2)[self.rank].contiguous()
        local_grad_TD = grad_TD.chunk(2)[self.rank].contiguous()
        layout = stock_config.sharding_config.in_src_shardings["x"]

        def run(module, checkpoint):
            module.zero_grad(set_to_none=True)
            x_TD = local_inputs_TD.detach().clone().requires_grad_()
            cotangent_TD = local_grad_TD.clone()
            with context.activate_spmd(typechecking=True):
                spmd.assert_type(x_TD, layout)
                spmd.assert_type(local_positions_T, token_id_placement())
                forward = remat.checkpoint()(module) if checkpoint else module
                output_TD = forward(
                    x_TD, attention_metadata=metadata, positions=local_positions_T
                )
                spmd.assert_type(cotangent_TD, layout)
                output_TD.backward(cotangent_TD)
            gradients = [x_TD.grad, *(p.grad for p in module.parameters())]
            self.assertTrue(all(grad is not None for grad in gradients))
            return output_TD.detach(), [grad.detach().clone() for grad in gradients]

        options = dict(FlexInnerAttention.inductor_configs)
        try:
            FlexInnerAttention.inductor_configs["max_autotune"] = False
            FlexInnerAttention.inductor_configs["coordinate_descent_tuning"] = False
            expected = run(stock, checkpoint=False)
            actual = run(fused, checkpoint=False)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            # Replaying fused computation must preserve every bit.
            replayed = run(fused, checkpoint=True)
            torch.testing.assert_close(replayed, actual, rtol=0, atol=0)
        finally:
            FlexInnerAttention.inductor_configs.clear()
            FlexInnerAttention.inductor_configs.update(options)
