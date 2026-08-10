# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from typing import cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
from torchtitan.models.gpt_oss import model_registry as gptoss_model_registry
from torchtitan.models.gpt_oss.model import GptOssModel
from torchtitan.models.kimi_k2_7 import model_registry as kimi_model_registry
from torchtitan.models.kimi_k2_7.model import KimiK25Model


_EP_WORLD_SIZE = 2
_EP_NUM_EXPERTS = 4
_EP_TOP_K = 2
_EP_DIM = 16
_EP_INPUTS = (
    torch.tensor(
        [
            [1.0, 1.5] + [0.0] * (_EP_DIM - 2),
            [2.0, 2.5] + [0.0] * (_EP_DIM - 2),
            [3.0, 3.5] + [0.0] * (_EP_DIM - 2),
        ],
        dtype=torch.bfloat16,
    ),
    torch.tensor(
        [
            [11.0, 11.5] + [0.0] * (_EP_DIM - 2),
            [12.0, 12.5] + [0.0] * (_EP_DIM - 2),
            [13.0, 13.5] + [0.0] * (_EP_DIM - 2),
        ],
        dtype=torch.bfloat16,
    ),
)
_EP_SCORES = (
    torch.tensor([[0.25, 0.5], [0.75, 1.0], [0.125, 0.625]]),
    torch.tensor([[0.375, 0.875], [0.625, 0.25], [0.5, 0.75]]),
)
_EP_EXPERT_IDS = (
    torch.tensor([[0, 2], [0, 3], [1, 2]]),
    torch.tensor([[2, 3], [1, 2], [3, 1]]),
)


def _grad_float(tensor: torch.Tensor) -> torch.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad.float()


def _gloo_score_transport_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=_EP_WORLD_SIZE,
        rank=rank,
    )
    try:
        mesh = init_device_mesh("cpu", (_EP_WORLD_SIZE,), mesh_dim_names=("ep",))
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=_EP_NUM_EXPERTS, top_k=_EP_TOP_K)
        )
        dispatcher.wire_meshes(ep_mesh=mesh, tp_mesh=None)
        x_TD = _EP_INPUTS[rank].clone().requires_grad_()
        scores_TK = _EP_SCORES[rank].clone().requires_grad_()
        expert_ids_TK = _EP_EXPERT_IDS[rank]
        counts_E = torch.bincount(expert_ids_TK.reshape(-1), minlength=_EP_NUM_EXPERTS)

        routed_input_RD, counts_e, metadata, routed_scores_R = dispatcher.dispatch(
            x_TD,
            scores_TK,
            expert_ids_TK,
            counts_E,
            absorb_router_scores=True,
        )
        assert routed_scores_R is not None

        local_experts = range(
            rank * (_EP_NUM_EXPERTS // _EP_WORLD_SIZE),
            (rank + 1) * (_EP_NUM_EXPERTS // _EP_WORLD_SIZE),
        )
        expected_input_rows = []
        expected_score_rows = []
        expected_counts = []
        for expert_id in local_experts:
            expert_rows = []
            expert_scores = []
            for source_rank in range(_EP_WORLD_SIZE):
                for token_index in range(_EP_INPUTS[source_rank].shape[0]):
                    for topk_index in range(_EP_TOP_K):
                        if (
                            _EP_EXPERT_IDS[source_rank][token_index, topk_index]
                            == expert_id
                        ):
                            expert_rows.append(_EP_INPUTS[source_rank][token_index])
                            expert_scores.append(
                                _EP_SCORES[source_rank][token_index, topk_index]
                            )
            expected_input_rows.extend(expert_rows)
            expected_score_rows.extend(expert_scores)
            expected_counts.append(len(expert_rows))

        expected_input_RD = torch.stack(expected_input_rows)
        expected_scores_R = torch.stack(expected_score_rows).bfloat16()
        torch.testing.assert_close(routed_input_RD, expected_input_RD)
        torch.testing.assert_close(routed_scores_R, expected_scores_R)
        torch.testing.assert_close(counts_e, torch.tensor(expected_counts))

        expert_scale_R = torch.tensor(
            [
                expert_id + 1
                for expert_id in local_experts
                for _ in range(expected_counts[expert_id - rank * 2])
            ],
            dtype=torch.bfloat16,
        ).reshape(-1, 1)
        routed_output_RD = (
            routed_input_RD * expert_scale_R * routed_scores_R.reshape(-1, 1)
        )
        actual_TD = dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
            num_local_tokens_after_padding=3,
            local_seq_len_after_padding=3,
            router_scores_applied=True,
        )

        expected_output_TD = torch.zeros_like(x_TD)
        for token_index in range(x_TD.shape[0]):
            for topk_index in range(_EP_TOP_K):
                expert_id = int(expert_ids_TK[token_index, topk_index])
                score = scores_TK[token_index, topk_index]
                expected_output_TD[token_index] += (
                    x_TD[token_index] * (expert_id + 1) * score
                )
        torch.testing.assert_close(actual_TD, expected_output_TD, rtol=0.03, atol=0.03)

        loss_weight_TD = torch.arange(1, 3 * _EP_DIM + 1, dtype=torch.float32).reshape(
            3, _EP_DIM
        )
        actual_loss = (actual_TD.float() * loss_weight_TD).sum()
        actual_loss.backward()
        expected_score_grad = torch.zeros_like(scores_TK)
        for token_index in range(x_TD.shape[0]):
            for topk_index in range(_EP_TOP_K):
                expert_id = int(expert_ids_TK[token_index, topk_index])
                expected_score_grad[token_index, topk_index] = (
                    x_TD[token_index].float()
                    * (expert_id + 1)
                    * loss_weight_TD[token_index]
                ).sum()
        torch.testing.assert_close(
            scores_TK.grad, expected_score_grad, rtol=0.03, atol=0.03
        )
    finally:
        dist.destroy_process_group()


class TestRouterScoreAbsorption(unittest.TestCase):
    def test_deepseek_v3_enables_router_score_absorption(self):
        model_config = cast(DeepSeekV3Model.Config, model_registry("debugmodel").model)
        routed_expert_configs = [
            layer.moe.routed_experts
            for layer in model_config.layers
            if layer.moe is not None
        ]

        self.assertTrue(routed_expert_configs)
        self.assertTrue(
            all(config.absorb_router_scores for config in routed_expert_configs)
        )

    def test_kimi_keeps_post_combine_router_scoring(self):
        model_config = cast(
            KimiK25Model.Config, kimi_model_registry("debugmodel").model
        )
        routed_expert_configs = [
            layer.moe.routed_experts
            for layer in model_config.layers
            if layer.moe is not None
        ]

        self.assertTrue(routed_expert_configs)
        self.assertTrue(
            all(not config.absorb_router_scores for config in routed_expert_configs)
        )

    def test_gpt_oss_keeps_post_combine_router_scoring(self):
        model_config = cast(
            GptOssModel.Config, gptoss_model_registry("debugmodel").model
        )
        routed_expert_configs = [
            layer.moe.routed_experts
            for layer in model_config.layers
            if layer.moe is not None
        ]

        self.assertTrue(routed_expert_configs)
        self.assertTrue(
            all(not config.absorb_router_scores for config in routed_expert_configs)
        )

    def test_routed_experts_passes_router_score_mode_to_consumers(self):
        torch.manual_seed(42)
        common_kwargs = dict(
            dim=16,
            hidden_dim=16,
            num_experts=2,
            top_k=2,
            param_init={},
            comm_backend="standard",
        )
        enabled = make_routed_experts_config(
            **common_kwargs, absorb_router_scores=True
        ).build()
        disabled = make_routed_experts_config(
            **common_kwargs, absorb_router_scores=False
        ).build()
        with torch.no_grad():
            for parameter in enabled.parameters():
                parameter.normal_(std=0.1)
            disabled.load_state_dict(enabled.state_dict())

        x_BLD = torch.randn(1, 2, 16, dtype=torch.bfloat16)
        scores_BLK = torch.tensor([[[0.25, 0.75], [0.5, 1.0]]], dtype=torch.float32)
        expert_ids_BLK = torch.tensor([[[0, 1], [1, 0]]])
        counts_E = torch.tensor([2, 2])

        def run_and_record(module):
            dispatch_calls = []
            combine_calls = []
            expert_calls = []

            original_dispatch = module.token_dispatcher.dispatch
            original_combine = module.token_dispatcher.combine

            def record_dispatch(*args, **kwargs):
                dispatch_calls.append(kwargs.copy())
                return original_dispatch(*args, **kwargs)

            def record_combine(*args, **kwargs):
                combine_calls.append(kwargs.copy())
                return original_combine(*args, **kwargs)

            def record_expert(_module, args, kwargs):
                expert_calls.append(kwargs.copy())

            hook = module.inner_experts.register_forward_pre_hook(
                record_expert, with_kwargs=True
            )
            try:
                with patch.object(
                    module.token_dispatcher, "dispatch", side_effect=record_dispatch
                ), patch.object(
                    module.token_dispatcher, "combine", side_effect=record_combine
                ):
                    output_BLD = module(
                        x_BLD,
                        scores_BLK,
                        expert_ids_BLK,
                        counts_E,
                        num_local_tokens_after_seq_dim_padding=2,
                    )
            finally:
                hook.remove()
            return output_BLD, dispatch_calls, expert_calls, combine_calls

        _, enabled_dispatch, enabled_expert, enabled_combine = run_and_record(enabled)
        _, disabled_dispatch, disabled_expert, disabled_combine = run_and_record(
            disabled
        )

        self.assertEqual(enabled_dispatch[0]["absorb_router_scores"], True)
        self.assertIn("routed_scores_R", enabled_expert[0])
        self.assertEqual(enabled_expert[0]["routed_scores_R"].shape, (4,))
        self.assertEqual(enabled_combine[0]["router_scores_applied"], True)
        self.assertEqual(disabled_dispatch[0]["absorb_router_scores"], False)
        self.assertIsNone(disabled_expert[0]["routed_scores_R"])
        self.assertEqual(disabled_combine[0]["router_scores_applied"], False)

    def test_local_dispatch_returns_expert_aligned_scores(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=3, top_k=2)
        )
        x_TD = torch.arange(48, dtype=torch.bfloat16).reshape(3, 16)
        topk_scores_TK = torch.tensor(
            [[0.25, 0.5], [0.75, 1.0], [0.125, 0.625]], requires_grad=True
        )
        topk_expert_ids_TK = torch.tensor([[2, 0], [1, 2], [0, 1]])
        num_tokens_per_expert_E = torch.bincount(
            topk_expert_ids_TK.reshape(-1), minlength=3
        )

        routed_input_RD, _, _, routed_scores_R = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
            absorb_router_scores=True,
        )

        self.assertIsNotNone(routed_scores_R)
        expert_order_N = torch.argsort(topk_expert_ids_TK.reshape(-1), stable=True)
        expected_token_indices_N = expert_order_N // 2
        torch.testing.assert_close(routed_input_RD, x_TD[expected_token_indices_N])
        torch.testing.assert_close(
            routed_scores_R, topk_scores_TK.reshape(-1)[expert_order_N]
        )

        routed_scores_R.sum().backward()
        torch.testing.assert_close(topk_scores_TK.grad, torch.ones_like(topk_scores_TK))

    def test_all_to_all_dispatch_keeps_scores_aligned_across_two_gloo_ranks(self):
        with tempfile.NamedTemporaryFile() as init_file:
            mp.start_processes(
                _gloo_score_transport_worker,
                args=(init_file.name,),
                nprocs=_EP_WORLD_SIZE,
                start_method="fork",
                join=True,
            )

    def test_score_free_combine_matches_independent_scatter_reference(self):
        torch.manual_seed(42)
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=2, top_k=2)
        )
        x_TD = torch.zeros(3, 16, dtype=torch.bfloat16)
        topk_scores_TK = torch.tensor([[0.25, 0.5], [0.75, 1.0], [0.125, 0.625]])
        topk_expert_ids_TK = torch.tensor([[0, 1], [1, 0], [0, 1]])
        num_tokens_per_expert_E = torch.tensor([3, 3])

        _, _, absorbed_metadata, routed_scores_R = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
            absorb_router_scores=True,
        )
        self.assertIsNotNone(routed_scores_R)
        routed_output_RD = torch.randn(6, 16, dtype=torch.bfloat16)
        preweighted_output_RD = (
            routed_output_RD.float() * routed_scores_R.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        expert_order_N = torch.argsort(topk_expert_ids_TK.reshape(-1), stable=True)
        expected_TD = torch.zeros_like(x_TD)
        for routed_row, token_index in zip(preweighted_output_RD, expert_order_N // 2):
            expected_TD[token_index] += routed_row
        actual_TD = dispatcher.combine(
            preweighted_output_RD,
            absorbed_metadata,
            x_TD,
            num_local_tokens_after_padding=3,
            local_seq_len_after_padding=3,
            router_scores_applied=True,
        )
        torch.testing.assert_close(actual_TD, expected_TD)

    def test_minimal_async_ep_declines_pre_w2_router_scores(self):
        class FakeGroup:
            def size(self):
                return 1

        class FakeMesh:
            def size(self):
                return 1

            def get_group(self):
                return FakeGroup()

        dispatcher = MinimalAsyncEPTokenDispatcher(
            MinimalAsyncEPTokenDispatcher.Config(
                num_experts=2,
                top_k=2,
                hidden_dim=16,
                tokens_per_rank=2,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
        )
        dispatcher.ep_mesh = cast(DeviceMesh, FakeMesh())
        x_TD = torch.zeros(2, 16, dtype=torch.bfloat16)
        scores_TK = torch.tensor([[0.25, 0.75], [0.5, 1.0]])
        expert_ids_TK = torch.tensor([[0, 1], [1, 0]])
        counts_E = torch.tensor([2, 2])
        dispatch_result = (
            torch.zeros(4, 16, dtype=torch.bfloat16),
            torch.tensor([2, 2]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor(4),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([2, 2]),
        )
        with patch(
            "torchtitan.models.common.token_dispatcher.minimal_async_ep_dispatch_op",
            return_value=dispatch_result,
        ):
            _, _, _, routed_scores_R = dispatcher.dispatch(
                x_TD,
                scores_TK,
                expert_ids_TK,
                counts_E,
                absorb_router_scores=True,
            )

        self.assertIsNone(routed_scores_R)

    def test_routed_experts_matches_postweighted_reference(self):
        torch.manual_seed(42)
        base = make_routed_experts_config(
            dim=16,
            hidden_dim=16,
            num_experts=2,
            top_k=2,
            param_init={},
            comm_backend="standard",
        ).build()
        absorbed = make_routed_experts_config(
            dim=16,
            hidden_dim=16,
            num_experts=2,
            top_k=2,
            param_init={},
            comm_backend="standard",
            absorb_router_scores=True,
        ).build()
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(std=0.1)
            absorbed.load_state_dict(base.state_dict())

        x_base_BLD = torch.randn(1, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        x_absorbed_BLD = x_base_BLD.detach().clone().requires_grad_()
        scores_base_BLK = torch.tensor(
            [[[0.25, 0.75], [0.5, 1.0], [0.125, 0.625], [0.875, 0.375]]],
            requires_grad=True,
        )
        scores_absorbed_BLK = scores_base_BLK.detach().clone().requires_grad_()
        expert_ids_BLK = torch.tensor([[[0, 1], [1, 0], [0, 1], [1, 0]]])
        num_tokens_per_expert_E = torch.tensor([4, 4])

        expected_BLD = base(
            x_base_BLD,
            scores_base_BLK,
            expert_ids_BLK,
            num_tokens_per_expert_E,
            num_local_tokens_after_seq_dim_padding=4,
        )
        actual_BLD = absorbed(
            x_absorbed_BLD,
            scores_absorbed_BLK,
            expert_ids_BLK,
            num_tokens_per_expert_E,
            num_local_tokens_after_seq_dim_padding=4,
        )

        torch.testing.assert_close(actual_BLD, expected_BLD, rtol=0.02, atol=0.02)

        expected_BLD.float().sum().backward()
        actual_BLD.float().sum().backward()
        torch.testing.assert_close(
            scores_absorbed_BLK.grad,
            scores_base_BLK.grad,
            rtol=0.02,
            atol=0.02,
        )
        torch.testing.assert_close(
            x_absorbed_BLD.grad,
            x_base_BLD.grad,
            rtol=0.02,
            atol=0.02,
        )

    def test_grouped_experts_matches_dense_reference_forward_and_backward(self):
        torch.manual_seed(42)
        experts = GroupedExperts.Config(
            dim=16,
            hidden_dim=16,
            num_experts=2,
        ).build()
        with torch.no_grad():
            for parameter in experts.parameters():
                parameter.normal_(std=0.1)

        x_RD = torch.randn(3, 16, dtype=torch.bfloat16, requires_grad=True)
        scores_R = torch.tensor([0.25, 0.5, 0.75], requires_grad=True)
        counts_E = torch.tensor([2, 1])
        loss_weight_RD = torch.arange(1, 49, dtype=torch.float32).reshape(3, 16)

        output_RD = experts(x_RD, counts_E, routed_scores_R=scores_R)
        output_loss = (output_RD.float() * loss_weight_RD).sum()

        x_ref_RD = x_RD.detach().clone().requires_grad_()
        scores_ref_R = scores_R.detach().float().requires_grad_()
        reference_weights = [
            parameter.detach().bfloat16().requires_grad_()
            for parameter in (experts.w1_EFD, experts.w2_EDF, experts.w3_EFD)
        ]
        w1_EFD, w2_EDF, w3_EFD = reference_weights
        reference_rows = []
        row_start = 0
        for expert_index, count in enumerate(counts_E.tolist()):
            rows = x_ref_RD[row_start : row_start + count]
            hidden = F.silu(rows @ w1_EFD[expert_index].transpose(-2, -1))
            hidden = hidden * (rows @ w3_EFD[expert_index].transpose(-2, -1))
            hidden = hidden * scores_ref_R[
                row_start : row_start + count
            ].bfloat16().reshape(-1, 1)
            reference_rows.append(hidden @ w2_EDF[expert_index].transpose(-2, -1))
            row_start += count
        reference_RD = torch.cat(reference_rows).bfloat16()
        reference_loss = (reference_RD.float() * loss_weight_RD).sum()

        torch.testing.assert_close(output_RD, reference_RD, rtol=0.03, atol=0.03)
        output_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(
            _grad_float(x_RD), _grad_float(x_ref_RD), rtol=0.03, atol=0.03
        )
        torch.testing.assert_close(
            scores_R.grad, scores_ref_R.grad, rtol=0.03, atol=0.03
        )
        for parameter, reference_parameter in (
            (experts.w1_EFD, w1_EFD),
            (experts.w2_EDF, w2_EDF),
            (experts.w3_EFD, w3_EFD),
        ):
            torch.testing.assert_close(
                _grad_float(parameter),
                _grad_float(reference_parameter),
                rtol=0.03,
                atol=0.03,
            )

    def test_score_free_combine_does_not_save_post_w2_output(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=2, top_k=2)
        )
        x_TD = torch.zeros(2, 16, dtype=torch.bfloat16)
        topk_scores_TK = torch.tensor([[0.25, 0.75], [0.5, 1.0]])
        topk_expert_ids_TK = torch.tensor([[0, 1], [1, 0]])
        num_tokens_per_expert_E = torch.tensor([2, 2])

        def saved_combine_tensors(*, router_scores_applied: bool):
            scores_TK = topk_scores_TK.detach().clone().requires_grad_()
            _, _, metadata, _ = dispatcher.dispatch(
                x_TD,
                scores_TK,
                topk_expert_ids_TK,
                num_tokens_per_expert_E,
            )
            routed_output_RD = torch.randn(
                4, 16, dtype=torch.bfloat16, requires_grad=True
            )
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: saved.append(tensor) or tensor,
                lambda tensor: tensor,
            ):
                combined_TD = dispatcher.combine(
                    routed_output_RD,
                    metadata,
                    x_TD,
                    num_local_tokens_after_padding=2,
                    local_seq_len_after_padding=2,
                    router_scores_applied=router_scores_applied,
                )
            combined_TD.float().sum().backward()
            return saved, scores_TK.grad

        base_saved, base_score_grad = saved_combine_tensors(router_scores_applied=False)
        absorbed_saved, absorbed_score_grad = saved_combine_tensors(
            router_scores_applied=True
        )

        self.assertTrue(
            any(
                tensor.shape == (4, 16) and tensor.dtype == torch.float32
                for tensor in base_saved
            )
        )
        self.assertFalse(
            any(
                tensor.shape == (4, 16) and tensor.dtype == torch.float32
                for tensor in absorbed_saved
            )
        )
        self.assertIsNotNone(base_score_grad)
        self.assertIsNone(absorbed_score_grad)


if __name__ == "__main__":
    unittest.main()
