# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for auxiliary losses, including MoE load-balance variants.

The single-process cases check the loss value, the injected gradient and the
metric register against an explicit Eqs 17-20 reference; the 8-rank cases
(dp2/cp2/tp2/ep2, CPU float64 + gloo) check that the per-DP-rank
statistics are whole-stream statistics and that the collected metric sums the
DP ranks' streams, with and without the SPMD typechecker.
"""

import contextlib
import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from spmd_types.checker import typecheck
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.aux_loss import (
    _zero_aux_losses,
    AuxLoss,
    collect_aux_loss_metrics,
)
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import (
    BatchWiseLoadBalanceLoss,
    MicrobatchWiseLoadBalanceLoss,
    MoE,
)
from torchtitan.models.gpt_oss import build_model_config as gpt_oss_build_model_config
from torchtitan.models.qwen3 import build_model_config as qwen3_build_model_config
from torchtitan.protocols.module import Module, ModuleDict

_COEFF = 0.1
_METRIC_KEY = ("dp", "microbatch_wise_load_balance_loss")
_BATCH_METRIC_KEY = ("dp", "batch_wise_load_balance_loss")


def _clear_aux_loss_registry():
    """Reset the class-level metric registry for the current process."""
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()


def _reference_loss(
    scores_TE: torch.Tensor,
    routing_map_TE: torch.Tensor,
    top_k: int,
    *,
    coeff: float = 1.0,
    padding_mask_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit DeepSeek-V3 Eqs 17-20 reference, in the loss's token-mode form.

    Eq. 18 ``f_i = (E / (K T)) * counts_i`` and Eq. 19
    ``p_i = (1 / T) * sum_t s'_t,i`` over the whole input, with ``T`` the
    input's token count; the sum-type (token-mode) form multiplies ``T`` back
    in, and ``coeff`` stands for the framework's ``coeff / denominator``.
    """
    E = scores_TE.size(-1)
    T = scores_TE.size(0) if padding_mask_T is None else (~padding_mask_T).sum()
    counts_E = routing_map_TE.sum(dim=0).to(scores_TE.dtype)
    probs_TE = scores_TE if padding_mask_T is None else scores_TE[~padding_mask_T]
    probs_TE = probs_TE / probs_TE.sum(dim=-1, keepdim=True)
    f_E = counts_E * (E / (top_k * T))
    p_E = probs_TE.sum(dim=0) / T
    return (f_E * p_E).sum() * T * coeff


def _routing_map(ids_TK: torch.Tensor, num_experts: int) -> torch.Tensor:
    """One-hot routing map for ``(T, K)`` expert ids."""
    return torch.zeros(ids_TK.shape[0], num_experts, dtype=torch.bool).scatter_(
        -1, ids_TK, True
    )


def _make_inputs(T: int, E: int, K: int):
    """Router-like ``(scores_TE, carrier_TK, routing_map_TE)`` in float64."""
    scores_TE = torch.rand(T, E, dtype=torch.float64, requires_grad=True)
    ids_TK = torch.topk(scores_TE.detach(), k=K, dim=-1, sorted=False).indices
    return scores_TE, scores_TE.gather(dim=-1, index=ids_TK), _routing_map(ids_TK, E)


def _make_loss(
    coeff: float, per_step_denominator: int
) -> MicrobatchWiseLoadBalanceLoss:
    """Loss with the given coeff and an explicit step denominator.

    The denominator is float64 here so the assertions stay exact; in training
    it is the step's int64 ``global_loss_token_counts``.
    """
    loss = MicrobatchWiseLoadBalanceLoss(
        MicrobatchWiseLoadBalanceLoss.Config(coeff=coeff)
    )
    loss.test_denominator = torch.tensor(
        float(per_step_denominator), dtype=torch.float64
    )
    loss.train()
    return loss


class _AuxLossTestCase(unittest.TestCase):
    """Use a clean auxiliary-loss metric registry for every test."""

    def setUp(self):
        _clear_aux_loss_registry()

    def tearDown(self):
        _clear_aux_loss_registry()


class TestMicrobatchWiseLoadBalanceLoss(_AuxLossTestCase):
    """Numerics vs the reference, metric accumulation, and torch_remat safety."""

    def setUp(self):
        super().setUp()
        self.T, self.E, self.K = 15, 7, 2
        self.coeff, self.denominator = 0.125, 8
        torch.manual_seed(0)

    def test_value_and_gradient_match_reference(self):
        """The forward is an identity on the carrier, the register accumulates
        the reference value, and the injected gradient equals the reference's:
        the gradient reaches the router through the normalized scores and the
        carrier, never through the one-hot counts."""
        scores_TE, carrier_TK, routing_map_TE = _make_inputs(self.T, self.E, self.K)
        loss = _make_loss(self.coeff, self.denominator)

        out_TK = loss(
            scores_TE,
            routing_map_TE,
            carrier=carrier_TK,
            denominator=loss.test_denominator,
        )
        self.assertTrue(torch.equal(out_TK, carrier_TK))

        # The register holds the raw value over the denominator (no coeff),
        # in float32, hence the tolerance.
        _zero_aux_losses([loss])
        ref = _reference_loss(scores_TE, routing_map_TE, self.K).item()
        self.assertAlmostEqual(
            AuxLoss.group_acc[_METRIC_KEY].item(),
            ref / self.denominator,
            places=4,
        )

        ref_scores = scores_TE.detach().clone().requires_grad_(True)
        ref_aux = _reference_loss(
            ref_scores, routing_map_TE, self.K, coeff=self.coeff / self.denominator
        )
        # The carrier adds its own gradient path, as it does in the forward.
        (ref_aux + (ref_scores * routing_map_TE).sum()).backward()
        out_TK.sum().backward()
        self.assertLess((scores_TE.grad - ref_scores.grad).abs().max().item(), 1e-10)

    def test_accumulates_forwards_then_clears(self):
        """Every forward adds its scaled value to the register, and the roll-up
        into group_acc zeroes the instance accumulator."""
        loss = _make_loss(self.coeff, self.denominator)
        ref_total = 0.0
        for _ in range(3):
            scores_TE, carrier_TK, routing_map_TE = _make_inputs(self.T, self.E, self.K)
            out_TK = loss(
                scores_TE,
                routing_map_TE,
                carrier=carrier_TK,
                denominator=loss.test_denominator,
            )
            out_TK.sum().backward()
            ref_total += _reference_loss(scores_TE, routing_map_TE, self.K).item()

        _zero_aux_losses([loss])
        self.assertAlmostEqual(
            AuxLoss.group_acc[_METRIC_KEY].item(),
            ref_total / self.denominator,
            places=3,
        )
        self.assertEqual(loss.instance_acc.item(), 0.0)

    def test_masked_routing_rows_do_not_affect_aux_loss(self):
        scores_TE, carrier_TK, routing_map_TE = _make_inputs(self.T, self.E, self.K)
        full_routing_map_TE = routing_map_TE.clone()
        routing_map_TE[self.T // 2 :] = False
        loss = _make_loss(self.coeff, self.denominator)

        padding_mask_T = ~routing_map_TE.any(dim=-1)
        out_TK = loss(
            scores_TE,
            routing_map_TE,
            carrier=carrier_TK,
            padding_mask_T=padding_mask_T,
            denominator=loss.test_denominator,
        )
        out_TK.sum().backward()
        _zero_aux_losses([loss])

        ref_scores_TE = scores_TE.detach().clone().requires_grad_(True)
        ref_aux = _reference_loss(
            ref_scores_TE,
            routing_map_TE,
            self.K,
            coeff=self.coeff / self.denominator,
            padding_mask_T=padding_mask_T,
        )
        (ref_aux + (ref_scores_TE * full_routing_map_TE).sum()).backward()

        ref_metric = _reference_loss(
            scores_TE.detach(),
            routing_map_TE,
            self.K,
            padding_mask_T=padding_mask_T,
        )
        self.assertAlmostEqual(
            AuxLoss.group_acc[_METRIC_KEY].item(),
            ref_metric.item() / self.denominator,
            places=4,
        )
        self.assertLess(
            (scores_TE.grad - ref_scores_TE.grad).abs().max().item(),
            1e-10,
        )

    def test_no_double_count_with_remat_checkpointing(self):
        """inject() keeps the accumulation in its own retained remat region, so
        replaying an enclosing checkpoint counts each forward exactly once."""
        loss = _make_loss(self.coeff, self.denominator)
        scores_TE, carrier_TK, routing_map_TE = _make_inputs(self.T, self.E, self.K)

        def _forward_once(module, carrier, scores_TE, routing_map_TE):
            return module(
                scores_TE,
                routing_map_TE,
                carrier=carrier,
                denominator=module.test_denominator,
            ).sum()

        out = remat.checkpoint()(_forward_once)(
            loss, carrier_TK, scores_TE, routing_map_TE
        )
        out.backward()
        _zero_aux_losses([loss])
        ref = _reference_loss(scores_TE, routing_map_TE, self.K).item()
        self.assertAlmostEqual(
            AuxLoss.group_acc[_METRIC_KEY].item(),
            ref / self.denominator,
            places=5,
        )


class TestBatchWiseLoadBalanceLoss(_AuxLossTestCase):
    def test_rolling_value_and_gradient_match_reference(self):
        T, E, K = 8, 7, 2
        coeff = 0.125
        torch.manual_seed(0)
        inputs = [_make_inputs(T, E, K) for _ in range(2)]
        full_routing_maps = [inputs[0][2].clone(), inputs[1][2].clone()]
        inputs[1][2][5:] = False
        second_padding_mask_T = ~inputs[1][2].any(dim=-1)
        padding_masks: list[torch.Tensor | None] = [None, second_padding_mask_T]
        denominator = T + int((~second_padding_mask_T).sum())
        denominator_t = torch.tensor(float(denominator), dtype=torch.float64)
        loss = BatchWiseLoadBalanceLoss(
            BatchWiseLoadBalanceLoss.Config(coeff=coeff, num_experts=E)
        )

        outputs = [
            loss(
                scores,
                routing_map,
                carrier=carrier,
                padding_mask_T=padding_mask,
                denominator=denominator_t,
            )
            for (scores, carrier, routing_map), padding_mask in zip(
                inputs, padding_masks, strict=True
            )
        ]
        for output, (_, carrier, _) in zip(outputs, inputs, strict=True):
            self.assertTrue(torch.equal(output, carrier))
        torch.stack([output.sum() for output in outputs]).sum().backward()
        _zero_aux_losses([loss])

        ref_scores = [
            scores.detach().clone().requires_grad_(True) for scores, _, _ in inputs
        ]
        cumulative_counts_E = torch.zeros(E, dtype=torch.float64)
        ref_raw = torch.zeros((), dtype=torch.float64)
        ref_main = torch.zeros((), dtype=torch.float64)
        for (
            ref_scores_TE,
            (_, _, routing_map_TE),
            full_routing_map_TE,
            padding_mask_T,
        ) in zip(ref_scores, inputs, full_routing_maps, padding_masks, strict=True):
            cumulative_counts_E += routing_map_TE.sum(dim=0)
            f_E = F.normalize(cumulative_counts_E, p=1, dim=0) * E
            probs_TE = F.normalize(ref_scores_TE, p=1, dim=-1)
            if padding_mask_T is not None:
                probs_TE = probs_TE * ~padding_mask_T.unsqueeze(-1)
            prob_sums_E = probs_TE.sum(dim=0)
            ref_raw = ref_raw + (f_E * prob_sums_E).sum()
            ref_main = ref_main + (ref_scores_TE * full_routing_map_TE).sum()
        (ref_main + ref_raw * (coeff / denominator)).backward()

        for (scores_TE, _, _), ref_scores_TE in zip(inputs, ref_scores, strict=True):
            assert scores_TE.grad is not None
            assert ref_scores_TE.grad is not None
            self.assertLess(
                (scores_TE.grad - ref_scores_TE.grad).abs().max().item(), 1e-10
            )
        self.assertAlmostEqual(
            AuxLoss.group_acc[_BATCH_METRIC_KEY].item(),
            ref_raw.item() / denominator,
            places=4,
        )
        self.assertEqual(loss._cumulative_expert_counts_E.count_nonzero().item(), 0)

    def test_activation_checkpoint_replay_does_not_advance_rolling_counts(self):
        """Checkpoint replays must match no AC.

        Under gradient accumulation (each backward right after its forward)
        every AC policy matches. Under a pipeline-style order (all forwards,
        then all backwards) only the torch_remat-based policies (RegionAC,
        SelectiveAC) do, because they retain the original rolling-count snapshot
        instead of re-reading the advanced buffer.
        """
        T, D, E, K = 8, 6, 5, 2
        num_microbatches = 2
        denominator = torch.tensor(float(T * num_microbatches), dtype=torch.float64)

        class _Block(Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Linear(D, E, dtype=torch.float64)
                self.aux_loss = BatchWiseLoadBalanceLoss(
                    BatchWiseLoadBalanceLoss.Config(coeff=_COEFF, num_experts=E)
                )

            def forward(self, x_TD):
                scores_TE = self.gate(x_TD).softmax(dim=-1)
                ids_TK = torch.topk(scores_TE.detach(), k=K, dim=-1).indices
                carrier_TK = scores_TE.gather(dim=-1, index=ids_TK)
                return self.aux_loss(
                    scores_TE,
                    _routing_map(ids_TK, E),
                    carrier=carrier_TK,
                    denominator=denominator,
                )

        class _Model(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({"0": _Block()})

            def forward(self, x_TD):
                return self.layers["0"](x_TD)

        torch.manual_seed(0)
        microbatches = [
            torch.randn(T, D, dtype=torch.float64) for _ in range(num_microbatches)
        ]
        initial_state = _Model().state_dict()

        def run(ac_config, *, forwards_first: bool = False):
            model = _Model()
            model.load_state_dict(initial_state)
            model.train()
            if ac_config is not None:
                ac_config.build().apply(model)
            if forwards_first:
                outputs = [model(x_TD).sum() for x_TD in microbatches]
                for output in outputs:
                    output.backward()
            else:
                for x_TD in microbatches:
                    model(x_TD).sum().backward()
            block = next(m for m in model.modules() if isinstance(m, _Block))
            return (
                block.aux_loss._cumulative_expert_counts_E.clone(),
                block.gate.weight.grad.clone(),
            )

        ref_counts_E, ref_grad = run(None)
        self.assertEqual(ref_counts_E.sum().item(), T * K * len(microbatches))
        for ac_config in (
            FullAC.Config(),
            SelectiveAC.Config(),
            RegionAC.Config(save_regions=[]),
        ):
            with self.subTest(ac=type(ac_config).__qualname__):
                counts_E, grad = run(ac_config)
                torch.testing.assert_close(counts_E, ref_counts_E, rtol=0, atol=0)
                torch.testing.assert_close(grad, ref_grad)

        ref_counts_E, ref_grad = run(None, forwards_first=True)
        for ac_config in (SelectiveAC.Config(), RegionAC.Config(save_regions=[])):
            with self.subTest(ac=type(ac_config).__qualname__, forwards_first=True):
                counts_E, grad = run(ac_config, forwards_first=True)
                torch.testing.assert_close(counts_E, ref_counts_E, rtol=0, atol=0)
                torch.testing.assert_close(grad, ref_grad)
        # FullAC re-reads the advanced buffer during the first replay, which is
        # why validation rejects it with multiple pipeline microbatches.
        _, full_ac_grad = run(FullAC.Config(), forwards_first=True)
        self.assertFalse(torch.allclose(full_ac_grad, ref_grad))

    def test_pipeline_microbatches_with_full_ac_is_rejected(self):
        seq_len = 16
        model = qwen3_build_model_config("debugmodel_moe", seq_len=seq_len)

        def validate(*, pp: int, num_microbatches: int, ac_config) -> None:
            validate_model_training_config(
                model,
                parallelism=ParallelismConfig(
                    pipeline_parallel_degree=pp,
                    num_pp_microbatches=num_microbatches,
                ),
                training=TrainingConfig(
                    max_context_length=seq_len, disable_cuda_graphs=True
                ),
                debug=DebugConfig(),
                activation_checkpoint=ac_config,
                max_num_documents=None,
            )

        validate(pp=2, num_microbatches=1, ac_config=SelectiveAC.Config())
        validate(pp=2, num_microbatches=2, ac_config=None)
        validate(pp=2, num_microbatches=2, ac_config=RegionAC.Config(save_regions=[]))
        validate(pp=2, num_microbatches=2, ac_config=SelectiveAC.Config())
        # Without PP, microbatches run as gradient accumulation.
        validate(pp=1, num_microbatches=2, ac_config=FullAC.Config())
        with self.assertRaisesRegex(ValueError, "multiple pipeline microbatches"):
            validate(pp=2, num_microbatches=2, ac_config=FullAC.Config())


class TestLoadBalanceLossConfig(_AuxLossTestCase):
    def test_moe_models_default_to_batch_wise_loss(self):
        for model_config in (
            gpt_oss_build_model_config("debugmodel", seq_len=16),
            qwen3_build_model_config("debugmodel_moe", seq_len=16),
        ):
            losses = list(model_config.traverse(BatchWiseLoadBalanceLoss.Config))
            self.assertTrue(losses)
            self.assertTrue(all(loss.coeff == 1e-3 for _, loss, _, _ in losses))
            self.assertTrue(
                all(
                    moe.load_balance_coeff is None
                    for _, moe, _, _ in model_config.traverse(MoE.Config)
                )
            )

    def test_config_builds_selected_loss(self):
        self.assertIs(
            type(MicrobatchWiseLoadBalanceLoss.Config(coeff=_COEFF).build()),
            MicrobatchWiseLoadBalanceLoss,
        )
        self.assertIs(type(AuxLoss.Config(coeff=_COEFF).build()), AuxLoss)

        moe_cfg = make_moe_config(
            num_experts=4,
            router=make_router_config(
                dim=8,
                num_experts=4,
                gate_param_init={},
                score_func=Sigmoid.Config(),
            ),
            routed_experts=make_routed_experts_config(
                dim=8,
                hidden_dim=16,
                num_experts=4,
                top_k=1,
                param_init={},
            ),
            aux_loss_coeff=_COEFF,
        )
        self.assertIs(
            type(moe_cfg.router.aux_loss.build()), MicrobatchWiseLoadBalanceLoss
        )
        moe_cfg = make_moe_config(
            num_experts=4,
            router=moe_cfg.router,
            routed_experts=moe_cfg.routed_experts,
            aux_loss_coeff=_COEFF,
            aux_loss_type="batch_wise",
        )
        self.assertIs(type(moe_cfg.router.aux_loss.build()), BatchWiseLoadBalanceLoss)
        moe_cfg = make_moe_config(
            num_experts=4,
            router=moe_cfg.router,
            routed_experts=moe_cfg.routed_experts,
            aux_loss_coeff=None,
        )
        self.assertIsNone(moe_cfg.router.aux_loss)


class TestLoadBalanceLossSpmdTypes(DTensorTestBase):
    """8-rank load-balance cases: dp2/cp2/tp2 on CPU float64 + gloo.

    Microbatch-wise statistics span one DP-local token stream; batch-wise
    statistics additionally span DP. Both must hold with and without the
    typechecker.
    """

    @property
    def world_size(self):
        return 8

    @property
    def device_type(self):
        return "cpu"

    def _build_dims(self, **overrides):
        """ParallelismContext on CPU; ``overrides`` replace the default dp2/cp2/tp2."""
        from torchtitan.distributed.parallelism_context import ParallelismContext

        kwargs = dict(
            dp_replicate=1,
            dp_shard=2,
            cp=2,
            tp=2,
            pp=1,
            ep=1,
            world_size=8,
            enable_sequence_parallel=False,
        )
        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            parallelism_context = ParallelismContext(**{**kwargs, **overrides})
            parallelism_context.build_mesh()
        return parallelism_context

    def _setup_mesh(self):
        """Register the meshes and return ``(parallelism_context, dense_mesh)``.

        The router output shards tokens over CP and TP. DP stays local: one
        stream per DP rank.
        """
        from torchtitan.distributed.spmd_types import set_spmd_meshes

        parallelism_context = self._build_dims(ep=2)
        dense_mesh = parallelism_context.get_mesh(["dp", "cp", "tp"])
        set_spmd_meshes(
            dense_mesh=dense_mesh,
            sparse_mesh=parallelism_context.spmd_sparse_mesh(),
            dense_sp_enabled=parallelism_context.sp_enabled,
        )
        return parallelism_context, dense_mesh

    @with_comms
    def test_pp_reduction_sums_stages(self):
        """collect_aux_loss_metrics sums the pipeline stages -- every layer
        lives on exactly one stage -- and divides by the build-time instance
        count, i.e. it reports the mean over layers.  Averaging the stages
        instead would under-report by the pipeline degree."""
        parallelism_context = self._build_dims(cp=2, tp=1, pp=2)
        _clear_aux_loss_registry()
        # This rank: 6 instances built, 3.0 accumulated, 2 DP coords in the
        # batch mesh; summing the 2 stages gives 12.0, divided by 6 gives 2.0.
        AuxLoss._group_counts[_METRIC_KEY] = 6
        AuxLoss.group_acc[_METRIC_KEY] = torch.tensor(3.0, dtype=torch.float32)

        metrics = collect_aux_loss_metrics(parallelism_context)
        self.assertAlmostEqual(metrics[f"{_METRIC_KEY[1]}/mean"], 2.0, places=6)
        _clear_aux_loss_registry()

    def _run_reduction_case(self, *, use_typecheck: bool):
        """Compare one distributed layout with the per-DP-rank reference."""
        parallelism_context, dense_mesh = self._setup_mesh()
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh

        T, E, K, dp, cp, tp = 128, 8, 2, 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        tp_rank = self.rank % tp
        t_dp = T // dp
        dp_start = dp_rank * t_dp

        from torchtitan.models.common.decoder_sharding import (
            dense_sequence_parallel_placement,
        )

        shard, t_blk = cp_rank * tp + tp_rank, t_dp // (cp * tp)
        placement = dense_sequence_parallel_placement()
        t_start = dp_start + shard * t_blk

        checker = typecheck(local=False) if use_typecheck else contextlib.nullcontext()
        _clear_aux_loss_registry()
        denominator = torch.tensor(1.0, dtype=torch.float64)
        with set_current_spmd_mesh(dense_mesh), checker:
            torch.manual_seed(0)
            global_scores_TE = torch.rand(T, E, dtype=torch.float64)
            with spmd.no_typecheck():
                # Distinct ids per token, like torch.topk in the router, so each
                # token contributes exactly K one-hot entries.
                global_ids_TK = torch.topk(torch.rand(T, E), k=K, dim=-1).indices
                local_scores = global_scores_TE[t_start : t_start + t_blk].contiguous()
                local_ids_TK = global_ids_TK[t_start : t_start + t_blk].contiguous()
                local_map = _routing_map(local_ids_TK, E)

            spmd.assert_type(local_scores, placement)
            spmd.assert_type(local_ids_TK, placement)
            spmd.assert_type(local_map, placement)

            loss = MicrobatchWiseLoadBalanceLoss(
                MicrobatchWiseLoadBalanceLoss.Config(coeff=_COEFF)
            )
            local_scores.requires_grad_(True)
            carrier_TK = local_scores.gather(dim=-1, index=local_ids_TK)
            out_TK = loss(
                local_scores,
                local_map,
                carrier=carrier_TK,
                denominator=denominator,
            )

            with spmd.no_typecheck():
                torch.testing.assert_close(out_TK, carrier_TK, rtol=0, atol=0)
                # Backward runs outside the checker in both modes.
                out_TK.sum().backward()

                dp_scores = global_scores_TE[dp_start : dp_start + t_dp]
                dp_ids_TK = global_ids_TK[dp_start : dp_start + t_dp]
                dp_map = _routing_map(dp_ids_TK, E)
                # The metric accumulator is float32 (the reference float64).
                ref_raw = _reference_loss(dp_scores, dp_map, K).item()
                self.assertAlmostEqual(loss.instance_acc.item(), ref_raw, places=4)

                ref_scores = dp_scores.detach().clone().requires_grad_(True)
                ref_aux = _reference_loss(ref_scores, dp_map, K, coeff=_COEFF)
                (ref_aux + (ref_scores * dp_map).sum()).backward()
                ref_local_grad = ref_scores.grad[shard * t_blk : (shard + 1) * t_blk]
                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )

        # The metric sums the reduce mesh, so it equals the sum over the DP
        # ranks' streams -- the step-global value a single-rank run over both
        # streams would produce (denominator 1 here).
        _zero_aux_losses([loss])
        ref_total = 0.0
        for stream in range(dp):
            stream_scores = global_scores_TE[stream * t_dp : (stream + 1) * t_dp]
            stream_ids_TK = global_ids_TK[stream * t_dp : (stream + 1) * t_dp]
            ref_total += _reference_loss(
                stream_scores, _routing_map(stream_ids_TK, E), K
            ).item()
        metrics = collect_aux_loss_metrics(parallelism_context)
        self.assertAlmostEqual(metrics[f"{_METRIC_KEY[1]}/mean"], ref_total, places=4)
        _clear_aux_loss_registry()

    @with_comms
    def test_reduction_matches_reference(self):
        """Loss, gradient and collected metric match the per-DP-rank reference:
        P->I over CP and TP, with and without the typechecker."""
        for use_typecheck in (True, False):
            with self.subTest(use_typecheck=use_typecheck):
                self._run_reduction_case(use_typecheck=use_typecheck)

    @with_comms
    def test_batch_wise_reduction_matches_reference(self):
        parallelism_context, dense_mesh = self._setup_mesh()
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh
        from torchtitan.models.common.decoder_sharding import (
            dense_sequence_parallel_placement,
            token_id_placement,
        )

        T, E, K, dp, cp, tp = 128, 8, 2, 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        tp_rank = self.rank % tp
        t_dp = T // dp
        dp_start = dp_rank * t_dp
        shard, t_blk = cp_rank * tp + tp_rank, t_dp // (cp * tp)
        t_start = dp_start + shard * t_blk

        with set_current_spmd_mesh(dense_mesh), typecheck(local=False):
            torch.manual_seed(0)
            global_scores_TE = torch.rand(T, E, dtype=torch.float64)
            with spmd.no_typecheck():
                global_ids_TK = torch.topk(torch.rand(T, E), k=K, dim=-1).indices
                global_padding_mask_T = torch.zeros(T, dtype=torch.bool)
                global_padding_mask_T[:3] = True
                global_padding_mask_T[t_dp : t_dp + 7] = True
                denominator = int((~global_padding_mask_T).sum())
                local_scores = global_scores_TE[t_start : t_start + t_blk].contiguous()
                local_ids_TK = global_ids_TK[t_start : t_start + t_blk].contiguous()
                local_padding_mask_T = global_padding_mask_T[
                    t_start : t_start + t_blk
                ].contiguous()
                full_local_map = _routing_map(local_ids_TK, E)
                local_map = full_local_map & ~local_padding_mask_T.unsqueeze(-1)

            placement = dense_sequence_parallel_placement()
            spmd.assert_type(local_scores, placement)
            spmd.assert_type(local_ids_TK, placement)
            spmd.assert_type(local_map, placement)
            spmd.assert_type(local_padding_mask_T, token_id_placement(enable_sp=True))

            loss = BatchWiseLoadBalanceLoss(
                BatchWiseLoadBalanceLoss.Config(coeff=_COEFF, num_experts=E)
            )
            local_scores.requires_grad_(True)
            carrier_TK = local_scores.gather(dim=-1, index=local_ids_TK)
            out_TK = loss(
                local_scores,
                local_map,
                carrier=carrier_TK,
                padding_mask_T=local_padding_mask_T,
                denominator=torch.tensor(float(denominator), dtype=torch.float64),
            )

            with spmd.no_typecheck():
                torch.testing.assert_close(out_TK, carrier_TK, rtol=0, atol=0)
                out_TK.sum().backward()

                full_global_map = _routing_map(global_ids_TK, E)
                global_map = full_global_map & ~global_padding_mask_T.unsqueeze(-1)
                ref_scores = global_scores_TE.detach().clone().requires_grad_(True)
                ref_aux = _reference_loss(
                    ref_scores,
                    global_map,
                    K,
                    coeff=_COEFF / denominator,
                    padding_mask_T=global_padding_mask_T,
                )
                (ref_aux + (ref_scores * full_global_map).sum()).backward()
                ref_local_grad = ref_scores.grad[t_start : t_start + t_blk]
                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )

        _zero_aux_losses([loss])
        metrics = collect_aux_loss_metrics(parallelism_context)
        ref_metric = (
            _reference_loss(
                global_scores_TE,
                global_map,
                K,
                padding_mask_T=global_padding_mask_T,
            ).item()
            / denominator
        )
        self.assertAlmostEqual(
            metrics[f"{_BATCH_METRIC_KEY[1]}/mean"], ref_metric, places=4
        )
        _clear_aux_loss_registry()


if __name__ == "__main__":
    unittest.main()
