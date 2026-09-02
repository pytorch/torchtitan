# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for LoggedAuxLoss and SeqwiseLoadBalanceLoss.

TestCase groups:
- TestSeqwiseLoadBalanceLossNumerics: forward/backward values vs an explicit
  formula reference, injected gradient vs explicit loss, and non-differentiability
  of the one-hot counts.
- TestLoggedAuxLossAccumulation: accumulation scaling, zero/snapshot semantics,
  and that activation checkpointing (none/full/full_compile/selective) never
  double-counts the metric.
- TestSeqwiseCountsSpmdTypes: distributed (8-rank) end-to-end test under
  spmd_types typechecking with EP enabled (P->I on CP+TP) and EP disabled
  (P->I on CP only). Loss and gradients match the reference within float64
  roundoff.
"""

import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.models.common.aux_loss import _zero_aux_losses, LoggedAuxLoss
from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss


def _reference_seqwise_aux_loss(
    scores_TE: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    top_k: int,
    coeff: float,
    per_step_denominator: int,
) -> torch.Tensor:
    """Explicit DeepSeek-V3 seq-wise aux loss (Sec A.2 Eqs 17-20), framework-scaled."""
    E = scores_TE.size(-1)
    routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
        -1, topk_expert_ids_TK, True
    )
    counts_E = routing_map_TE.sum(dim=0).to(scores_TE.dtype)
    probs_TE = scores_TE / scores_TE.sum(dim=-1, keepdim=True)
    prob_sums_E = probs_TE.sum(dim=0)
    num_tokens = counts_E.sum() / top_k
    f_E = counts_E * (E / (top_k * num_tokens))
    p_E = prob_sums_E / num_tokens
    return (f_E * p_E).sum() * (coeff / per_step_denominator)


def _make_loss_module(top_k: int, coeff: float, per_step_denominator: int):
    """Build a SeqwiseLoadBalanceLoss with the given parameters and no parallelization."""
    cfg = SeqwiseLoadBalanceLoss.Config(coeff=coeff, top_k=top_k)
    cfg.per_step_denominator = per_step_denominator
    loss = SeqwiseLoadBalanceLoss(cfg)
    loss.train()
    return loss


def _seqwise_loss_config(
    top_k: int, per_step_denominator: int, *, enable_ep: bool = True
):
    """Build a SeqwiseLoadBalanceLoss config with the framework-set fields and sharding config."""
    from torchtitan.models.common.moe_sharding import _seqwise_counts_sharding_config

    cfg = SeqwiseLoadBalanceLoss.Config(coeff=0.1, top_k=top_k)
    cfg.counts.sharding_config = _seqwise_counts_sharding_config(enable_ep=enable_ep)
    cfg.per_step_denominator = per_step_denominator
    return cfg


def _set_spmd_types_backend():
    from torchtitan.distributed.utils import set_spmd_backend

    set_spmd_backend("spmd_types")


def _restore_default_backend():
    from torchtitan.distributed.utils import set_spmd_backend

    set_spmd_backend("partial_dtensor")


def _clear_aux_loss_registry():
    """Reset the class-level metric registry for the current process."""
    LoggedAuxLoss._group_counts.clear()
    LoggedAuxLoss._step_snapshots.clear()


class TestSeqwiseLoadBalanceLossNumerics(unittest.TestCase):
    """Forward/backward of SeqwiseLoadBalanceLoss vs an explicit reference."""

    def setUp(self):
        self.T, self.E, self.K = 15, 7, 2
        self.coeff = 0.125
        self.per_step_denominator = 1
        torch.manual_seed(0)
        _set_spmd_types_backend()

    def tearDown(self):
        _restore_default_backend()
        _clear_aux_loss_registry()

    def _make_inputs(self):
        scores_TE = torch.rand(self.T, self.E, dtype=torch.float32, requires_grad=True)
        topk_expert_ids_TK = torch.topk(
            scores_TE.detach(), k=self.K, dim=-1, sorted=False
        ).indices
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        return scores_TE, topk_scores_TK, topk_expert_ids_TK

    def test_loss_value_matches_formula(self):
        """The accumulated metric after backward equals the reference formula."""
        scores_TE, _, topk_expert_ids_TK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        out = loss(scores_TE, scores_TE, topk_expert_ids_TK)
        self.assertEqual(out.shape, scores_TE.shape)
        torch.testing.assert_close(out, scores_TE, rtol=0, atol=0)

        out.sum().backward()
        expected = _reference_seqwise_aux_loss(
            scores_TE,
            topk_expert_ids_TK,
            self.K,
            1.0,
            self.per_step_denominator,
        )
        torch.testing.assert_close(loss._acc_sum, expected, rtol=1e-12, atol=1e-12)

    def test_gradient_matches_explicit_loss(self):
        """Injected gradient matches adding the aux loss explicitly to the main loss."""
        scores_TE, _, topk_expert_ids_TK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        injected_scores = scores_TE.clone().detach().requires_grad_(True)
        injected_topk = injected_scores.gather(dim=-1, index=topk_expert_ids_TK)
        injected = loss(injected_topk, injected_scores, topk_expert_ids_TK)
        injected.sum().backward()

        explicit_scores = scores_TE.clone().detach().requires_grad_(True)
        explicit_topk = explicit_scores.gather(dim=-1, index=topk_expert_ids_TK)
        explicit_aux = _reference_seqwise_aux_loss(
            explicit_scores,
            topk_expert_ids_TK,
            self.K,
            self.coeff,
            self.per_step_denominator,
        )
        (explicit_topk.sum() + explicit_aux).backward()

        torch.testing.assert_close(
            injected_scores.grad, explicit_scores.grad, rtol=1e-12, atol=1e-12
        )

    def test_counts_do_not_receive_gradient(self):
        """The one-hot counts (Eq 18) are non-differentiable, so gradient flows through router."""
        scores_TE, _, topk_expert_ids_TK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)
        out = loss(scores_TE, scores_TE, topk_expert_ids_TK)
        out.sum().backward()
        self.assertIsNotNone(scores_TE.grad)
        self.assertTrue(torch.isfinite(scores_TE.grad).all())
        self.assertGreater(scores_TE.grad.abs().sum().item(), 0)


class TestLoggedAuxLossAccumulation(unittest.TestCase):
    """Accumulation semantics of LoggedAuxLoss."""

    def setUp(self):
        torch.manual_seed(0)
        self.coeff = 0.1
        self.per_step_denominator = 100
        _set_spmd_types_backend()

    def tearDown(self):
        _restore_default_backend()
        _clear_aux_loss_registry()

    def _make_loss(self):
        return _make_loss_module(2, self.coeff, self.per_step_denominator)

    def _forward_once(self, loss, carrier, scores, ids, *, use_checkpoint, use_compile):
        def run(c, s, i):
            return loss(c, s, i).sum()

        if not use_checkpoint:
            fn = run
            if use_compile:
                fn = torch.compile(fn, backend="aot_eager")
            return fn(carrier, scores, ids)

        def fwd(c, s, i):
            return torch.utils.checkpoint.checkpoint(run, c, s, i, use_reentrant=False)

        fn = fwd
        if use_compile:
            fn = torch.compile(fn, backend="aot_eager")
        return fn(carrier, scores, ids)

    def _run_plain_once(self, loss):
        """Single forward-backward pass, returning the accumulated metric."""
        torch.manual_seed(0)
        scores_TE = torch.rand(24, 8, requires_grad=True)
        ids = torch.randint(0, 8, (24, 2))
        carrier = scores_TE.gather(dim=-1, index=ids)
        out = loss(carrier, scores_TE, ids).sum()
        out.backward()
        return loss._acc_sum.item()

    def _run_full_ac_once(self, loss, *, use_compile):
        """Forward-backward with full activation checkpointing."""
        torch.manual_seed(0)
        scores_TE = torch.rand(24, 8, requires_grad=True)
        ids = torch.randint(0, 8, (24, 2))
        carrier = scores_TE.gather(dim=-1, index=ids)
        out = self._forward_once(
            loss,
            carrier,
            scores_TE,
            ids,
            use_checkpoint=True,
            use_compile=use_compile,
        )
        out.backward()
        return loss._acc_sum.item()

    def _run_selective_ac_once(self, loss):
        """Forward-backward where only the carrier-producing gather is checkpointed.

        This is a lightweight stand-in for TorchTitan's per-op SelectiveAC; it
        deliberately checkpoints one op to verify that partial recomputation
        does not double-count the aux metric.
        """
        torch.manual_seed(0)
        scores_TE = torch.rand(24, 8, requires_grad=True)
        ids = torch.randint(0, 8, (24, 2))

        def pre(c, i):
            return c.gather(dim=-1, index=i)

        def fwd(c, i):
            carrier = torch.utils.checkpoint.checkpoint(pre, c, i, use_reentrant=False)
            return loss(carrier, c, i).sum()

        fwd(scores_TE, ids).backward()
        return loss._acc_sum.item()

    def test_accumulation_is_scaled(self):
        """The accumulated metric is scaled by 1/per_step_denominator."""
        loss = self._make_loss()
        carrier = torch.rand(4, 6, requires_grad=True)
        out1 = loss.inject(torch.tensor(50.0), carrier=carrier)
        out2 = loss.inject(torch.tensor(50.0), carrier=carrier)
        (out1.sum() + out2.sum()).backward()
        self.assertAlmostEqual(loss._acc_sum.item(), 1.0, places=6)

    def test_zero_all_snapshots_and_clears(self):
        """_zero_aux_losses snapshots the accumulators and clears them."""
        loss1 = self._make_loss()
        loss2 = self._make_loss()
        for loss in (loss1, loss2):
            carrier = torch.rand(4, 6, requires_grad=True)
            out = loss.inject(torch.tensor(10.0), carrier=carrier)
            out.sum().backward()
        _zero_aux_losses([loss1, loss2])
        self.assertEqual(loss1._acc_sum.item(), 0.0)
        self.assertEqual(loss2._acc_sum.item(), 0.0)
        key = ("batch", "seqwise_load_balance_loss")
        self.assertAlmostEqual(LoggedAuxLoss._step_snapshots[key].item(), 0.2)

    def test_ac_recompute_does_not_double_count(self):
        """AC recompute never adds the metric twice across none/full/full_compile/selective."""
        torch.manual_seed(0)
        plain = self._run_plain_once(self._make_loss())

        def run(loss, mode):
            if mode == "none":
                return self._run_plain_once(loss)
            if mode == "full":
                return self._run_full_ac_once(loss, use_compile=False)
            if mode == "full_compile":
                return self._run_full_ac_once(loss, use_compile=True)
            if mode == "selective":
                return self._run_selective_ac_once(loss)
            raise ValueError(f"unknown mode {mode}")

        for mode in ("none", "full", "full_compile", "selective"):
            with self.subTest(mode=mode):
                loss = self._make_loss()
                self.assertAlmostEqual(run(loss, mode), plain, places=6)


class TestSeqwiseCountsSpmdTypes(DTensorTestBase):
    """Distributed spmd_types test for sequence-wise counts reduction."""

    @property
    def world_size(self):
        return 8

    def _setup_mesh(self):
        """Build parallel dims and register meshes. Returns (parallel_dims, dense_mesh)."""
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.distributed.spmd_types import set_spmd_meshes
        from torchtitan.distributed.utils import set_spmd_backend

        set_spmd_backend("spmd_types")
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=2,
                cp=2,
                tp=2,
                pp=1,
                ep=2,
                world_size=8,
                spmd_backend="spmd_types",
            )
            parallel_dims.build_mesh()
            dense_mesh = parallel_dims.get_mesh(["dp", "cp", "tp"])
            set_spmd_meshes(
                dense_mesh=dense_mesh,
                sparse_mesh=parallel_dims.get_optional_mesh(
                    ["dp_replicate", "efsdp", "ep"], include_singleton_axes=True
                ),
            )
        return parallel_dims, dense_mesh

    def _reference_for_stream(self, scores, ids, top_k, E, coeff=0.1):
        """Compute reference loss and gradient for a single dp-rank stream."""
        with torch.no_grad():
            routing_map = torch.zeros(scores.shape[0], E).scatter_(-1, ids, 1.0)
            counts_E = routing_map.sum(dim=0)
            probs = scores / scores.sum(dim=-1, keepdim=True)
            prob_sums_E = probs.sum(dim=0)
            num_tokens = counts_E.sum() / top_k
            f_E = counts_E * (E / (top_k * num_tokens))
            p_E = prob_sums_E / num_tokens
            ref_loss = (f_E * p_E).sum()
        ref_scores = scores.detach().clone().requires_grad_(True)
        rm = torch.zeros(scores.shape[0], E).scatter_(-1, ids, 1.0)
        cts = rm.sum(dim=0)
        prs = ref_scores / ref_scores.sum(dim=-1, keepdim=True)
        nt = cts.sum() / top_k
        f = cts * (E / (top_k * nt))
        p = prs.sum(dim=0) / nt
        ref_aux = (f * p).sum() * coeff
        ref_carrier = ref_scores.gather(dim=-1, index=ids)
        (ref_aux + ref_carrier.sum()).backward()
        return ref_loss, ref_scores.grad

    @with_comms
    def test_ep_enabled_reduction(self):
        """EP=2: P->I on CP and TP tokens; loss and gradient match the reference."""
        parallel_dims, dense_mesh = self._setup_mesh()
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh
        from torchtitan.models.common.decoder_sharding import (
            dense_sequence_parallel_placement,
        )
        from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss

        T, E, K = 128, 8, 2
        dp, cp, tp = 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        tp_rank = self.rank % tp
        t_dp = T // dp
        t_blk = t_dp // (cp * tp)
        dp_start = dp_rank * t_dp
        t_start = dp_start + (cp_rank * tp + tp_rank) * t_blk
        t_end = t_start + t_blk

        with set_current_spmd_mesh(dense_mesh), typecheck(local=False):
            torch.manual_seed(0)
            global_scores = torch.rand(T, E, dtype=torch.float64)
            global_ids = torch.randint(0, E, (T, K))
            with spmd.no_typecheck():
                local_scores = global_scores[t_start:t_end].contiguous()
                local_ids = global_ids[t_start:t_end].contiguous()

            spmd.assert_type(local_scores, dense_sequence_parallel_placement())
            spmd.assert_type(local_ids, dense_sequence_parallel_placement())

            loss = SeqwiseLoadBalanceLoss(_seqwise_loss_config(K, 1))
            loss.parallelize(parallel_dims)

            local_scores.requires_grad_(True)
            carrier = local_scores.gather(dim=-1, index=local_ids)
            out = loss(carrier, local_scores, local_ids)
            with spmd.no_typecheck():
                torch.testing.assert_close(out, carrier, rtol=0, atol=0)
            out.sum().backward()
            local_scores.grad = None

            loss_grad = SeqwiseLoadBalanceLoss(_seqwise_loss_config(K, 1))
            loss_grad.parallelize(parallel_dims)
            carrier_g = local_scores.gather(dim=-1, index=local_ids)
            out2 = loss_grad(carrier_g, local_scores, local_ids)
            out2.sum().backward()

            with spmd.no_typecheck():
                dp_scores = global_scores[dp_start : dp_start + t_dp]
                dp_ids = global_ids[dp_start : dp_start + t_dp]
                ref_loss, ref_grad = self._reference_for_stream(dp_scores, dp_ids, K, E)
                self.assertAlmostEqual(loss._acc_sum.item(), ref_loss.item(), places=6)
                ref_local_grad = ref_grad[
                    (cp_rank * tp + tp_rank) * t_blk : (cp_rank * tp + tp_rank) * t_blk
                    + t_blk
                ]
                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )

    @with_comms
    def test_ep_disabled_reduction(self):
        """EP=1: P->I on CP only; loss and gradient match the reference."""
        parallel_dims, dense_mesh = self._setup_mesh()
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh
        from torchtitan.models.common.decoder_sharding import dense_activation_placement
        from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss

        T, E, K = 128, 8, 2
        dp, cp, tp = 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        t_dp = T // dp
        no_ep_blk = t_dp // cp
        dp_start = dp_rank * t_dp
        no_ep_start = dp_start + cp_rank * no_ep_blk
        no_ep_end = no_ep_start + no_ep_blk

        with set_current_spmd_mesh(dense_mesh), typecheck(local=False):
            torch.manual_seed(0)
            global_scores = torch.rand(T, E, dtype=torch.float64)
            global_ids = torch.randint(0, E, (T, K))
            with spmd.no_typecheck():
                local_scores = global_scores[no_ep_start:no_ep_end].contiguous()
                local_ids = global_ids[no_ep_start:no_ep_end].contiguous()

            no_ep_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            spmd.assert_type(local_scores, no_ep_layout)
            spmd.assert_type(local_ids, no_ep_layout)

            loss = SeqwiseLoadBalanceLoss(_seqwise_loss_config(K, 1, enable_ep=False))
            loss.parallelize(parallel_dims)

            local_scores.requires_grad_(True)
            carrier = local_scores.gather(dim=-1, index=local_ids)
            out = loss(carrier, local_scores, local_ids)
            with spmd.no_typecheck():
                torch.testing.assert_close(out, carrier, rtol=0, atol=0)
                out.sum().backward()

            with spmd.no_typecheck():
                dp_scores = global_scores[dp_start : dp_start + t_dp]
                dp_ids = global_ids[dp_start : dp_start + t_dp]
                ref_loss, ref_grad = self._reference_for_stream(dp_scores, dp_ids, K, E)
                self.assertAlmostEqual(loss._acc_sum.item(), ref_loss.item(), places=6)
                ref_local_grad = ref_grad[
                    cp_rank * no_ep_blk : (cp_rank + 1) * no_ep_blk
                ]
                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )


if __name__ == "__main__":
    unittest.main()
