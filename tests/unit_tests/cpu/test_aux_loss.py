# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for LoggedAuxLoss and SeqwiseLoadBalanceLoss (kept lean per review).

- TestSeqwiseLoadBalanceLossNumerics: forward/backward vs an explicit
  reference of DeepSeek-V3 Eqs 17-20, and injected gradient vs adding the
  reference loss to the graph.
- TestLoggedAuxLossAccumulation: accumulation semantics plus a parameterized
  AC test (none / full eager / full compiled / selective vs a plain
  reference) -- recompute must never add the metric twice; real-run AC
  agreement is in the PR description.
- TestSeqwiseCountsSpmdTypes: 8-rank dp2/cp2/tp2/ep2 under typechecking --
  values and gradients after the boundary Partial -> Invariant redistributes
  match the global-tensor reference within float64 roundoff.
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

from torchtitan.models.common.aux_loss import LoggedAuxLoss, _zero_aux_losses
from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss


def _reference_seqwise_aux_loss(
    scores_TE: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    top_k: int,
    coeff: float,
    per_step_denominator: int,
) -> torch.Tensor:
    """Explicit DeepSeek-V3 seq-wise aux loss (Sec A.2 Eqs 17-20), framework-scaled.

    Single-sequence view: the whole token stream is one sequence.
    """
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
    # ``per_step_denominator`` is not user config: framework code fills it
    # before model build.
    cfg = SeqwiseLoadBalanceLoss.Config(coeff=coeff, top_k=top_k)
    cfg.per_step_denominator = per_step_denominator
    loss = SeqwiseLoadBalanceLoss(cfg)
    loss.train()
    return loss


def _seqwise_loss_config(
    top_k: int, per_step_denominator: int, *, enable_ep: bool = True
):
    """Build a SeqwiseLoadBalanceLoss config with the framework-set fields."""
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

    set_spmd_backend("default")


def _clear_aux_loss_registry():
    """Reset the class-level metric registry for the current process."""
    LoggedAuxLoss._group_counts.clear()
    LoggedAuxLoss._step_snapshots.clear()


class TestSeqwiseLoadBalanceLossNumerics(unittest.TestCase):
    def setUp(self):
        self.T, self.E, self.K = 15, 7, 2
        self.coeff = 0.125
        # Single-sequence view: the whole (folded) token stream is one
        # sequence; the framework denominator is the per-step sequence count.
        self.per_step_denominator = 1
        torch.manual_seed(0)
        _set_spmd_types_backend()

    def tearDown(self):
        # The backend global and metric registry are process-wide; restore
        # them so other test modules are unaffected.
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
        scores_TE, _, topk_expert_ids_TK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        out = loss(scores_TE, scores_TE, topk_expert_ids_TK)

        # The forward returns the carrier unchanged (identity injection).
        self.assertEqual(out.shape, scores_TE.shape)
        torch.testing.assert_close(out, scores_TE, rtol=0, atol=0)

        # The metric accumulates in the backward (see _AuxLossInjection); the
        # accumulated value is the raw per-sequence mean, i.e. the injected
        # gradient divided by the coefficient.
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
        """Injected aux loss must match adding it to the main loss explicitly."""
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
        """f_i (Eq 18) is computed from one-hot ids, so it is non-differentiable."""
        scores_TE, _, topk_expert_ids_TK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)
        out = loss(scores_TE, scores_TE, topk_expert_ids_TK)
        out.sum().backward()
        self.assertIsNotNone(scores_TE.grad)
        self.assertTrue(torch.isfinite(scores_TE.grad).all())
        self.assertGreater(scores_TE.grad.abs().sum().item(), 0)


class TestLoggedAuxLossAccumulation(unittest.TestCase):
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

    def test_no_accumulation_in_eval_mode(self):
        loss = self._make_loss()
        loss.eval()
        loss.inject(torch.tensor(5.0), carrier=torch.rand(4, 6))
        self.assertEqual(loss._acc_sum.item(), 0.0)

    def test_accumulation_is_scaled_by_per_step_denominator(self):
        loss = self._make_loss()
        carrier = torch.rand(4, 6, requires_grad=True)
        out1 = loss.inject(torch.tensor(50.0), carrier=carrier)
        out2 = loss.inject(torch.tensor(50.0), carrier=carrier)
        # Backward through the injected outputs: the injection nodes are
        # downstream of the carrier, so carrier.backward() would skip them.
        (out1.sum() + out2.sum()).backward()
        self.assertAlmostEqual(loss._acc_sum.item(), 1.0, places=6)

    def test_zero_all_snapshots_and_clears(self):
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
        # 2 modules x (10 / 100) = 0.2
        self.assertAlmostEqual(LoggedAuxLoss._step_snapshots[key].item(), 0.2)

    def _forward_once(self, loss, carrier, scores, ids, *, use_checkpoint, use_compile):
        def run(c, s, i):
            return loss(c, s, i).sum()

        if not use_checkpoint:
            fn = run
            if use_compile:
                fn = torch.compile(fn, backend="aot_eager")
            return fn(carrier, scores, ids)

        # TorchTitan applies AC wrappers first, then compiles around them, so
        # the checkpoint lives inside the compiled region.
        def fwd(c, s, i):
            return torch.utils.checkpoint.checkpoint(run, c, s, i, use_reentrant=False)

        fn = fwd
        if use_compile:
            fn = torch.compile(fn, backend="aot_eager")
        return fn(carrier, scores, ids)

    def _run_plain_once(self, loss):
        """None AC: a single plain forward, accumulating the metric exactly once."""
        torch.manual_seed(0)
        scores_TE = torch.rand(24, 8, requires_grad=True)
        ids = torch.randint(0, 8, (24, 2))
        carrier = scores_TE.gather(dim=-1, index=ids)
        out = loss(carrier, scores_TE, ids).sum()
        out.backward()
        return loss._acc_sum.item()

    def _run_full_ac_once(self, loss, *, use_compile):
        """Full AC: the whole loss forward is recomputed on backward."""
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
        """Selective AC: only the loss's inputs (carrier-producing gather)
        are recomputed; the loss forward itself runs once."""
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

    def test_ac_modes_accumulate_identically(self):
        """none / full / selective (and full under compile) each accumulate
        exactly once per microbatch: every mode must match the plain reference
        value on identical inputs. A recompute path that added the metric
        twice would double the full/selective values relative to none."""
        torch.manual_seed(0)
        plain = self._run_plain_once(self._make_loss())

        def run(loss, mode):
            if mode == "none":
                return self._run_plain_once(loss)
            if mode == "full":
                return self._run_full_ac_once(loss, use_compile=False)
            if mode == "full_compiled":
                return self._run_full_ac_once(loss, use_compile=True)
            return self._run_selective_ac_once(loss)

        for mode in ("none", "full", "full_compiled", "selective"):
            with self.subTest(mode=mode):
                # Each mode draws the same inputs (fixed seed inside the
                # helpers), so the accumulated values must agree exactly.
                loss = self._make_loss()
                self.assertAlmostEqual(run(loss, mode), plain, places=6)


class TestSeqwiseCountsSpmdTypes(DTensorTestBase):
    """Sequence-wise counts reduction under spmd_types with real sharding.

    Runs dp=2, cp=2, tp=2, ep=2 (8 ranks) on the CPU: each rank holds a
    distinct (B/dp, L/(cp*tp)) slice of the global batch. The per-sequence
    counts must all-reduce over the CP and TP token axes within each dp slice
    -- with the dp-outermost mesh ordering, both per-axis groups stay inside
    the dp slice, so batch rows are never mixed -- and produce per-sequence
    stats identical to the reference computed on the global tensor, with
    matching gradients. The forward and backward run under the SPMD
    typechecker, so the boundary Partial -> Invariant redistributes and the
    identity backward are type-validated as well.
    """

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_reduction_values_and_gradients(self):
        """8-rank dp2/cp2/tp2/ep2 under typechecking, folded single-stream.

        With the fold-batch-dim training layout, each dp rank processes its
        own token stream (T_dp), which the loss treats as one sequence. The
        EP-enabled case makes per-token counts Partial over CP and TP and
        all-reduces both axes at the _SeqwiseCounts boundary; the EP-disabled
        case only all-reduces CP. Reference and gradient are computed per dp
        rank on its full 64-token stream.
        """
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.distributed.spmd_types import (
            set_current_spmd_mesh,
            set_spmd_meshes,
        )
        from torchtitan.distributed.utils import set_spmd_backend
        from torchtitan.models.common.decoder_sharding import (
            dense_activation_placement,
            dense_sequence_parallel_placement,
        )
        from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss

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

        T, E, K = 128, 8, 2  # T = global token stream of one train step
        dp, cp, tp = 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        tp_rank = self.rank % tp
        t_dp = T // dp  # tokens per dp rank
        t_blk = t_dp // (cp * tp)  # tokens per rank after CP/TP split
        dp_start = dp_rank * t_dp
        dp_end = dp_start + t_dp
        t_start = dp_start + (cp_rank * tp + tp_rank) * t_blk
        t_end = t_start + t_blk

        with set_current_spmd_mesh(dense_mesh), typecheck(local=False):
            torch.manual_seed(0)
            global_scores = torch.rand(T, E, dtype=torch.float64)
            global_ids = torch.randint(0, E, (T, K))
            with spmd.no_typecheck():
                local_scores = global_scores[t_start:t_end].contiguous()
                local_ids = global_ids[t_start:t_end].contiguous()

            router_out_layout = dense_sequence_parallel_placement()
            spmd.assert_type(local_scores, router_out_layout)
            spmd.assert_type(local_ids, router_out_layout)

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
                # Reference on this dp rank's full stream (its single sequence).
                dp_scores = global_scores[dp_start:dp_end]
                dp_ids = global_ids[dp_start:dp_end]
                routing_map = torch.zeros(t_dp, E).scatter_(-1, dp_ids, 1.0)
                counts_E = routing_map.sum(dim=0)
                probs = dp_scores / dp_scores.sum(dim=-1, keepdim=True)
                prob_sums_E = probs.sum(dim=0)
                num_tokens = counts_E.sum() / K
                f_E = counts_E * (E / (K * num_tokens))
                p_E = prob_sums_E / num_tokens
                ref_loss = (f_E * p_E).sum()

                self.assertAlmostEqual(loss._acc_sum.item(), ref_loss.item(), places=6)

                # Gradient: dL / ds on this dp rank's stream (L includes the
                # identity carrier term), compared on the local token slice.
                ref_scores = dp_scores.detach().clone().requires_grad_(True)
                rm = torch.zeros(t_dp, E).scatter_(-1, dp_ids, 1.0)
                cts = rm.sum(dim=0)
                prs = ref_scores / ref_scores.sum(dim=-1, keepdim=True)
                nt = cts.sum() / K
                f = cts * (E / (K * nt))
                p = prs.sum(dim=0) / nt
                ref_aux = (f * p).sum() * 0.1
                ref_carrier = ref_scores.gather(dim=-1, index=dp_ids)
                (ref_aux + ref_carrier.sum()).backward()
                rel_start = (cp_rank * tp + tp_rank) * t_blk
                ref_local_grad = ref_scores.grad[rel_start : rel_start + t_blk]

                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )

            # Same validation for the EP-disabled layout: only CP shards
            # the token stream, TP carries a Replicate copy, so the child
            # boundary performs a single CP all-reduce.
            no_ep_blk = t_dp // cp
            no_ep_start = dp_start + cp_rank * no_ep_blk
            no_ep_end = no_ep_start + no_ep_blk
            with spmd.no_typecheck():
                no_ep_scores = global_scores[no_ep_start:no_ep_end].contiguous()
                no_ep_ids = global_ids[no_ep_start:no_ep_end].contiguous()
            no_ep_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            spmd.assert_type(no_ep_scores, no_ep_layout)
            spmd.assert_type(no_ep_ids, no_ep_layout)

            no_ep_loss = SeqwiseLoadBalanceLoss(
                _seqwise_loss_config(K, 1, enable_ep=False)
            )
            no_ep_loss.parallelize(parallel_dims)
            no_ep_scores.requires_grad_(True)
            no_ep_carrier = no_ep_scores.gather(dim=-1, index=no_ep_ids)
            no_ep_out = no_ep_loss(no_ep_carrier, no_ep_scores, no_ep_ids)
            with spmd.no_typecheck():
                torch.testing.assert_close(no_ep_out, no_ep_carrier, rtol=0, atol=0)
                # The carrier is Replicate on TP in this layout, so an implicit
                # scalar backward is intentionally opaque, matching the trainer
                # (`loss.backward()` runs under spmd.no_typecheck()).
                no_ep_out.sum().backward()

            with spmd.no_typecheck():
                ref_no_ep_grad = ref_scores.grad[
                    cp_rank * no_ep_blk : (cp_rank + 1) * no_ep_blk
                ]
                self.assertAlmostEqual(
                    no_ep_loss._acc_sum.item(), ref_loss.item(), places=6
                )
                self.assertLess(
                    (no_ep_scores.grad - ref_no_ep_grad).abs().max().item(), 1e-10
                )


if __name__ == "__main__":
    unittest.main()
