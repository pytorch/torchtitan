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

from torchtitan.models.common.aux_loss import _zero_aux_losses, LoggedAuxLoss
from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss


def _reference_seqwise_aux_loss(
    scores_BLE: torch.Tensor,
    topk_expert_ids_BLK: torch.Tensor,
    top_k: int,
    coeff: float,
    per_step_denominator: int,
) -> torch.Tensor:
    """Explicit DeepSeek-V3 seq-wise aux loss (Sec A.2 Eqs 17-20), framework-scaled."""
    E = scores_BLE.size(-1)
    routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
        -1, topk_expert_ids_BLK, True
    )
    counts_BE = routing_map_BLE.sum(dim=1).to(scores_BLE.dtype)
    probs_BLE = scores_BLE / scores_BLE.sum(dim=-1, keepdim=True)
    prob_sums_BE = probs_BLE.sum(dim=1)
    num_tokens_B = counts_BE.sum(dim=1) / top_k
    f_BE = counts_BE * (E / (top_k * num_tokens_B.unsqueeze(1)))
    p_BE = prob_sums_BE / num_tokens_B.unsqueeze(1)
    loss_per_seq_B = (f_BE * p_BE).sum(dim=1)
    return loss_per_seq_B.sum() * (coeff / per_step_denominator)


def _make_loss_module(top_k: int, coeff: float, per_step_denominator: int):
    # ``per_step_denominator`` is not user config: the decoder's
    # ``update_from_config`` fills it before modules are built.
    cfg = SeqwiseLoadBalanceLoss.Config(coeff=coeff, top_k=top_k)
    cfg.per_step_denominator = per_step_denominator
    loss = SeqwiseLoadBalanceLoss(cfg)
    loss.train()
    return loss


def _seqwise_loss_config(top_k: int, per_step_denominator: int, *, enable_ep: bool):
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
        self.B, self.L, self.E, self.K = 3, 5, 7, 2
        self.coeff = 0.125
        # SeqwiseLoadBalanceLoss normalizes by the batch size in sequences
        # (DS-V3 Eq 17 convention), not by tokens.
        self.per_step_denominator = self.B
        torch.manual_seed(0)
        _set_spmd_types_backend()

    def tearDown(self):
        # The backend global and metric registry are process-wide; restore
        # them so other test modules are unaffected.
        _restore_default_backend()
        _clear_aux_loss_registry()

    def _make_inputs(self):
        scores_BLE = torch.rand(
            self.B, self.L, self.E, dtype=torch.float32, requires_grad=True
        )
        topk_expert_ids_BLK = torch.topk(
            scores_BLE.detach(), k=self.K, dim=-1, sorted=False
        ).indices
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)
        return scores_BLE, topk_scores_BLK, topk_expert_ids_BLK

    def test_loss_value_matches_formula(self):
        scores_BLE, _, topk_expert_ids_BLK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        out = loss(scores_BLE, scores_BLE, topk_expert_ids_BLK)

        # The forward returns the carrier unchanged (identity injection).
        self.assertEqual(out.shape, scores_BLE.shape)
        torch.testing.assert_close(out, scores_BLE, rtol=0, atol=0)

        # The metric accumulates in the backward (see _AuxLossInjection); the
        # accumulated value is the raw per-sequence mean, i.e. the injected
        # gradient divided by the coefficient.
        out.sum().backward()
        expected = _reference_seqwise_aux_loss(
            scores_BLE,
            topk_expert_ids_BLK,
            self.K,
            1.0,
            self.per_step_denominator,
        )
        torch.testing.assert_close(loss._acc_sum, expected, rtol=1e-12, atol=1e-12)

    def test_gradient_matches_explicit_loss(self):
        """Injected aux loss must match adding it to the main loss explicitly."""
        scores_BLE, topk_scores_BLK, topk_expert_ids_BLK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        injected_scores = scores_BLE.clone().detach().requires_grad_(True)
        injected_topk = injected_scores.gather(dim=-1, index=topk_expert_ids_BLK)
        injected = loss(injected_topk, injected_scores, topk_expert_ids_BLK)
        injected.sum().backward()

        explicit_scores = scores_BLE.clone().detach().requires_grad_(True)
        explicit_topk = explicit_scores.gather(dim=-1, index=topk_expert_ids_BLK)
        explicit_aux = _reference_seqwise_aux_loss(
            explicit_scores,
            topk_expert_ids_BLK,
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
        scores_BLE, _, topk_expert_ids_BLK = self._make_inputs()
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)
        out = loss(scores_BLE, scores_BLE, topk_expert_ids_BLK)
        out.sum().backward()
        self.assertIsNotNone(scores_BLE.grad)
        self.assertTrue(torch.isfinite(scores_BLE.grad).all())
        self.assertGreater(scores_BLE.grad.abs().sum().item(), 0)


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
        scores_BLE = torch.rand(4, 6, 8, requires_grad=True)
        ids = torch.randint(0, 8, (4, 6, 2))
        carrier = scores_BLE.gather(dim=-1, index=ids)
        out = loss(carrier, scores_BLE, ids).sum()
        out.backward()
        return loss._acc_sum.item()

    def _run_full_ac_once(self, loss, *, use_compile):
        """Full AC: the whole loss forward is recomputed on backward."""
        torch.manual_seed(0)
        scores_BLE = torch.rand(4, 6, 8, requires_grad=True)
        ids = torch.randint(0, 8, (4, 6, 2))
        carrier = scores_BLE.gather(dim=-1, index=ids)
        out = self._forward_once(
            loss,
            carrier,
            scores_BLE,
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
        scores_BLE = torch.rand(4, 6, 8, requires_grad=True)
        ids = torch.randint(0, 8, (4, 6, 2))

        def pre(c, i):
            return c.gather(dim=-1, index=i)

        def fwd(c, i):
            carrier = torch.utils.checkpoint.checkpoint(pre, c, i, use_reentrant=False)
            return loss(carrier, c, i).sum()

        fwd(scores_BLE, ids).backward()
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
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.distributed.spmd_types import (
            set_current_spmd_mesh,
            set_spmd_meshes,
        )
        from torchtitan.distributed.utils import set_spmd_backend
        from torchtitan.models.common.decoder_sharding import (
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

        B, L, E, K = 4, 32, 8, 2
        dp, cp, tp = 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        tp_rank = self.rank % tp
        l_blk = L // (cp * tp)
        l_start = (cp_rank * tp + tp_rank) * l_blk
        l_end = l_start + l_blk
        b_local = B // dp
        b_start = dp_rank * b_local
        b_end = b_start + b_local

        with set_current_spmd_mesh(dense_mesh), typecheck(local=False):
            torch.manual_seed(0)
            # float64 data so the value and gradient checks run at near-exact
            # precision: the only float32 bottleneck is the metric accumulator.
            global_scores = torch.rand(B, L, E, dtype=torch.float64)
            global_ids = torch.randint(0, E, (B, L, K))
            with spmd.no_typecheck():
                local_scores = global_scores[b_start:b_end, l_start:l_end].contiguous()
                local_ids = global_ids[b_start:b_end, l_start:l_end].contiguous()

            router_out_layout = dense_sequence_parallel_placement()
            spmd.assert_type(
                local_scores,
                router_out_layout.axis_types,
                router_out_layout.partition_spec,
            )
            spmd.assert_type(
                local_ids,
                router_out_layout.axis_types,
                router_out_layout.partition_spec,
            )

            loss = SeqwiseLoadBalanceLoss(_seqwise_loss_config(K, B, enable_ep=True))
            loss.parallelize(parallel_dims)

            local_scores.requires_grad_(True)
            carrier = local_scores.gather(dim=-1, index=local_ids)
            out = loss(carrier, local_scores, local_ids)
            with spmd.no_typecheck():
                torch.testing.assert_close(out, carrier, rtol=0, atol=0)
            # The metric accumulates in the backward (see _AuxLossInjection).
            out.sum().backward()
            local_scores.grad = None

            # Backward for the gradient check: a fresh module so the step
            # accumulator only covers the first forward.
            loss_grad = SeqwiseLoadBalanceLoss(
                _seqwise_loss_config(K, B, enable_ep=True)
            )
            loss_grad.parallelize(parallel_dims)
            carrier_g = local_scores.gather(dim=-1, index=local_ids)
            out2 = loss_grad(carrier_g, local_scores, local_ids)
            out2.sum().backward()

            with spmd.no_typecheck():
                # Reference on the global tensor (identical on every rank).
                routing_map = torch.zeros(B, L, E).scatter_(-1, global_ids, 1.0)
                counts = routing_map.sum(dim=1)
                probs = global_scores / global_scores.sum(dim=-1, keepdim=True)
                num_tokens_B = counts.sum(dim=1) / K
                f_BE = counts * (E / (K * num_tokens_B.unsqueeze(1)))
                p_BE = probs.sum(dim=1) / num_tokens_B.unsqueeze(1)
                ref_loss_per_seq = (f_BE * p_BE).sum(dim=1)

                # Each rank's accumulated value = sum over its local rows
                # (per-sequence normalization: scaled by 1 / global_B). The
                # math runs in float64 but the metric accumulator is float32,
                # capping the match at ~1e-6 absolute.
                expected_sum = ref_loss_per_seq[b_start:b_end].sum().item()
                actual_sum = loss._acc_sum.item() * B
                self.assertAlmostEqual(expected_sum, actual_sum, places=6)

                # Gradient: dL_global / ds for the local token slices, where
                # L_global includes the identity carrier term.
                ref_scores = global_scores.detach().clone().requires_grad_(True)
                rm = torch.zeros(B, L, E).scatter_(-1, global_ids, 1.0)
                cts = rm.sum(dim=1)
                prs = ref_scores / ref_scores.sum(dim=-1, keepdim=True)
                nt = cts.sum(dim=1) / K
                f = cts * (E / (K * nt.unsqueeze(1)))
                p = prs.sum(dim=1) / nt.unsqueeze(1)
                ref_aux = (f * p).sum() * (0.1 / B)
                ref_carrier = ref_scores.gather(dim=-1, index=global_ids)
                (ref_aux + ref_carrier.sum()).backward()
                ref_local_grad = ref_scores.grad[b_start:b_end, l_start:l_end]

                # float64 backward: the identity P->I gradient must match the
                # analytic derivative to near machine precision (the observed
                # diff ~5e-12 is float64 roundoff over the reduction chain,
                # many orders below the ~1e-3 gradient magnitudes).
                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )


if __name__ == "__main__":
    unittest.main()
