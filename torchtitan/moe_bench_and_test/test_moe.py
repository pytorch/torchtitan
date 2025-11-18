# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from einops import rearrange

from torchtitan.models.moe.moe import FeedForward, MoE, MoEArgs
from torchtitan.moe_bench_and_test import (
    apply_old_moe_monkey_patches,
    assert_close,
    get_err_ratio,
)


class TestModel:
    # Default following DSv3 16B
    assert_close_ratio = 1e-2
    bsz = 2
    device = "cuda"
    dim = 2048
    is_moe_list = None
    moe_inter_dim = 1408
    num_experts = 64
    num_shared_experts = 2
    perf_reps = 1000
    perf_warmups = 100
    route_norm = False
    score_before_experts = False
    seqlen = 64
    top_k = 6
    use_grouped_mm = True

    def _get_moe_old_and_moe_layers(
        self, score_before_experts: bool | None = None
    ) -> tuple[MoE, MoE]:
        """
        Create MoEOld and MOE layers with equivalent parameters.
        """
        score_before_experts = score_before_experts or self.score_before_experts
        moe_args = MoEArgs(
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            score_func="softmax",
            route_norm=self.route_norm,
            score_before_experts=score_before_experts,
            top_k=self.top_k,
            use_grouped_mm=self.use_grouped_mm,
        )
        moe_old = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        apply_old_moe_monkey_patches(moe_old)
        moe = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )

        moe_old.init_weights(1 / self.dim**0.5, self.device)

        with torch.no_grad():
            # Set the MoE model params equal
            for p, p2 in zip(
                moe_old.parameters(),
                moe.parameters(),
                strict=True,
            ):
                p2.data.copy_(p.data)

        return moe_old, moe

    def _get_equiv_layers(self) -> tuple[MoE, MoE, FeedForward]:
        """
        Create MoEOld, MOE, and FeedForward layers which are all configured so that they should have
        the same outputs. Accomplished by breaking the FeedForward weights into experts, choosing
        top_k = num_shared_experts, and ensuring that the router gives every expert weight 1.
        """
        top_k = 4
        moe_args = MoEArgs(
            num_experts=self.num_experts,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
            top_k=self.num_experts,
            route_scale=self.num_experts,  # Required for equivalence
            use_grouped_mm=self.use_grouped_mm,
        )
        moe_old = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        apply_old_moe_monkey_patches(moe_old)
        moe = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        ffn = FeedForward(
            dim=self.dim, hidden_dim=self.moe_inter_dim * self.num_experts
        ).to(device=self.device, dtype=torch.bfloat16)

        moe_old.init_weights(1 / self.dim**0.5, self.device)
        ffn.init_weights(1 / self.dim**0.5)

        with torch.no_grad():
            ffn_w1_experts = rearrange(
                ffn.w1.weight, "(e h) d -> e h d", e=self.num_experts
            )
            ffn_w2_experts = rearrange(
                ffn.w2.weight, "d (e h) -> e d h", e=self.num_experts
            )
            ffn_w3_experts = rearrange(
                ffn.w3.weight, "(e h) d -> e h d", e=self.num_experts
            )
            moe_old.experts.w1.data.copy_(ffn_w1_experts)
            moe_old.experts.w2.data.copy_(ffn_w2_experts)
            moe_old.experts.w3.data.copy_(ffn_w3_experts)

            # Zero out the router weights, so every expert has equal weighting.
            moe_old.router.gate.weight.zero_()
            # Set the MoE model params equal
            for p, p2 in zip(moe_old.parameters(), moe.parameters(), strict=True):
                p2.data.copy_(p.data)

        return moe_old, moe, ffn

    @pytest.mark.parametrize("score_before_experts", [False, True])
    def test_moe_equivalence(self, score_before_experts: bool) -> None:
        torch.manual_seed(42)
        moe_old, moe = self._get_moe_old_and_moe_layers(score_before_experts)
        inputs = torch.randn(
            self.bsz,
            self.seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        inputs_moe_old = inputs.clone().requires_grad_()
        inputs_moe = inputs.clone().requires_grad_()

        out_moe_old = moe_old(inputs_moe_old)
        out_moe = moe(inputs_moe)

        assert_close("moe_old vs moe", out_moe_old, out_moe, self.assert_close_ratio)

        out_moe_old.pow(2).mean().backward()
        out_moe.pow(2).mean().backward()

        for (name, p1), (_, p2) in zip(
            moe_old.named_parameters(),
            moe.named_parameters(),
            strict=True,
        ):
            assert_close(
                f"{name} grad",
                p1.grad,
                p2.grad,
                self.assert_close_ratio,
            )
        assert_close(
            "input clone grad",
            inputs_moe_old.grad,
            inputs_moe.grad,
            self.assert_close_ratio,
        )

    def test_moe_ffn_equivalence(self) -> None:
        torch.manual_seed(42)
        moe_old, moe, ffn = self._get_equiv_layers()
        with torch.no_grad():
            inputs = torch.randn(
                self.bsz,
                self.seqlen,
                self.dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            out_moe_old = moe_old(inputs)
            out_moe = moe(inputs)
            out_ffn = ffn(inputs)

            assert_close(
                "moe_old vs ffn", out_ffn, out_moe_old, self.assert_close_ratio
            )
            assert_close("moe vs ffn", out_ffn, out_moe, self.assert_close_ratio)

            moe_old_rel_err = get_err_ratio(out_ffn, out_moe_old)
            moe_rel_err = get_err_ratio(out_ffn, out_moe)
            print(f"{moe_old_rel_err=}")
            print(f"{moe_rel_err=}")
            print(f"{moe_old_rel_err/moe_rel_err=}")

    def test_determinism(self):
        torch.manual_seed(42)
        moe_old, moe = self._get_moe_old_and_moe_layers(score_before_experts=False)
        inputs = torch.randn(
            self.bsz,
            self.seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        out_moe_empty_like = []
        out_moe_old = []
        out_moe = []
        with torch.no_grad():
            for _ in range(100):
                out_moe_old.append(moe_old(inputs))
                out_moe.append(moe(inputs))

        out_moe_old = torch.stack(out_moe_old, dim=0)
        out_moe = torch.stack(out_moe, dim=0)

        out_old_std = out_moe_old.std(dim=0).mean()
        out_std = out_moe.std(dim=0).mean()

        print(f"{out_old_std=}")
        print(f"{out_std=}")
        # And relative to the mean element size:
        print(f"{out_old_std/out_moe_old.abs().mean()=}")
        print(f"{out_std/out_moe.abs().mean()=}")

        torch.testing.assert_close(out_std, torch.zeros_like(out_std))
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out_old_std, torch.zeros_like(out_old_std))
