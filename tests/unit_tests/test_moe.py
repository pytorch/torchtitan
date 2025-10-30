# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from einops import rearrange
from triton.testing import do_bench

from torchtitan.models.moe.moe import FeedForward, MoE, MoEArgs, MoEOld


# NOTE: @goon -  torch.testing.assert_close is very strict and hard to pass. Use the more-lenient
# assert_close from FLA, slightly modified.
# https://github.com/fla-org/flash-linear-attention/blob/3ddba2a043100837a1f6499b5eb6692de71a477b/fla/utils.py?plain=1#L82
def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    assert error_rate < ratio, msg


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
    ) -> tuple[MoEOld, MoE]:
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
        moe_old = MoEOld(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        moe = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )

        moe_old.init_weights(1 / self.dim**0.5, self.device)

        with torch.no_grad():
            # Set the MoE model params equal to the MoEOld ones.
            for p, p2 in zip(moe_old.parameters(), moe.parameters(), strict=True):
                p2.data.copy_(p.data)

        return moe_old, moe

    def _get_equiv_layers(self) -> tuple[MoEOld, MoE, FeedForward]:
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
        moe_old = MoEOld(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
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
            # Set the MoE model params equal to the MoEOld ones.
            for p, p2 in zip(moe_old.parameters(), moe.parameters(), strict=True):
                p2.data.copy_(p.data)

        return moe_old, moe, ffn

    def test_moe_old_moe_equivalence(
        self, score_before_experts: bool = True
    ) -> tuple[float, float]:
        torch.manual_seed(42)
        moe_old, moe = self._get_moe_old_and_moe_layers(score_before_experts)
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
            assert_close(
                "moe_old vs moe", out_moe_old, out_moe, self.assert_close_ratio
            )

    def test_moe_ffn_equivalence(self, iteration: int = 0) -> tuple[float, float]:
        torch.manual_seed(42 + iteration)
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
            return moe_old_rel_err, moe_rel_err

    def test_perf(
        self, bsz: int | None = None, seqlen: int | None = None
    ) -> tuple[float, float]:
        seqlen = seqlen or self.seqlen
        bsz = bsz or self.bsz
        torch.manual_seed(42)
        moe_old, moe = self._get_moe_old_and_moe_layers()
        inputs = torch.randn(
            bsz,
            seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        moe_old_time_ms = do_bench(
            lambda: moe_old(inputs).sum().backward(),
            warmup=self.perf_warmups,
            rep=self.perf_reps,
        )
        moe_time_ms = do_bench(
            lambda: moe(inputs).sum().backward(),
            warmup=self.perf_warmups,
            rep=self.perf_reps,
        )
        print(f"{moe_old_time_ms=}")
        print(f"{moe_time_ms=}")
        print(f"Speedup: {moe_old_time_ms/moe_time_ms=}")


if __name__ == "__main__":
    t = TestModel()

    # Collect some accuracy stats
    moe_old_rel_errs = []
    moe_rel_errs = []
    accuracy_iters = 10
    for idx in range(accuracy_iters):
        moe_old_rel_err, moe_rel_err = t.test_moe_ffn_equivalence(idx)
        moe_old_rel_errs.append(moe_old_rel_err)
        moe_rel_errs.append(moe_rel_err)
    mean_moe_old_rel_err = torch.tensor(moe_old_rel_errs)
    mean_moe_rel_err = torch.tensor(moe_rel_errs)

    print(f"\nACCURACY VS FFN: {accuracy_iters} iterations\n")
    print(f"{mean_moe_old_rel_err.mean()=}, {mean_moe_old_rel_err.std()=}")
    print(f"{mean_moe_rel_err.mean()=}, {mean_moe_rel_err.std()=}")
    print(f"{mean_moe_old_rel_err.mean()/mean_moe_rel_err.mean()=}")

    # Perf bsz and seqlen as in torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml
    perf_seqlen = 4096
    perf_bsz = 4
    print(
        f"\nTRITON BENCH: {perf_seqlen=} {perf_bsz=} warmups={t.perf_warmups} repeats={t.perf_reps}\n"
    )
    t.test_perf(bsz=perf_bsz, seqlen=perf_seqlen)

    t.test_moe_old_moe_equivalence(True)
    print("\nMoEOld AND MoE CLOSE: score_before_experts=True")
    t.test_moe_old_moe_equivalence(False)
    print("\nMoEOld AND MoE CLOSE: score_before_experts=False")
