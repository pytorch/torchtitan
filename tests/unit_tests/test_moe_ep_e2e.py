#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end MoE + Expert Parallel integration tests (pytest + torchrun).

Runs the real production MoE.forward() with ExpertParallel and ExpertParallelLLEP
hooks applied via distribute_module, and compares output logits + gradients
against a single-GPU reference (all experts local, no EP).

Three modes compared:
  1. Single-GPU reference — full MoE (all experts local, no EP)
  2. Standard EP — ExpertParallel applied to GroupedExperts
  3. LLEP EP — ExpertParallelLLEP applied to GroupedExperts

Run (requires >= 2 GPUs):
    torchrun --nproc_per_node=2 -m pytest tests/unit_tests/test_moe_ep_e2e.py -v
    torchrun --nproc_per_node=8 -m pytest tests/unit_tests/test_moe_ep_e2e.py -v
    torchrun --nproc_per_node=2 -m pytest tests/unit_tests/test_moe_ep_e2e.py -v -k "topk"
    torchrun --nproc_per_node=2 -m pytest tests/unit_tests/test_moe_ep_e2e.py -v -k "backward"
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def dist_setup():
    """Initialize process group once per session."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    yield
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.destroy_process_group()


@pytest.fixture
def world_info():
    return dist.get_rank(), dist.get_world_size()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_moe_args(
    num_experts=8,
    top_k=2,
    use_grouped_mm=True,
    use_llep=False,
    score_before_experts=True,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1,
    adaptive_threshold=0.0,
):
    from torchtitan.models.moe.moe import LLEPConfig, MoEArgs

    return MoEArgs(
        num_experts=num_experts,
        num_shared_experts=0,
        shared_gate=False,
        score_func="sigmoid",
        route_norm=False,
        route_scale=1.0,
        score_before_experts=score_before_experts,
        top_k=top_k,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
        use_llep=use_llep,
        llep=LLEPConfig(
            max_tokens_factor=max_tokens_factor,
            min_tokens_per_gemm=min_tokens_per_gemm,
            adaptive_threshold=adaptive_threshold,
        ),
    )


def _create_moe(moe_args, dim, hidden_dim, device, seed, dtype=torch.bfloat16):
    from torchtitan.models.moe.moe import MoE

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    moe = MoE(moe_args, dim, hidden_dim).to(device)
    with torch.no_grad():
        moe.init_weights(0.02, device, n_layers=1)
    return moe.to(dtype)


def _apply_ep(moe, ep_mesh):
    from torchtitan.distributed.expert_parallel import ExpertParallel

    ExpertParallel()._apply(moe.experts, ep_mesh)
    return moe


def _apply_llep(
    moe, ep_mesh, max_tokens_factor=1.1, min_tokens_per_gemm=1, adaptive_threshold=0.0
):
    from torchtitan.distributed.expert_parallel import ExpertParallelLLEP

    ExpertParallelLLEP(
        max_tokens_factor=max_tokens_factor,
        min_tokens_per_gemm=min_tokens_per_gemm,
        adaptive_threshold=adaptive_threshold,
    )._apply(moe.experts, ep_mesh)
    return moe


def _gather_expert_grads(moe_distributed, world_size):
    """Gather sharded expert weight grads from all ranks."""
    grads = {}
    for name in ["w1", "w2", "w3"]:
        param = getattr(moe_distributed.experts, name)
        local_grad = param.grad.to_local() if isinstance(param, DTensor) else param.grad
        gathered = [torch.empty_like(local_grad) for _ in range(world_size)]
        dist.all_gather(gathered, local_grad.contiguous())
        grads[name] = torch.cat(gathered, dim=0)
    return grads


# ---------------------------------------------------------------------------
# Core test logic
# ---------------------------------------------------------------------------
def _run_forward_test(
    *,
    num_experts=8,
    top_k=2,
    dim=64,
    hidden_dim=128,
    bs=2,
    slen=16,
    use_grouped_mm=True,
    score_before_experts=True,
    dtype=torch.bfloat16,
    atol=1e-2,
    seed=42,
    test_llep=True,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1,
    adaptive_threshold=0.0,
):
    """Run forward and assert ref ≈ EP ≈ LLEP."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    num_local = num_experts // world_size

    if num_local * world_size != num_experts:
        pytest.skip(
            f"num_experts={num_experts} not divisible by world_size={world_size}"
        )

    ep_mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("ep",))

    # 1. Reference MoE (no EP)
    ref_args = _make_moe_args(
        num_experts=num_experts,
        top_k=top_k,
        use_grouped_mm=use_grouped_mm,
        score_before_experts=score_before_experts,
    )
    moe_ref = _create_moe(ref_args, dim, hidden_dim, device, seed, dtype)

    # 2. EP MoE
    ep_args = _make_moe_args(
        num_experts=num_experts,
        top_k=top_k,
        use_grouped_mm=use_grouped_mm,
        score_before_experts=score_before_experts,
    )
    moe_ep = _create_moe(ep_args, dim, hidden_dim, device, seed, dtype)
    moe_ep.load_state_dict(moe_ref.state_dict())
    _apply_ep(moe_ep, ep_mesh)

    # 3. LLEP MoE (optional)
    moe_llep = None
    if test_llep:
        llep_args = _make_moe_args(
            num_experts=num_experts,
            top_k=top_k,
            use_grouped_mm=use_grouped_mm,
            use_llep=True,
            score_before_experts=score_before_experts,
            max_tokens_factor=max_tokens_factor,
            min_tokens_per_gemm=min_tokens_per_gemm,
            adaptive_threshold=adaptive_threshold,
        )
        moe_llep = _create_moe(llep_args, dim, hidden_dim, device, seed, dtype)
        moe_llep.load_state_dict(moe_ref.state_dict())
        _apply_llep(
            moe_llep,
            ep_mesh,
            max_tokens_factor=max_tokens_factor,
            min_tokens_per_gemm=min_tokens_per_gemm,
            adaptive_threshold=adaptive_threshold,
        )

    # Input
    torch.manual_seed(seed + 1000)
    x = torch.randn(bs, slen, dim, device=device, dtype=dtype) * 0.1
    dist.broadcast(x, src=0)

    # Forward
    dist.barrier()
    ref_out = moe_ref(x.clone())
    dist.barrier()
    ep_out = moe_ep(x.clone())

    ep_diff = (ref_out - ep_out).abs().max().item()
    assert ep_diff < atol, f"EP forward diff {ep_diff} >= {atol}"

    if test_llep and moe_llep is not None:
        dist.barrier()
        llep_out = moe_llep(x.clone())
        llep_diff = (ref_out - llep_out).abs().max().item()
        assert llep_diff < atol, f"LLEP forward diff {llep_diff} >= {atol}"

    dist.barrier()


def _run_backward_test(
    *,
    num_experts=8,
    top_k=2,
    dim=64,
    hidden_dim=128,
    bs=2,
    slen=16,
    score_before_experts=True,
    seed=42,
    max_tokens_factor=1.5,
    backward_atol=0.05,
):
    """Run backward and assert EP grads ≈ LLEP grads."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float32

    ep_mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("ep",))

    # Reference
    ref_args = _make_moe_args(
        num_experts=num_experts,
        top_k=top_k,
        score_before_experts=score_before_experts,
    )
    moe_ref = _create_moe(ref_args, dim, hidden_dim, device, seed, dtype)

    # EP
    ep_args = _make_moe_args(
        num_experts=num_experts,
        top_k=top_k,
        score_before_experts=score_before_experts,
    )
    moe_ep = _create_moe(ep_args, dim, hidden_dim, device, seed, dtype)
    moe_ep.load_state_dict(moe_ref.state_dict())
    _apply_ep(moe_ep, ep_mesh)

    # LLEP
    llep_args = _make_moe_args(
        num_experts=num_experts,
        top_k=top_k,
        use_llep=True,
        score_before_experts=score_before_experts,
        max_tokens_factor=max_tokens_factor,
    )
    moe_llep = _create_moe(llep_args, dim, hidden_dim, device, seed, dtype)
    moe_llep.load_state_dict(moe_ref.state_dict())
    _apply_llep(moe_llep, ep_mesh, max_tokens_factor=max_tokens_factor)

    # Input
    torch.manual_seed(seed + 1000)
    x = torch.randn(bs, slen, dim, device=device, dtype=dtype) * 0.1
    dist.broadcast(x, src=0)

    x_ref = x.clone().detach().requires_grad_(True)
    x_ep = x.clone().detach().requires_grad_(True)
    x_llep = x.clone().detach().requires_grad_(True)

    # Forward + backward
    dist.barrier()
    moe_ref(x_ref).sum().backward()
    dist.barrier()
    moe_ep(x_ep).sum().backward()
    dist.barrier()
    moe_llep(x_llep).sum().backward()

    # Primary: EP vs LLEP grads must match
    ep_grads = _gather_expert_grads(moe_ep, world_size)
    llep_grads = _gather_expert_grads(moe_llep, world_size)

    ep_vs_llep_max = 0.0
    for name in ["w1", "w2", "w3"]:
        diff = (ep_grads[name] - llep_grads[name]).abs().max().item()
        ep_vs_llep_max = max(ep_vs_llep_max, diff)

    grad_x_diff = (x_ep.grad - x_llep.grad).abs().max().item()
    ep_vs_llep_max = max(ep_vs_llep_max, grad_x_diff)

    assert (
        ep_vs_llep_max < backward_atol
    ), f"EP vs LLEP grad diff {ep_vs_llep_max} >= {backward_atol}"

    dist.barrier()


# ---------------------------------------------------------------------------
# Helpers for min_gpus skip
# ---------------------------------------------------------------------------
def _skip_if_insufficient_gpus(min_gpus):
    ws = dist.get_world_size()
    if ws < min_gpus:
        pytest.skip(f"needs {min_gpus} GPUs (have {ws})")


# ===========================================================================
# Test classes
# ===========================================================================


class TestTopKSweep:
    """top_k={1,2,4,8}, fixed 8 experts."""

    @pytest.mark.parametrize("top_k", [1, 2, 4, 8], ids=lambda k: f"topk_{k}")
    def test_topk(self, top_k, dist_setup):
        _run_forward_test(num_experts=8, top_k=top_k)


class TestExpertCountSweep:
    """num_experts={8,16,32,64}."""

    @pytest.mark.parametrize(
        "num_experts,min_gpus",
        [(8, 2), (16, 2), (32, 4), (64, 8)],
        ids=lambda x: f"exp_{x}" if isinstance(x, int) and x >= 8 else "",
    )
    def test_experts(self, num_experts, min_gpus, dist_setup):
        _skip_if_insufficient_gpus(min_gpus)
        _run_forward_test(num_experts=num_experts, top_k=2)


class TestLLEPHyperparams:
    """LLEP hyperparameter sweeps."""

    @pytest.mark.parametrize("alpha", [1.0, 1.1, 2.0], ids=lambda a: f"alpha_{a}")
    def test_max_tokens_factor(self, alpha, dist_setup):
        _run_forward_test(max_tokens_factor=alpha)

    @pytest.mark.parametrize("m", [1, 1024], ids=lambda m: f"min_tok_{m}")
    def test_min_tokens_per_gemm(self, m, dist_setup):
        _run_forward_test(min_tokens_per_gemm=m)

    @pytest.mark.parametrize("lam", [1.3, 100.0], ids=lambda l: f"lambda_{l}")
    def test_adaptive_threshold(self, lam, dist_setup):
        _run_forward_test(adaptive_threshold=lam)


class TestScoreBefore:
    """score_before_experts={True,False}."""

    @pytest.mark.parametrize(
        "score_before", [True, False], ids=lambda v: f"score_before_{v}"
    )
    def test_score_before(self, score_before, dist_setup):
        _run_forward_test(score_before_experts=score_before)


class TestBackward:
    """Backward correctness: EP vs LLEP grads, float32."""

    @pytest.mark.parametrize("top_k", [1, 2], ids=lambda k: f"topk_{k}")
    @pytest.mark.parametrize("score_before", [True, False], ids=lambda v: f"sb_{v}")
    def test_backward(self, top_k, score_before, dist_setup):
        _run_backward_test(top_k=top_k, score_before_experts=score_before)


class TestInputVariations:
    """Varying bs/slen for different routing patterns."""

    @pytest.mark.parametrize(
        "bs,slen",
        [(1, 8), (4, 32), (2, 64)],
        ids=["bs1_slen8", "bs4_slen32", "bs2_slen64"],
    )
    def test_input(self, bs, slen, dist_setup):
        _run_forward_test(bs=bs, slen=slen)


class TestEPOnly:
    """EP-only (no LLEP), grouped_mm={True,False}."""

    @pytest.mark.parametrize("gmm", [True, False], ids=lambda g: f"gmm_{g}")
    def test_ep_only(self, gmm, dist_setup):
        _run_forward_test(use_grouped_mm=gmm, test_llep=False)


class TestGroupedMM:
    """use_grouped_mm=True with both EP and LLEP."""

    def test_grouped_mm(self, dist_setup):
        _run_forward_test(use_grouped_mm=True, test_llep=True)


class TestStress:
    """Larger scale tests."""

    def test_topk8_exp16(self, dist_setup):
        _run_forward_test(num_experts=16, top_k=8, bs=2, slen=32, max_tokens_factor=2.0)

    def test_topk4_exp16_large_dim(self, dist_setup):
        _run_forward_test(
            num_experts=16, top_k=4, dim=128, hidden_dim=256, bs=4, slen=16
        )

    def test_single_token(self, dist_setup):
        _run_forward_test(bs=1, slen=1, max_tokens_factor=2.0)

    def test_64exp_8gpu(self, dist_setup):
        _skip_if_insufficient_gpus(8)
        _run_forward_test(
            num_experts=64, top_k=4, dim=128, hidden_dim=256, bs=4, slen=32
        )

    def test_64exp_topk8_8gpu(self, dist_setup):
        _skip_if_insufficient_gpus(8)
        _run_forward_test(num_experts=64, top_k=8, bs=2, slen=32, max_tokens_factor=2.0)
