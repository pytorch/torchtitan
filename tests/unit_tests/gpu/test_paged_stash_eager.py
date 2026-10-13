# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the eager trainer's MoE paged stashing frontend.

A toy MoE stands in for the routed experts: a static-capacity dispatch custom
op that publishes its padded and live row counts to the HybridEP dispatch
observers, grouped GEMMs over the live rows, and a combine. Every test checks
the loss and every gradient bit for bit against the same model without paged
stashing:

* without activation checkpointing, and under full
  checkpointing, whose recompute replays the region from its stashes -- with
  and without an op after the region that makes the recompute pass through it;
* in a CUDA graph replayed with live row counts other than the captured ones;
* for MXFP8-style saves: float8 data and a flat e8m0 block-scale buffer;
* for pipeline metadata inference, which keeps the region's saves resident;
* for a step whose routing outgrows the stash, which the ``blocking`` overflow
  check reruns eagerly with paging disabled, dropping the CUDA graph, and
  measures, so that the stash grows to fit it.

Run with::

    pytest tests/unit_tests/gpu/test_paged_stash_eager.py
"""

import copy
import logging
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.pipelining import PipelineStageInfo

from torchtitan.components.optim.optimizer import AdamW, OptimizersContainer
from torchtitan.distributed.activation_checkpoint import paged_stash_regions, PagedAC
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    reset_cuda_graphs,
    wrap_fwd_bwd_with_cuda_graph,
)
from torchtitan.distributed.deepep import hybridep
from torchtitan.distributed.deepep.paged_stash.config import PagedStashConfig
from torchtitan.distributed.deepep.paged_stash.eager import EagerPagedStash
from torchtitan.distributed.deepep.paged_stash.runtime import (
    PagedStashManager,
    PagedStashRunner,
    warmup_paged_stash_kernels,
)
from torchtitan.models.base.module import Module

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged stash requires CUDA"
)

NUM_TOKENS = 4096
CAPACITY = 4096  # padded rows of every dispatch
DIM = 256
FFN_DIM = 512
NUM_EXPERTS = 4
NUM_LAYERS = 3
# Live rows per step. The first, the measure step, sizes the stash for the
# most rows any step routes, so that no later step overflows it.
LIVE_PER_STEP = (3584, 2048, 3072, 1024, 2560)
BF16 = torch.bfloat16
E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


# ---------------------------------------------------------------------------
# A toy static-capacity dispatch
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "paged_stash_test::dispatch", mutates_args=(), device_types="cuda"
)
def _dispatch(
    x_TD: torch.Tensor, rows_R: torch.Tensor, tokens_per_expert_E: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather tokens into capacity rows, grouped by expert: row ``r`` holds
    token ``rows_R[r]``, or zeros where it is ``NUM_TOKENS`` (padding)."""
    padded_TD = torch.cat([x_TD, x_TD.new_zeros(1, x_TD.shape[1])])
    routed_RD = padded_TD[rows_R]
    tokens_E = tokens_per_expert_E.clone()
    for observer in hybridep._dispatch_observers:
        observer(routed_RD, tokens_E)
    return routed_RD, tokens_E


@_dispatch.register_fake
def _dispatch_fake(x_TD, rows_R, tokens_per_expert_E):
    return x_TD.new_empty(rows_R.shape[0], x_TD.shape[1]), torch.empty_like(
        tokens_per_expert_E
    )


def _dispatch_setup_context(ctx, inputs, output):
    x_TD, rows_R, _ = inputs
    ctx.save_for_backward(rows_R)
    ctx.num_tokens = x_TD.shape[0]


def _dispatch_backward(ctx, grad_routed_RD, grad_tokens_E):
    (rows_R,) = ctx.saved_tensors
    grad_TD = grad_routed_RD.new_zeros(ctx.num_tokens + 1, grad_routed_RD.shape[1])
    # Every live row adds to a distinct token, so the result is exact; padding
    # rows all add to the dropped last row.
    grad_TD.index_add_(0, rows_R, grad_routed_RD)
    return grad_TD[:-1], None, None


_dispatch.register_autograd(_dispatch_backward, setup_context=_dispatch_setup_context)


@torch.library.custom_op(
    "paged_stash_test::combine", mutates_args=(), device_types="cuda"
)
def _combine(y_RD: torch.Tensor, rows_R: torch.Tensor, num_tokens: int) -> torch.Tensor:
    out_TD = y_RD.new_zeros(num_tokens + 1, y_RD.shape[1])
    out_TD.index_add_(0, rows_R, y_RD)
    return out_TD[:-1].clone()


@_combine.register_fake
def _combine_fake(y_RD, rows_R, num_tokens):
    return y_RD.new_empty(num_tokens, y_RD.shape[1])


def _combine_setup_context(ctx, inputs, output):
    _, rows_R, _ = inputs
    ctx.save_for_backward(rows_R)


def _combine_backward(ctx, grad_TD):
    (rows_R,) = ctx.saved_tensors
    padded_TD = torch.cat([grad_TD, grad_TD.new_zeros(1, grad_TD.shape[1])])
    return padded_TD[rows_R], None, None


_combine.register_autograd(_combine_backward, setup_context=_combine_setup_context)


class _MXFP8LikeSave(torch.autograd.Function):
    """Saves what an MXFP8 expert GEMM saves for its weight gradient: float8
    rows and their e8m0 block scales, one per 32 rows, as a flat buffer.

    The float8 rows come from a column-major cast, as a columnwise quantizer
    emits them, made token-major for the save: under selective checkpointing
    the cast's output is a capacity-padded tensor only checkpointing keeps,
    which the stash cannot page and which must not stay resident.
    """

    @staticmethod
    def forward(ctx, x_RD):
        col_major_RD = torch.empty_strided(
            x_RD.shape, (1, x_RD.shape[0]), dtype=E4M3, device=x_RD.device
        )
        col_major_RD.copy_(x_RD)
        data_RD = col_major_RD.contiguous()
        exponents = (x_RD.float().abs().amax(dim=-1) * 7).to(torch.uint8)
        scales = exponents.view(-1, 32).amax(dim=1, keepdim=True)
        scales = scales.expand(-1, x_RD.shape[1]).contiguous().view(E8M0)
        ctx.save_for_backward(data_RD, scales.reshape(-1))
        return x_RD.clone()

    @staticmethod
    def backward(ctx, grad_RD):
        data_RD, flat_scales = ctx.saved_tensors
        scales_RD = (
            flat_scales.view(torch.uint8)
            .view(-1, grad_RD.shape[1])
            .repeat_interleave(32, dim=0)
            .to(grad_RD.dtype)
        )
        return grad_RD * data_RD.to(grad_RD.dtype) + scales_RD


class ToyRoutedExperts(Module):
    """The paged region: a static-capacity dispatch, grouped GEMMs, a combine.

    Every step of the body is a torch_remat region, as the real
    ``RoutedExperts`` declares them, so ``PagedAC`` can own the subtree and
    ``RegionAC`` can recompute parts of it when paging does not.
    """

    def __init__(self, *, mxfp8_saves: bool) -> None:
        super().__init__()
        self.w13 = nn.Parameter(torch.randn(NUM_EXPERTS, 2 * FFN_DIM, DIM) / DIM**0.5)
        self.w2 = nn.Parameter(torch.randn(NUM_EXPERTS, DIM, FFN_DIM) / FFN_DIM**0.5)
        self.mxfp8_saves = mxfp8_saves

    def _region(self, local_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        return remat.region(
            fn,
            self.remat_region_name(local_name),
            recompute=self.remat_should_recompute(local_name),
        )

    def _w13_mm(self, routed_RD, offsets_E):
        return torch._grouped_mm(
            routed_RD, self.w13.transpose(-2, -1), offs=offsets_E
        ).view(-1, 2, FFN_DIM)

    @staticmethod
    def _activation(gate_up_R2F):
        # Two views of one storage, both saved: one stash.
        gate_RF, up_RF = gate_up_R2F.unbind(dim=1)
        return F.silu(gate_RF) * up_RF

    def _w2_mm(self, hidden_RF, offsets_E):
        return torch._grouped_mm(hidden_RF, self.w2.transpose(-2, -1), offs=offsets_E)

    def forward(self, x_TD, rows_R, tokens_per_expert_E):
        routed_RD, tokens_E = self._region(
            "dispatch", torch.ops.paged_stash_test.dispatch
        )(x_TD, rows_R, tokens_per_expert_E)
        if self.mxfp8_saves:
            routed_RD = self._region("mxfp8", _MXFP8LikeSave.apply)(routed_RD)
        # Bare ops read both dispatch outputs, so pin them for a recompute
        # that skips the region that produced them.
        remat.recompute_needs_tensor(routed_RD)
        remat.recompute_needs_tensor(tokens_E)
        offsets_E = torch.cumsum(tokens_E, dim=0, dtype=torch.int32)
        gate_up_R2F = self._region("w13", self._w13_mm)(routed_RD, offsets_E)
        hidden_RF = self._region("activation", self._activation)(gate_up_R2F)
        out_RD = self._region("w2", self._w2_mm)(hidden_RF, offsets_E)
        return self._region("combine", torch.ops.paged_stash_test.combine)(
            out_RD, rows_R, x_TD.shape[0]
        )


class ToyMoE(Module):
    def __init__(self, *, mxfp8_saves: bool) -> None:
        super().__init__()
        self.routed_experts = ToyRoutedExperts(mxfp8_saves=mxfp8_saves)

    def forward(self, x_TD, rows_R, tokens_per_expert_E):
        out_TD = self.routed_experts(x_TD, rows_R, tokens_per_expert_E)
        # The block's residual add reads this with a bare op.
        remat.recompute_needs_tensor(out_TD)
        return out_TD


class ToyBlock(Module):
    def __init__(self, *, shared_after: bool, mxfp8_saves: bool) -> None:
        super().__init__()
        self.attention = nn.Linear(DIM, DIM, bias=False)
        self.moe = ToyMoE(mxfp8_saves=mxfp8_saves)
        # Saves for backward after the region, so a checkpointed block's
        # recompute has to replay the region to reach it -- as the shared
        # experts that follow the routed experts in MoE.forward do.
        self.shared = nn.Linear(DIM, DIM, bias=False) if shared_after else None

    def forward(self, x_TD, rows_R, tokens_per_expert_E):
        h_TD = x_TD + torch.tanh(self.attention(x_TD))
        out_TD = h_TD + self.moe(h_TD, rows_R, tokens_per_expert_E)
        if self.shared is not None:
            out_TD = out_TD + torch.tanh(self.shared(h_TD))
        return out_TD


class ToyModel(nn.Module):
    def __init__(self, *, shared_after: bool = True, mxfp8_saves: bool = False):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): ToyBlock(shared_after=shared_after, mxfp8_saves=mxfp8_saves)
                for i in range(NUM_LAYERS)
            }
        )

    def forward(self, x_TD, routes):
        for layer_id, layer in self.layers.items():
            x_TD = layer(x_TD, *routes[int(layer_id)])
        return x_TD


def _routes(live: int, generator: torch.Generator) -> list[tuple[torch.Tensor, ...]]:
    """Each layer's dispatch: ``live`` of ``CAPACITY`` rows, in equal expert
    groups, each holding a distinct token."""
    routes = []
    for _ in range(NUM_LAYERS):
        rows_R = torch.full((CAPACITY,), NUM_TOKENS, dtype=torch.int64)
        rows_R[:live] = torch.randperm(NUM_TOKENS, generator=generator)[:live]
        tokens_E = torch.full((NUM_EXPERTS,), live // NUM_EXPERTS, dtype=torch.int64)
        routes.append((rows_R.cuda(), tokens_E.cuda()))
    return routes


def _inputs(
    live: int, seed: int
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, ...]]]:
    generator = torch.Generator().manual_seed(seed)
    x_TD = torch.randn(NUM_TOKENS, DIM, generator=generator).to(BF16).cuda()
    return x_TD, _routes(live, generator)


def _fwd_bwd(model: nn.Module, x_TD, routes) -> torch.Tensor:
    x_TD = x_TD.detach().requires_grad_()
    loss = model(x_TD, routes).float().pow(2).mean()
    loss.backward()
    return loss.detach()


# The region policy *outside* the paged region. Inside it PagedAC owns
# everything, so a recompute pattern that falls there is dropped -- which is
# what lets SelectiveAC's own default policy compose with paging at all.
AC_POLICIES: dict[str, dict[str, list[str]]] = {
    "retain_all": {"save_regions": ["*"], "recompute_regions": []},
    "recompute_outside": {
        "save_regions": ["*routed_experts*"],
        "recompute_regions": [],
    },
    "selective": {
        "save_regions": ["*"],
        "recompute_regions": ["*routed_experts.w13*"],
    },
}


def _paged_ac(ac: str, **paged_stash: Any) -> PagedAC.Config:
    return PagedAC.Config(
        **AC_POLICIES[ac],
        paged_stash=PagedStashConfig(**paged_stash),
    )


def _model_pair(
    ac: str = "retain_all", **kwargs: Any
) -> tuple[nn.Module, nn.Module, PagedAC.Config]:
    """A model and an identical copy under the same ``PagedAC`` policy.

    Both sides get the same checkpointing, so the only difference between them
    is that the caller attaches a paging frontend to the first.
    """
    torch.manual_seed(0)
    model = ToyModel(**kwargs).to(device="cuda", dtype=BF16)
    reference = copy.deepcopy(model)
    ac_config = _paged_ac(ac)
    ac_config.build().apply(model)
    ac_config.build().apply(reference)
    return model, reference, ac_config


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        grads[name] = param.grad.clone()
        param.grad = None
    return grads


def _assert_same_step(paged, reference) -> None:
    (paged_loss, paged_grads), (ref_loss, ref_grads) = paged, reference
    assert torch.equal(paged_loss, ref_loss), (paged_loss, ref_loss)
    assert paged_grads.keys() == ref_grads.keys()
    for name in ref_grads:
        assert torch.equal(paged_grads[name], ref_grads[name]), name


@pytest.fixture
def paged_stash():
    """Builds the frontend over a fresh manager; closes it afterwards."""
    frontends = []

    def build(
        model: nn.Module, ac_config: PagedAC.Config | None = None, **paged_stash: Any
    ) -> EagerPagedStash:
        if ac_config is None:
            ac_config = _paged_ac("retain_all", **paged_stash)
        elif paged_stash:
            ac_config = PagedAC.Config(
                save_regions=list(ac_config.save_regions),
                recompute_regions=list(ac_config.recompute_regions),
                page_regions=list(ac_config.page_regions),
                paged_stash=PagedStashConfig(**paged_stash),
            )
        frontend = EagerPagedStash(
            ac_config,
            model_parts=[model],
            pp_schedule=None,
            device=torch.device("cuda"),
        )
        frontends.append(frontend)
        return frontend

    PagedStashManager.reset_instance()
    yield build
    for frontend in frontends:
        frontend.close()
    cuda_graph_teardown()
    PagedStashManager.reset_instance()


def _paged_step(frontend: EagerPagedStash, fn: Callable[..., Any], *args) -> Any:
    """One step as the runner runs it."""
    frontend.manager.begin_step()
    warmup_paged_stash_kernels()
    return frontend.run_step(fn, *args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_page_regions_name_a_region_root():
    """A pattern names the root of a region; the search does not descend into
    one, so the default does not also match ``routed_experts.w13``."""
    model = ToyModel().to(device="cuda", dtype=BF16)
    regions = paged_stash_regions(model, PagedAC.Config().page_regions)
    assert sorted(regions) == list(range(NUM_LAYERS))
    assert regions[0][0] == "layers.0.moe.routed_experts"


@requires_cuda
def test_paging_owns_its_region_against_a_recompute_pattern():
    """A recompute pattern falling inside a paged region is dropped.

    This is what lets a SelectiveAC-style policy compose with paging at all:
    its ``*routed_experts.w13*`` recompute pattern would otherwise recompute
    the very activation paging exists to hold. Ownership is scoped to the
    paged region -- the same pattern list still recomputes outside it.
    """
    model = ToyModel().to(device="cuda", dtype=BF16)
    policy: dict[str, list[str]] = {
        "save_regions": ["*"],
        "recompute_regions": ["*routed_experts.w13*", "*attention*"],
    }
    PagedAC.Config(**policy).build().apply(model)
    block = model.layers["0"]
    assert not block.moe.routed_experts.remat_should_recompute("w13")
    assert block.remat_should_recompute("attention")


@requires_cuda
@pytest.mark.parametrize("shared_after", [True, False])
@pytest.mark.parametrize("ac", list(AC_POLICIES))
def test_paged_steps_match_unpaged(paged_stash, ac, shared_after, caplog):
    caplog.set_level(logging.INFO, logger=EagerPagedStash.__module__)
    model, reference, ac_config = _model_pair(ac, shared_after=shared_after)
    frontend = paged_stash(model, ac_config)
    for step, live in enumerate(LIVE_PER_STEP):
        x_TD, routes = _inputs(live, seed=step)
        paged = (_paged_step(frontend, _fwd_bwd, model, x_TD, routes), _grads(model))
        expected = (_fwd_bwd(reference, x_TD, routes), _grads(reference))
        _assert_same_step(paged, expected)
        if step == 0:
            assert frontend.manager.measuring
        else:
            assert frontend.manager.active
        # Per layer: the dispatch output, the fused gate/up output, the
        # activation's output and the down projection's input. Checkpointing
        # also keeps the down projection's output, which its replay never
        # reads, so it is dropped rather than stashed.
        assert frontend._stashes == 4 * NUM_LAYERS
    assert "reloads of the first paging step matched their stashes" in caplog.text


@requires_cuda
@pytest.mark.parametrize("ac", list(AC_POLICIES))
def test_cuda_graph_replays_other_live_counts(paged_stash, ac):
    model, reference, ac_config = _model_pair(ac)
    frontend = paged_stash(model, ac_config)
    run_paged = wrap_fwd_bwd_with_cuda_graph(
        lambda x_TD, routes: _fwd_bwd(model, x_TD, routes),
        parameters=model.parameters(),
        num_warmup_iterations=2,
    )
    run_reference = wrap_fwd_bwd_with_cuda_graph(
        lambda x_TD, routes: _fwd_bwd(reference, x_TD, routes),
        parameters=reference.parameters(),
        num_warmup_iterations=2,
    )
    # Two eager warmup steps -- the measure step and the first paging step --
    # then a capture, then replays.
    for step, live in enumerate(LIVE_PER_STEP + (512, 3584)):
        x_TD, routes = _inputs(live, seed=step)
        paged = (_paged_step(frontend, run_paged, x_TD, routes), _grads(model))
        expected = (run_reference(x_TD, routes), _grads(reference))
        _assert_same_step(paged, expected)


@requires_cuda
@pytest.mark.parametrize("ac", list(AC_POLICIES))
def test_mxfp8_block_scales_are_paged(paged_stash, ac, caplog):
    model, reference, ac_config = _model_pair(ac, mxfp8_saves=True)
    frontend = paged_stash(model, ac_config)
    for step, live in enumerate(LIVE_PER_STEP):
        x_TD, routes = _inputs(live, seed=step)
        paged = (_paged_step(frontend, _fwd_bwd, model, x_TD, routes), _grads(model))
        expected = (_fwd_bwd(reference, x_TD, routes), _grads(reference))
        _assert_same_step(paged, expected)
    slots = frontend.manager.slots.values()
    scale_slots = [slot for slot in slots if slot.dtype is E8M0]
    assert len(scale_slots) == NUM_LAYERS
    assert all(slot.token_divisor == 32 for slot in scale_slots)
    assert "cannot hold" not in caplog.text


@requires_cuda
@pytest.mark.parametrize("ac", list(AC_POLICIES))
def test_pipeline_metadata_inference_keeps_saves_resident(paged_stash, ac):
    model, reference, ac_config = _model_pair(ac)
    frontend = paged_stash(model, ac_config)
    x_TD, routes = _inputs(LIVE_PER_STEP[0], seed=0)
    info = PipelineStageInfo(
        stage_index=0, microbatch_index=0, is_metadata_inference=True
    )
    with frontend._stage_forward(info):
        loss = _fwd_bwd(model, x_TD, routes)
    _assert_same_step(
        (loss, _grads(model)),
        (_fwd_bwd(reference, x_TD, routes), _grads(reference)),
    )
    assert frontend._stashes == 0 and not frontend.manager.slots


@requires_cuda
def test_schedule_change_after_measure_step_raises(paged_stash):
    model, _, _ = _model_pair()
    frontend = paged_stash(model)

    def two_microbatches(x_TD, routes):
        _fwd_bwd(model, x_TD, routes)
        _fwd_bwd(model, x_TD, routes)

    x_TD, routes = _inputs(LIVE_PER_STEP[0], seed=0)
    _paged_step(frontend, two_microbatches, x_TD, routes)
    with pytest.raises(RuntimeError, match="different order"):
        _paged_step(frontend, _fwd_bwd, model, x_TD, routes)


@requires_cuda
def test_one_region_per_layer(paged_stash):
    model, _, _ = _model_pair()
    two_per_block = PagedAC.Config(page_regions=["attention", "shared"])
    with pytest.raises(ValueError, match="at most one per layer"):
        paged_stash(model, two_per_block)
    with pytest.raises(ValueError, match="matches no module"):
        paged_stash(model, PagedAC.Config(page_regions=["*shared_experts*"]))


@requires_cuda
def test_blocking_overflow_reruns_the_step_unpaged(paged_stash, caplog):
    caplog.set_level(logging.INFO)
    model, reference, _ = _model_pair()
    frontend = paged_stash(model, overflow_check="blocking")
    optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*")]
    ).build(model_parts=[model])
    runner = PagedStashRunner(
        optimizers, overflow_check="blocking", reset_cuda_graphs=reset_cuda_graphs
    )
    ran: list[bool] = []  # per attempt, whether it paged; none when replayed

    def fwd_bwd(x_TD, routes):
        ran.append(frontend.manager.active)
        return _fwd_bwd(model, x_TD, routes)

    run_paged = wrap_fwd_bwd_with_cuda_graph(
        fwd_bwd, parameters=model.parameters(), num_warmup_iterations=2
    )
    run_reference = wrap_fwd_bwd_with_cuda_graph(
        lambda x_TD, routes: _fwd_bwd(reference, x_TD, routes),
        parameters=reference.parameters(),
        num_warmup_iterations=2,
    )
    # Measure, page eagerly, capture, replay; outgrow the stash in a replay,
    # which reruns eagerly and measures; then page and capture again over
    # buffers that fit the step that overflowed, which no longer overflows.
    live_per_step = (2048, 2048, 1024, 2048, 3584, 2048, 1024, 3584)
    attempts = ([False], [True], [True], [], [False], [True], [], [])
    for step, live in enumerate(live_per_step):
        ran.clear()
        optimizers.zero_grad(set_to_none=True)
        x_TD, routes = _inputs(live, seed=step)
        paged = (
            runner(lambda: frontend.run_step(run_paged, x_TD, routes)),
            _grads(model),
        )
        expected = (run_reference(x_TD, routes), _grads(reference))
        _assert_same_step(paged, expected)
        assert ran == attempts[step], (step, ran)
    assert caplog.text.count("rerunning forward-backward") == 1
