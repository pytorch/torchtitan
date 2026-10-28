# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp64 parity of absorbed router scores through the DeepEP dispatcher, with stubbed kernels."""

import importlib
import sys
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import DeepEPTokenDispatcher

_MODEL_DIM = 8
_NUM_TOKENS = 64
_TOP_K = 2


class _FakeEvent:
    """No-op stand-in for deep_ep's async combine event."""

    def current_stream_wait(self) -> None:
        pass


class _FakeHandle:
    """Routing layout of one dispatch, computed identically on every rank."""

    def __init__(self, per_rank_recv, num_recv_tokens_per_expert_list, num_tokens):
        self.per_rank_recv = per_rank_recv
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.num_tokens = num_tokens
        self.num_sms = 0


def _all_gather_rows(rows: torch.Tensor, group) -> list[torch.Tensor]:
    """all_gather a tensor whose leading dimension varies per rank."""
    num_ranks = dist.get_world_size(group)
    count = torch.tensor([rows.shape[0]], device=rows.device)
    counts = [torch.empty_like(count) for _ in range(num_ranks)]
    dist.all_gather(counts, count, group=group)
    max_rows = max(int(c.item()) for c in counts)
    padded = rows.new_zeros(max_rows, *rows.shape[1:])
    padded[: rows.shape[0]] = rows
    gathered = [torch.empty_like(padded) for _ in range(num_ranks)]
    dist.all_gather(gathered, padded, group=group)
    return [g[: int(c.item())] for g, c in zip(gathered, counts)]


class _FakeElasticBuffer:
    """Pure-PyTorch stand-in for deep_ep.ElasticBuffer, compact layout only."""

    def __init__(
        self,
        group,
        *,
        num_bytes,
        num_max_tokens_per_rank,
        hidden,
        num_topk,
        use_fp8_dispatch=False,
        deterministic=False,
        explicitly_destroy=False,
    ):
        self.group = group
        self.num_bytes = num_bytes
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.deterministic = deterministic

    @staticmethod
    def get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden, *, num_topk, use_fp8_dispatch=False
    ):
        return 1

    def get_theoretical_num_sms(self, num_experts, num_topk):
        return 1

    def dispatch(
        self,
        x,
        *,
        topk_idx=None,
        topk_weights=None,
        num_experts=None,
        num_max_tokens_per_rank=None,
        num_sms=0,
        handle=None,
        do_expand=False,
        do_cpu_sync=True,
    ):
        if handle is not None:
            return self._dispatch_from_handle(x, handle)
        assert not do_expand, "the stub implements the compact layout only"
        return self._dispatch_fresh(x, topk_idx, topk_weights, num_experts)

    def _dispatch_fresh(self, x, topk_idx, topk_weights, num_experts):
        num_ranks = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        num_local_experts = num_experts // num_ranks
        xs = _all_gather_rows(x, self.group)
        idxs = _all_gather_rows(topk_idx, self.group)
        ws = _all_gather_rows(topk_weights, self.group)

        per_rank_recv = []
        recv_x, recv_idx, recv_scores = [], [], []
        for r in range(num_ranks):
            lo = r * num_local_experts
            src_ranks, src_tokens = [], []
            for s in range(num_ranks):
                # -1 picks and other ranks' experts fall outside [lo, lo + num_local_experts).
                is_local = (idxs[s] >= lo) & (idxs[s] < lo + num_local_experts)
                keep = is_local.any(dim=1)
                src_ranks.append(
                    torch.full((int(keep.sum()),), s, dtype=torch.long, device=x.device)
                )
                src_tokens.append(keep.nonzero(as_tuple=False).squeeze(-1))
                if r == rank:
                    recv_x.append(xs[s][keep])
                    recv_idx.append(torch.where(is_local[keep], idxs[s][keep] - lo, -1))
                    recv_scores.append(ws[s][keep])
            per_rank_recv.append((torch.cat(src_ranks), torch.cat(src_tokens)))
        recv_idx = torch.cat(recv_idx)
        handle = _FakeHandle(
            per_rank_recv=per_rank_recv,
            num_recv_tokens_per_expert_list=torch.bincount(
                recv_idx[recv_idx >= 0], minlength=num_local_experts
            ).tolist(),
            num_tokens=x.shape[0],
        )
        return (
            torch.cat(recv_x),
            recv_idx,
            torch.cat(recv_scores),
            handle,
            _FakeEvent(),
        )

    def _dispatch_from_handle(self, grad_TD, handle):
        # Combine backward: a received row's gradient is its owner token's row.
        num_ranks = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        grads = _all_gather_rows(grad_TD, self.group)
        src_rank_R, src_token_R = handle.per_rank_recv[rank]
        grad_rows = grad_TD.new_zeros(src_rank_R.numel(), grad_TD.shape[1])
        for s in range(num_ranks):
            from_s = src_rank_R == s
            if bool(from_s.any()):
                grad_rows[from_s] = grads[s][src_token_R[from_s]]
        empty = grad_rows.new_zeros(0)
        return grad_rows, empty.long(), empty, handle, _FakeEvent()

    def combine(self, x, *, handle, topk_weights=None, async_with_compute_stream=False):
        # Forward combine and dispatch backward: each owner sums its rows across ranks.
        num_ranks = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        xs = _all_gather_rows(x, self.group)
        ws = (
            None if topk_weights is None else _all_gather_rows(topk_weights, self.group)
        )
        out = x.new_zeros(handle.num_tokens, x.shape[1])
        out_weights = (
            None
            if ws is None
            else topk_weights.new_zeros(handle.num_tokens, topk_weights.shape[1])
        )
        for r in range(num_ranks):
            src_rank_R, src_token_R = handle.per_rank_recv[r]
            mine = src_rank_R == rank
            if bool(mine.any()):
                out.index_add_(0, src_token_R[mine], xs[r][mine])
                if out_weights is not None:
                    out_weights.index_add_(0, src_token_R[mine], ws[r][mine])
        return out, out_weights, _FakeEvent()


def _import_wrapper_with_stub():
    """Import the deepep wrapper against the stub when deep_ep is absent."""
    try:
        import deep_ep  # noqa: F401
    except ImportError:
        stub = types.ModuleType("deep_ep")
        stub.ElasticBuffer = _FakeElasticBuffer
        sys.modules["deep_ep"] = stub
        injected = True
    else:
        injected = False
    wrapper = importlib.import_module("torchtitan.distributed.deepep.deepep")
    if injected:
        # Sibling tests' importorskip("deep_ep") must still skip honestly.
        del sys.modules["deep_ep"]
    return wrapper


_deepep_wrapper = _import_wrapper_with_stub()
# The wrapper registers its impls for CUDA only; register them for CPU too.
_cpu_impl_library = torch.library.Library("deepep", "FRAGMENT")
_cpu_impl_library.impl("dispatch", _deepep_wrapper._dispatch_op_impl, "CPU")
_cpu_impl_library.impl("combine", _deepep_wrapper._combine_op_impl, "CPU")


class _ExactGroupedLinear(GroupedLinear):
    """GroupedLinear whose GEMM runs in the weight dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
        input_RI = input_RI.to(weight_EOI.dtype)
        starts = [0, *offsets_E[:-1].tolist()]
        return torch.cat(
            [
                input_RI[start:end] @ weight_EOI[expert].transpose(-2, -1)
                for expert, (start, end) in enumerate(zip(starts, offsets_E.tolist()))
            ],
            dim=0,
        )


class _DeepEPBlock(Module):
    """One routed-experts block behind a DeepEP dispatcher."""

    def __init__(self, num_experts: int, *, absorb_router_scores: bool):
        super().__init__()
        # Build the EP-local expert view directly; the dispatcher keeps the global E.
        num_local_experts = num_experts // dist.get_world_size()
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.w13 = _ExactGroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
            num_linears=2,
        ).build()
        routed_experts.w2 = _ExactGroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
        ).build()
        routed_experts.activation_fn = SwiGLU.Config().build()
        routed_experts.output_postprocess = None
        routed_experts.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=_TOP_K,
            hidden_dim=_MODEL_DIM,
            num_max_tokens_per_rank=_NUM_TOKENS,
            absorb_router_scores=absorb_router_scores,
        ).build()
        self.gate = torch.nn.Linear(_MODEL_DIM, num_experts, bias=False)
        with torch.no_grad():
            for parameter in routed_experts.parameters():
                parameter.normal_(std=0.5)
        self.routed_experts = routed_experts
        self.num_experts = num_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = torch.arange(_TOP_K, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        # Scores depend on the gate, so a misplaced score moves the gradients.
        scores_TK = torch.sigmoid(self.gate(x_TD)).gather(1, expert_ids_TK)
        routing_map_TE = torch.nn.functional.one_hot(
            expert_ids_TK, num_classes=self.num_experts
        ).any(dim=1)
        out_TD = self.routed_experts(
            x_TD,
            scores_TK,
            expert_ids_TK,
            routing_map_TE,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD.square().sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _forward_backward(
    model: Module,
    x_TD: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run one forward/backward; return the loss and input/parameter grads."""
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    loss = model(input_TD)
    loss.backward()
    assert input_TD.grad is not None
    grads = [input_TD.grad.detach().clone()]
    grads += [parameter.grad.detach().clone() for parameter in model.parameters()]
    return loss.detach().clone(), grads


class TestDeepEPAbsorbRouterScoresParity(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_absorbed_equals_post_combine(self):
        # Force the stub buffer even when the real deep_ep is importable.
        saved = (_deepep_wrapper.ElasticBuffer, _deepep_wrapper._buffer)
        _deepep_wrapper.ElasticBuffer = _FakeElasticBuffer
        _deepep_wrapper._buffer = None
        try:
            self._check_absorb_parity()
        finally:
            _deepep_wrapper.ElasticBuffer, _deepep_wrapper._buffer = saved

    def _check_absorb_parity(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("ep",),
        )
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=mesh,
            dense_sp_enabled=False,
        )
        num_experts = 4 * self.world_size
        with (
            torch.autograd.set_multithreading_enabled(False),
            set_current_spmd_mesh(mesh),
        ):
            torch.manual_seed(42)
            baseline = _Model(_DeepEPBlock(num_experts, absorb_router_scores=False)).to(
                self.device_type, torch.float64
            )
            absorbed = _Model(_DeepEPBlock(num_experts, absorb_router_scores=True)).to(
                self.device_type, torch.float64
            )
            absorbed.load_state_dict(baseline.state_dict())
            # Both models share one stub buffer.
            baseline.layers["0"].routed_experts.token_dispatcher.init_buffer()
            absorbed.layers["0"].routed_experts.token_dispatcher.init_buffer()

            torch.manual_seed(self.rank)
            x_TD = torch.randn(
                _NUM_TOKENS,
                _MODEL_DIM,
                device=self.device_type,
                dtype=torch.float64,
            )
            expected_loss, expected = _forward_backward(baseline, x_TD)
            actual_loss, actual = _forward_backward(absorbed, x_TD)

            # Scores are applied in float32, so compare at float32 scale.
            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
            for actual_grad, expected_grad in zip(actual, expected):
                torch.testing.assert_close(
                    actual_grad, expected_grad, rtol=1e-5, atol=1e-6
                )
