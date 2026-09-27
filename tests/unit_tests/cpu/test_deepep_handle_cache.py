# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepEP dispatch handles are freed under FullAC early-stopped recompute and torch.compile."""

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from torchtitan.distributed.activation_checkpoint import _full_ac_policy

deepep = pytest.importorskip(
    "torchtitan.distributed.deepep.deepep", exc_type=ImportError
)

_NUM_TOKENS = 16
_HIDDEN = 8
_NUM_EXPERTS = 4
_TOP_K = 2


class _FakeEvent:
    def current_stream_wait(self):
        pass


class _FakeHandle:
    def __init__(self, topk_idx):
        self.num_recv_tokens_per_expert_list = torch.bincount(
            topk_idx.flatten(), minlength=_NUM_EXPERTS
        ).tolist()
        self.num_sms = 1


class _SingleRankBuffer:
    """ElasticBuffer stand-in for one rank: every token stays local, so dispatch/combine copy."""

    num_max_tokens_per_rank = _NUM_TOKENS

    def get_theoretical_num_sms(self, num_experts, num_topk):
        return 1

    def dispatch(self, x, handle=None, **kwargs):
        if handle is None:
            topk_idx = kwargs["topk_idx"]
            return (
                x.clone(),
                topk_idx.clone(),
                kwargs["topk_weights"].clone(),
                _FakeHandle(topk_idx),
                _FakeEvent(),
            )
        return x.clone(), None, None, handle, _FakeEvent()

    def combine(self, x, handle=None, topk_weights=None, **kwargs):
        return x.clone(), topk_weights, _FakeEvent()


class _MoEBlock(nn.Module):
    """Router -> DeepEP dispatch -> expert -> DeepEP combine -> residual."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(_HIDDEN, _NUM_EXPERTS, bias=False)
        self.expert = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def forward(self, x):
        top_scores, top_expert_ids = torch.softmax(self.router(x), dim=-1).topk(
            _TOP_K, dim=-1
        )
        routed, _, state = deepep.dispatch_tokens(
            x,
            top_expert_ids,
            top_scores,
            _NUM_EXPERTS,
            _NUM_EXPERTS,
            num_tokens_per_rank=_NUM_TOKENS,
            remat_region_name="dispatch",
            recompute=True,
        )
        combined = deepep.combine_tokens(
            self.expert(routed), state, remat_region_name="combine", recompute=True
        )
        deepep.sync_combine()
        return x + combined


@pytest.fixture
def single_rank_deepep():
    """Run the DeepEP ops on CPU against a single-rank buffer."""
    cpu_impls = torch.library.Library("deepep", "IMPL")
    cpu_impls.impl("dispatch", deepep._dispatch_op_impl, "CPU")
    cpu_impls.impl("combine", deepep._combine_op_impl, "CPU")
    original_buffer = deepep._buffer
    deepep._buffer = _SingleRankBuffer()
    deepep._handle_cache.clear()
    yield
    deepep._buffer = original_buffer
    cpu_impls._destroy()


def test_full_ac_recompute_frees_handles(single_rank_deepep):
    # The early-stopped recompute replays dispatch but stops before combine.
    block = ptd_checkpoint_wrapper(
        _MoEBlock(),
        context_fn=lambda: create_selective_checkpoint_contexts(_full_ac_policy),
        early_stop=True,
    )
    for _ in range(2):
        block(torch.randn(_NUM_TOKENS, _HIDDEN, requires_grad=True)).sum().backward()
        assert not deepep._handle_cache


def test_compile_frees_handles(single_rank_deepep):
    # torch.compile graph-breaks on the DeepEP ops; dynamo must not trace their kernels.
    block = torch.compile(_MoEBlock(), backend="aot_eager")
    for _ in range(2):
        block(torch.randn(_NUM_TOKENS, _HIDDEN, requires_grad=True)).sum().backward()
        assert not deepep._handle_cache
