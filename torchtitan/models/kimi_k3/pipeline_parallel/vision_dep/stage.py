# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline stage that runs the decoupled encoder process's work around its actions."""

from __future__ import annotations

from typing import Any

from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import _batch_p2p

from .plan import Anchor
from .runtime import VisionDep


class VisionDepPipelineStage(PipelineStage):
    """Pipeline stage that gives stage 0 the encoded features and runs the vision work
    and transfers planned around each of its actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dep: VisionDep | None = None

    def set_vision_dep(self, dep: VisionDep) -> None:
        self._dep = dep

    def _sends_to(self, stage: int) -> bool:
        return (
            0 <= stage < self.num_stages
            and self.stage_index_to_group_rank[stage] != self.group_rank
        )

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        mb = int(fwd_chunk_id)
        anchor = ("F", self.stage_index, mb)
        if self._dep is not None:
            self._dep.before_action(anchor)
            if self.is_first:
                kwargs = self._dep.stage0_kwargs(mb, kwargs)
        output = super().forward_one_chunk(
            fwd_chunk_id, args, kwargs, save_forward_output
        )
        if self._dep is not None:
            self._dep.after_action(
                anchor, sends_follow=self._sends_to(self.stage_index + 1)
            )
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        mb = int(bwd_chunk_id)
        anchor = ("B" if full_backward else "I", self.stage_index, mb)
        if self._dep is not None:
            self._dep.before_action(anchor)
        super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )
        if self._dep is None:
            return
        if self.is_first:
            self._dep.after_stage0_backward(mb)
        self._dep.after_action(
            anchor, sends_follow=self._sends_to(self.stage_index - 1)
        )

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        anchor = ("W", self.stage_index, int(bwd_chunk_id))
        if self._dep is not None:
            self._dep.before_action(anchor)
        super().backward_weight_one_chunk(bwd_chunk_id, last_backward=last_backward)
        if self._dep is not None:
            self._dep.after_action(anchor, sends_follow=False)

    def _send_then_run(self, ops: list, anchors: tuple[Anchor, ...]) -> list:
        dep = self._dep
        held = [a for a in anchors if dep is not None and dep.holds_for_send(a)]
        if not held:
            return ops
        assert dep is not None
        # Work queued on the stream ahead of a send would hold the send back.
        works = _batch_p2p(ops)
        for anchor in held:
            dep.after_send(anchor, works, ops)
        return []

    def get_fwd_send_ops(self, fwd_chunk_id: int):
        ops = super().get_fwd_send_ops(fwd_chunk_id)
        return self._send_then_run(ops, (("F", self.stage_index, int(fwd_chunk_id)),))

    def get_bwd_send_ops(self, bwd_chunk_id: int):
        ops = super().get_bwd_send_ops(bwd_chunk_id)
        return self._send_then_run(
            ops,
            (
                ("B", self.stage_index, int(bwd_chunk_id)),
                ("I", self.stage_index, int(bwd_chunk_id)),
            ),
        )
