# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import InferenceMode
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.tools.logging import logger

_UPSTREAM_FAKE_PG_MARKER = (
    "Fake process group detected; set inference_mode=static "
    "for %d stage(s) without voting"
)


def ensure_fake_pg_static_metadata_support() -> None:
    """Backport fake-PG static metadata handling to older PyTorch builds."""
    original_warmup_p2p = _PipelineSchedule._warmup_p2p
    if getattr(original_warmup_p2p, "_torchtitan_fake_pg_backport", False):
        return

    # The upstream fix has no API marker. Its unique log constant lets this
    # compatibility shim become inert once that implementation is installed.
    code = getattr(original_warmup_p2p, "__code__", None)
    if code is not None and _UPSTREAM_FAKE_PG_MARKER in code.co_consts:
        return

    def _warmup_p2p(self, stages, has_backward, p2p_done):
        if all(isinstance(stage, PipelineStage) for stage in stages):
            has_cross_rank = any(
                (not stage.is_first and not stage._is_same_rank(stage.stage_index - 1))
                or (
                    not stage.is_last and not stage._is_same_rank(stage.stage_index + 1)
                )
                for stage in stages
            )
            if has_cross_rank and any(
                dist.get_backend(stage.group) == "fake" for stage in stages
            ):
                for stage in stages:
                    if InferenceMode.needs_dynamic(stage._user_meta, has_backward):
                        raise RuntimeError(
                            f"Stage {stage.stage_index} requires dynamic shape "
                            "inference, which is not supported with a fake "
                            "process group. Provide complete static metadata "
                            "(inputs/outputs, plus input_grads/output_grads "
                            "for DTensors with backward) to the PipelineStage "
                            "constructor."
                        )
                    stage._inference_mode = InferenceMode.STATIC
                logger.debug(_UPSTREAM_FAKE_PG_MARKER, len(stages))
                return
        return original_warmup_p2p(self, stages, has_backward, p2p_done)

    setattr(_warmup_p2p, "_torchtitan_fake_pg_backport", True)
    _PipelineSchedule._warmup_p2p = _warmup_p2p
