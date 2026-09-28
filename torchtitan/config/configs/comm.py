# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Communication configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class CommConfig:
    init_timeout_seconds: int = 300
    """Timeout for communication operations, during initialization and first train step."""

    train_timeout_seconds: int = 100
    """
    Timeout for communication operations after the first train step --
    usually a tighter bound than during initialization.
    """

    trace_buf_size: int = 20000
    """Flight recorder ring buffer size, >0 means recording by default, 0 means disabled"""

    save_traces_folder: str = "comm_traces"
    """Flight recorder trace files location"""

    save_traces_file_prefix: str = "rank_"
    """Flight recorder trace files prefix"""

    backend: Literal["default", "fake", "real_pp_fake_spmd"] = "default"
    """Communication topology used for training or distributed debugging.

    Options:
    - ``"default"`` uses real process groups for every configured mesh axis.
    - ``"fake"`` represents PP coordinate ``FAKE_PP_RANK`` and SPMD coordinate
      zero in a completely fake logical mesh. It validates configuration,
      shapes, ownership, and PyTorch-managed memory without real transport.
    - ``"real_pp_fake_spmd"`` runs one physical process per PP rank and
      uses a real NCCL PP group while DP, TP, CP, and EP remain fake. It
      exercises pipeline transport, buffers, and CUDA graphs without allocating
      the complete logical world.

    ``NGPU`` is the complete logical world size. ``FAKE_PP_RANK`` applies only
    to ``"fake"``; ``"real_pp_fake_spmd"`` uses physical ``RANK`` as its PP
    coordinate. See ``docs/debugging.md`` for launch examples and limitations.
    """
