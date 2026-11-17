# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.distributed as dist

from torchtitan.config.configurable import Configurable


class NCCLProcessGroupOptions(Configurable):
    """NCCL communicator options for the process group of one mesh axis."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        min_ctas: int | None = None
        """Lower bound on the CTAs (channels) per collective; None: NCCL's choice."""

        max_ctas: int | None = None
        """Upper bound on the CTAs per collective; None: NCCL's choice."""

    def __init__(self, config: Config) -> None:
        options = dist.ProcessGroupNCCL.Options()
        if config.min_ctas is not None:
            options.config.min_ctas = config.min_ctas
        if config.max_ctas is not None:
            options.config.max_ctas = config.max_ctas
        # The (backend, options) pair DeviceMesh takes for one mesh axis.
        self.backend_override = ("nccl", options)
