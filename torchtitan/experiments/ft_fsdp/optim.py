# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.components.optim import Optim
from torchtitan.distributed import utils as dist_utils


class NonFiniteStepError(RuntimeError):
    """Loss or gradient norm is not finite; the update was skipped."""


class FTOptim(Optim):
    """Optim that raises on a non-finite step instead of a device assert.

    An aborted collective leaves garbage (often NaN) in its output. The core
    ``torch._assert_async`` check then poisons the CUDA context and forces a
    process restart; raising on the host keeps the process recoverable.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Optim.Config):
        pass

    def _update(self, loss: torch.Tensor) -> torch.Tensor:
        grad_norm = dist_utils.clip_grad_norm_(
            self.parameters,
            self.config.max_norm,
            foreach=True,
            pp_mesh=None,
            ep_enabled=False,
        )
        if not bool(torch.isfinite(loss).all() & torch.isfinite(grad_norm).all()):
            raise NonFiniteStepError(
                f"loss {loss.item()} or grad norm {grad_norm.item()} is not finite"
            )
        self.optimizers.step()
        return grad_norm
